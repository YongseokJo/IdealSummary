import argparse
import json
import os
import multiprocessing as mp
from functools import partial

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
import time
import gc

from data import HDF5SetDataset, hdf5_collate, smf_collate
from mem_debug import log_memory
from deepset import DeepSet, MLP
from setpooling import SlotSetPool
import train as train_mod
from train import compute_stats_from_dataset, normalize_batch, _inverse_output_norm, run_model
from torch.utils.data import Subset
import wandb


def _wandb_log_metrics(run, step, prefix, mse, r2, ep=None, trial=None):
    if run is None:
        return
    payload = {
        f"{prefix}/mse": float(mse),
        f"{prefix}/r2": float(r2),
        "global_step": int(step),
    }
    if ep is not None:
        payload["epoch"] = int(ep)
    if trial is not None:
        payload["trial"] = int(trial)
    run.log(payload, step=step)



def safe_wandb_init(retries: int = 6, base_sleep: float = 2.0, **kwargs):
    """Initialize wandb with retries to avoid project-creation race (HTTP 409).

    Returns the wandb Run object or raises an exception if a non-retryable
    error occurs.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            run = wandb.init(reinit=True, **kwargs)
            return run
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # common W&B race indication strings
            if ("project may already be created" in msg) or ("409" in msg) or ("already exists" in msg):
                sleep = min(base_sleep * (2 ** attempt), 60)
                time.sleep(sleep)
                continue
            # non-retryable
            raise
    # final attempt (let any exception propagate)
    return wandb.init(reinit=True, **kwargs)

@torch.no_grad()
def eval_loader_metrics(model, loader, device, max_batches=None,
                        input_norm: str = "none", input_stats: dict = None,
                        output_norm: str = "none", output_stats: dict = None):
    model.eval()
    y_trues, y_preds = [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        if len(batch) == 3:
            X, mask, y = batch
            X = X.to(device); mask = mask.to(device)
        else:
            X, y = batch
            X = X.to(device)
            mask = None

        # apply input/output normalization
        Xn, maskn, yn = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
        pred = run_model(model, Xn, maskn)

        # inverse-transform predictions back to original units
        pred_orig = _inverse_output_norm(pred, output_norm, output_stats, device)

        y_np = y.cpu().numpy()
        p_np = pred_orig.detach().cpu().numpy()
        y_np = y_np.reshape(y_np.shape[0], -1) if y_np.ndim == 1 else y_np
        p_np = p_np.reshape(p_np.shape[0], -1) if p_np.ndim == 1 else p_np
        y_trues.append(y_np)
        y_preds.append(p_np)

    if len(y_trues) == 0:
        return float("nan"), float("nan")

    y_trues = np.concatenate(y_trues, axis=0)
    y_preds = np.concatenate(y_preds, axis=0)

    mse = float(np.mean((y_trues - y_preds) ** 2))

    K = y_trues.shape[1]
    r2_list = []
    for k in range(K):
        yt = y_trues[:, k]
        yp = y_preds[:, k]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_list.append(r2)
    r2 = float(np.mean(r2_list))

    return mse, r2

def get_data_loaders(h5_path, snap, param_keys, data_field, batch, train_frac, val_frac, test_frac, use_smf):
    ds = HDF5SetDataset(h5_path, snap=snap, param_keys=param_keys, data_field=data_field)
    total = len(ds)

    total_frac = train_frac + val_frac + test_frac
    if total_frac <= 0:
        raise ValueError("train/val/test fractions must sum to a positive value")
    if total_frac > 1.0 + 1e-6:
        raise ValueError("train/val/test fractions must sum to <= 1")

    train_size = int(round(train_frac * total))
    val_size = int(round(val_frac * total))
    test_size = int(round(test_frac * total))

    assigned = train_size + val_size + test_size
    if assigned == 0:
        raise ValueError("Computed zero samples for all splits; adjust fractions")
    if assigned != total:
        train_size = max(1, train_size + (total - assigned))
    # ensure at least one sample for val/test when a non-zero fraction is requested
    if val_frac > 0 and val_size == 0 and total >= 2:
        val_size = 1
        train_size = max(1, train_size - 1)
    if test_frac > 0 and test_size == 0 and total - train_size - val_size >= 1:
        test_size = 1
        train_size = max(1, train_size - 1)

    # final trim to avoid overflow
    total_assigned = train_size + val_size + test_size
    if total_assigned > total:
        overflow = total_assigned - total
        trim_train = min(overflow, max(0, train_size - 1))
        train_size -= trim_train
        overflow -= trim_train
        if overflow > 0 and val_size > 0:
            trim_val = min(overflow, val_size)
            val_size -= trim_val
            overflow -= trim_val
        if overflow > 0:
            test_size = max(0, test_size - overflow)

    if train_size <= 0:
        raise ValueError("Training split is empty; increase train_frac")
    if val_size <= 0:
        raise ValueError("Validation split is empty; increase val_frac")

    indices = np.arange(total)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:train_size+val_size+test_size]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx) if test_size > 0 else None

    if use_smf:
        collate_fn = smf_collate
    else:
        collate_fn = partial(hdf5_collate, max_size=ds.max_size)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn) if test_ds is not None else None
    return ds, train_loader, val_loader, test_loader


def _build_model_from_spec(model_spec, input_dim, target_dim, use_mask):
    """Instantiate a model given a lightweight spec dict."""
    mtype = model_spec["model_type"]
    if model_spec.get("use_train_model", False):
        if mtype == "deepset" and use_mask:
            return DeepSet(
                input_dim=input_dim,
                phi_hidden=model_spec["phi_hidden"],
                rho_hidden=model_spec["rho_hidden"],
                agg=model_spec["agg"],
                out_dim=target_dim,
            )
        if mtype == "slotsetpool" and use_mask:
            return SlotSetPool(
                input_dim=input_dim,
                H=model_spec["slot_H"],
                K=model_spec["slot_K"],
                out_dim=target_dim,
            )
        return MLP(in_dim=input_dim, hidden=model_spec.get("mlp_hidden", [128, 64]), out_dim=target_dim)

    if mtype == "deepset" and use_mask:
        return DeepSet(
            input_dim=input_dim,
            phi_hidden=model_spec["phi_hidden"],
            rho_hidden=model_spec["rho_hidden"],
            agg=model_spec.get("agg", "sum"),
            out_dim=target_dim,
        )
    if mtype == "slotsetpool" and use_mask:
        return SlotSetPool(
            input_dim=input_dim,
            H=model_spec["slot_H"],
            K=model_spec["slot_K"],
            out_dim=target_dim,
            phi_hidden=tuple(model_spec["slot_phi_hidden"]),
            head_hidden=tuple(model_spec["slot_head_hidden"]),
            dropout=model_spec["slot_dropout"],
        )
    # default to MLP
    return MLP(in_dim=input_dim, hidden=model_spec.get("mlp_hidden", [128, 64]), out_dim=target_dim)


def _child_objective(payload, result_queue):
    """Run one trial in a fresh process and report via queue."""
    try:
        args_dict = payload["args"]
        model_spec = payload["model_spec"]
        lr = payload["lr"]
        device = torch.device(payload["device"])
        trial_number = payload["trial_number"]

        # reconstruct args Namespace
        args_ns = argparse.Namespace(**args_dict)

        if getattr(args_ns, "max_batches", None) == 0:
            args_ns.max_batches = None

        base_ds, train_loader, val_loader, test_loader = get_data_loaders(
            args_ns.h5_path,
            args_ns.snap,
            args_ns.param_keys,
            args_ns.data_field,
            args_ns.batch,
            args_ns.train_frac,
            args_ns.val_frac,
            args_ns.test_frac,
            args_ns.use_smf,
        )

        if getattr(args_ns, "mem_debug", False):
            log_memory("trial_start", run=None, log_file=getattr(args_ns, "mem_log_file", None), trial=trial_number)

        # determine dims again to stay consistent with data in this process
        sample_batch = next(iter(train_loader))
        if len(sample_batch) == 3:
            Xb, maskb, yb = sample_batch
            use_mask = True
            input_dim = int(Xb.shape[2])
        else:
            Xb, yb = sample_batch
            use_mask = False
            input_dim = int(Xb.shape[1])
        target_dim = 1 if yb.ndim == 1 else int(yb.shape[1])

        model = _build_model_from_spec(model_spec, input_dim, target_dim, use_mask)

        wandb_run = None
        if getattr(args_ns, "wandb", False):
            wandb_run = safe_wandb_init(
                project=args_ns.wandb_project,
                entity=args_ns.wandb_entity,
                name=f"{args_ns.study_name}_trial{trial_number}",
                group=args_ns.study_name,
                config={"optuna": model_spec, "lr": lr, "trial": trial_number},
                settings=wandb.Settings(start_method="fork"),
            )
            if wandb_run is not None:
                wandb_run.define_metric("train/*", step_metric="global_step")
                wandb_run.define_metric("val/*", step_metric="global_step")
                wandb_run.define_metric("test/*", step_metric="global_step")

        if getattr(args_ns, "mem_debug", False):
            log_memory("model_created", run=wandb_run, step=trial_number, log_file=getattr(args_ns, "mem_log_file", None), trial=trial_number)
            log_memory("before_train", run=wandb_run, step=trial_number, log_file=getattr(args_ns, "mem_log_file", None), trial=trial_number)

        val_mse, val_r2, train_mse, train_r2, test_mse, test_r2 = train_and_eval(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args_ns.epochs,
            lr=lr,
            max_batches=args_ns.max_batches,
            wandb_run=wandb_run,
            trial_number=trial_number,
            test_loader=test_loader,
            input_norm=getattr(args_ns, "_input_norm", "none"),
            input_stats=getattr(args_ns, "_input_stats", None),
            output_norm=getattr(args_ns, "_output_norm", "none"),
            output_stats=getattr(args_ns, "_output_stats", None),
        )

        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass

        if getattr(args_ns, "mem_debug", False):
            log_memory("after_train", run=None, log_file=getattr(args_ns, "mem_log_file", None), trial=trial_number)
            try:
                del model
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            log_memory("after_cleanup", run=None, log_file=getattr(args_ns, "mem_log_file", None), trial=trial_number)

        result_queue.put({
            "status": "ok",
            "val_mse": float(val_mse),
            "val_r2": float(val_r2),
            "train_mse": float(train_mse),
            "train_r2": float(train_r2),
            "test_mse": float(test_mse),
            "test_r2": float(test_r2),
        })
    except Exception as exc:  # pragma: no cover - defensive
        import traceback
        result_queue.put({"status": "error", "error": f"{exc}\n{traceback.format_exc()}"})


def _run_trial_subprocess(payload):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child_objective, args=(payload, q))
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Child process exited with code {p.exitcode}")
    if q.empty():
        raise RuntimeError("Child process produced no result")

    res = q.get()
    if res.get("status") != "ok":
        raise RuntimeError(res.get("error", "Child trial failed"))
    return res


def train_and_eval(model, train_loader, val_loader, device, epochs=3, lr=1e-3, max_batches=None,
                   wandb_run=None, trial_number=None, test_loader=None,
                   input_norm: str = "none", input_stats: dict = None,
                   output_norm: str = "none", output_stats: dict = None):
    """Train model and return validation/train (and optional test) metrics.

    Returns (val_mse, val_r2, train_mse, train_r2, test_mse, test_r2)
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    global_step = 0

    # derive labels from dataset if available (train_loader.dataset may be a Subset)
    labels = None
    try:
        ds_obj = getattr(train_loader, 'dataset', None)
        if hasattr(ds_obj, 'dataset'):
            ds_real = ds_obj.dataset
        else:
            ds_real = ds_obj
        # prefer explicit param_names exposed by HDF5SetDataset, fall back to param_keys
        labels = getattr(ds_real, 'param_names', None) or getattr(ds_real, 'param_keys', None)
    except Exception:
        labels = None

    # Use training utilities from `train.py` to keep metrics consistent
    for ep in range(epochs):
        # train for one epoch using train.train_epoch
        avg_loss, train_metrics = train_mod.train_epoch(
            model, train_loader, opt, loss_fn, device,
            input_norm=input_norm, input_stats=input_stats,
            output_norm=output_norm, output_stats=output_stats,
        )

        # quick validation metrics using train.eval_model
        _, val_metrics = train_mod.eval_model(
            model, val_loader, loss_fn, device,
            input_norm=input_norm, input_stats=input_stats,
            output_norm=output_norm, output_stats=output_stats,
        )

        # optional per-epoch test evaluation
        test_metrics = None
        if test_loader is not None:
            _, test_metrics = train_mod.eval_model(
                model, test_loader, loss_fn, device,
                input_norm=input_norm, input_stats=input_stats,
                output_norm=output_norm, output_stats=output_stats,
            )

        # per-epoch logging
        if wandb_run is not None:
            _wandb_log_metrics(wandb_run, global_step, "val", val_metrics.get("mse", float("nan")), val_metrics.get("r2", float("nan")), ep=ep, trial=trial_number)
            _wandb_log_metrics(wandb_run, global_step, "train", train_metrics.get("mse", float("nan")), train_metrics.get("r2", float("nan")), ep=ep, trial=trial_number)
            if test_metrics is not None:
                _wandb_log_metrics(wandb_run, global_step, "test", test_metrics.get("mse", float("nan")), test_metrics.get("r2", float("nan")), ep=ep, trial=trial_number)
            wandb_run.log({"train/loss": float(avg_loss), "epoch": ep, "trial": trial_number}, step=global_step)
            # log per-target metrics as individual metrics
            try:
                    mse_per = train_metrics.get("mse_per_target", None)
                    r2_per = train_metrics.get("r2_per_target", None)
                    if mse_per is not None and r2_per is not None and len(mse_per) == len(r2_per):
                        d = {}
                        for i in range(len(mse_per)):
                            label = (labels[i] if labels is not None and i < len(labels) else f"target_{i}")
                            safe_label = str(label).replace('/', '_').replace(' ', '_')
                            d[f"train/mse/{safe_label}"] = float(mse_per[i])
                            d[f"train/r2/{safe_label}"] = float(r2_per[i])
                        d.update({"epoch": ep, "trial": trial_number})
                        wandb_run.log(d, step=global_step)

                    mse_per_v = val_metrics.get("mse_per_target", None)
                    r2_per_v = val_metrics.get("r2_per_target", None)
                    if mse_per_v is not None and r2_per_v is not None and len(mse_per_v) == len(r2_per_v):
                        d = {}
                        for i in range(len(mse_per_v)):
                            label = (labels[i] if labels is not None and i < len(labels) else f"target_{i}")
                            safe_label = str(label).replace('/', '_').replace(' ', '_')
                            d[f"val/mse/{safe_label}"] = float(mse_per_v[i])
                            d[f"val/r2/{safe_label}"] = float(r2_per_v[i])
                        d.update({"epoch": ep, "trial": trial_number})
                        wandb_run.log(d, step=global_step)
                    # per-target test metrics
                    try:
                        if test_metrics is not None:
                            mse_per_t = test_metrics.get("mse_per_target", None)
                            r2_per_t = test_metrics.get("r2_per_target", None)
                            if mse_per_t is not None and r2_per_t is not None and len(mse_per_t) == len(r2_per_t):
                                d = {}
                                for i in range(len(mse_per_t)):
                                    label = (labels[i] if labels is not None and i < len(labels) else f"target_{i}")
                                    safe_label = str(label).replace('/', '_').replace(' ', '_')
                                    d[f"test/mse/{safe_label}"] = float(mse_per_t[i])
                                    d[f"test/r2/{safe_label}"] = float(r2_per_t[i])
                                d.update({"epoch": ep, "trial": trial_number})
                                wandb_run.log(d, step=global_step)
                    except Exception:
                        pass
            except Exception:
                pass
        global_step += 1

    # ---- Full eval (same as your original) ----
    model.eval()
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            if len(batch) == 3:
                X, mask, y = batch
                X = X.to(device)
                mask = mask.to(device)
            else:
                X, y = batch
                X = X.to(device)
                mask = None

            Xn, maskn, yn = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
            pred_n = run_model(model, Xn, maskn)
            pred_orig = _inverse_output_norm(pred_n, output_norm, output_stats, device)

            y_np = y.cpu().numpy()
            pred_np = pred_orig.cpu().numpy()
            y_np = y_np.reshape(y_np.shape[0], -1) if y_np.ndim == 1 else y_np
            pred_np = pred_np.reshape(pred_np.shape[0], -1) if pred_np.ndim == 1 else pred_np
            y_trues.append(y_np)
            y_preds.append(pred_np)

    if len(y_trues) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')

    y_trues = np.concatenate(y_trues, axis=0)
    y_preds = np.concatenate(y_preds, axis=0)

    mse = float(np.mean((y_trues - y_preds) ** 2))

    K = y_trues.shape[1]
    r2_list = []
    for k in range(K):
        yt = y_trues[:, k]
        yp = y_preds[:, k]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_list.append(r2)
    r2 = float(np.mean(r2_list))

    # training metrics on limited subset
    train_trues = []
    train_preds = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break
            if len(batch) == 3:
                X, mask, y = batch
                X = X.to(device)
                mask = mask.to(device)
            else:
                X, y = batch
                X = X.to(device)
                mask = None

            Xn, maskn, yn = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
            pred_n = run_model(model, Xn, maskn)
            pred_orig = _inverse_output_norm(pred_n, output_norm, output_stats, device)

            y_np = y.cpu().numpy()
            pred_np = pred_orig.cpu().numpy()
            y_np = y_np.reshape(y_np.shape[0], -1) if y_np.ndim == 1 else y_np
            pred_np = pred_np.reshape(pred_np.shape[0], -1) if pred_np.ndim == 1 else pred_np
            train_trues.append(y_np)
            train_preds.append(pred_np)

    if len(train_trues) == 0:
        train_mse = float('nan')
        train_r2 = float('nan')
    else:
        train_trues = np.concatenate(train_trues, axis=0)
        train_preds = np.concatenate(train_preds, axis=0)
        train_mse = float(np.mean((train_trues - train_preds) ** 2))
        Kt = train_trues.shape[1]
        r2_list_t = []
        for k in range(Kt):
            yt = train_trues[:, k]
            yp = train_preds[:, k]
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2t = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r2_list_t.append(r2t)
        train_r2 = float(np.mean(r2_list_t))

    # Optional test evaluation
    test_mse = float('nan')
    test_r2 = float('nan')
    if test_loader is not None:
        test_trues, test_preds = [], []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if max_batches is not None and i >= max_batches:
                    break
                if len(batch) == 3:
                    X, mask, y = batch
                    X = X.to(device)
                    mask = mask.to(device)
                else:
                    X, y = batch
                    X = X.to(device)
                    mask = None

                Xn, maskn, yn = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
                pred_n = run_model(model, Xn, maskn)
                pred_orig = _inverse_output_norm(pred_n, output_norm, output_stats, device)

                y_np = y.cpu().numpy()
                pred_np = pred_orig.cpu().numpy()
                y_np = y_np.reshape(y_np.shape[0], -1) if y_np.ndim == 1 else y_np
                pred_np = pred_np.reshape(pred_np.shape[0], -1) if pred_np.ndim == 1 else pred_np
                test_trues.append(y_np)
                test_preds.append(pred_np)

        if len(test_trues) > 0:
            test_trues = np.concatenate(test_trues, axis=0)
            test_preds = np.concatenate(test_preds, axis=0)
            test_mse = float(np.mean((test_trues - test_preds) ** 2))
            Kt = test_trues.shape[1]
            r2_list_test = []
            for k in range(Kt):
                yt = test_trues[:, k]
                yp = test_preds[:, k]
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - yt.mean()) ** 2)
                r2t = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                r2_list_test.append(r2t)
            test_r2 = float(np.mean(r2_list_test))

    # log final summary metrics (single point is fine here)
    if wandb_run is not None:
        _wandb_log_metrics(wandb_run, global_step, "val", mse, r2, ep=epochs, trial=trial_number)
        _wandb_log_metrics(wandb_run, global_step, "train", train_mse, train_r2, ep=epochs, trial=trial_number)
        if test_loader is not None:
            _wandb_log_metrics(wandb_run, global_step, "test", test_mse, test_r2, ep=epochs, trial=trial_number)

        wandb_run.summary["val_mse"] = float(mse)
        wandb_run.summary["val_r2"] = float(r2)
        wandb_run.summary["train_mse"] = float(train_mse)
        wandb_run.summary["train_r2"] = float(train_r2)
        if test_loader is not None:
            wandb_run.summary["test_mse"] = float(test_mse)
            wandb_run.summary["test_r2"] = float(test_r2)


    return mse, r2, train_mse, train_r2, test_mse, test_r2



def objective(trial, args, base_ds, train_loader, val_loader, device):
    """Spawn a fresh process per trial to reclaim memory between runs."""
    model_type = getattr(args, "model_type", "deepset")

    # infer dims from the pre-created loader to set parameter ranges
    sample_batch = next(iter(train_loader))
    if len(sample_batch) == 3:
        Xb, maskb, yb = sample_batch
        use_mask = True
        input_dim = int(Xb.shape[2])
    else:
        Xb, yb = sample_batch
        use_mask = False
        input_dim = int(Xb.shape[1])
    target_dim = 1 if yb.ndim == 1 else int(yb.shape[1])

    # Sample hyperparameters in the parent (Optuna needs access to trial).
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    model_spec = {"model_type": model_type, "use_train_model": getattr(args, "use_train_model", False)}

    if getattr(args, "use_train_model", False):
        if model_type == "deepset" and use_mask:
            model_spec.update({"phi_hidden": [64, 128, 512], "rho_hidden": [128, 128, 128], "agg": "mean"})
        elif model_type == "slotsetpool" and use_mask:
            model_spec.update({"slot_H": 128, "slot_K": 8})
        else:
            model_spec.update({"mlp_hidden": [128, 64]})
    else:
        if model_type == "deepset" and use_mask:
            n_phi_layers = trial.suggest_int("phi_layers", 1, 5)
            phi_hidden = [trial.suggest_int(f"phi_h_{i}", 32, 256, log=True) for i in range(n_phi_layers)]
            n_rho_layers = trial.suggest_int("rho_layers", 1, 5)
            rho_hidden = [trial.suggest_int(f"rho_h_{i}", 32, 512, log=True) for i in range(n_rho_layers)]
            agg = trial.suggest_categorical("agg", ["sum", "mean"])
            model_spec.update({"phi_hidden": phi_hidden, "rho_hidden": rho_hidden, "agg": agg})
        elif model_type == "slotsetpool" and use_mask:
            K = trial.suggest_int("slot_K", 2, 16)
            H = trial.suggest_int("slot_H", 64, 512, log=True)
            phi_h0 = trial.suggest_int("slot_phi_h0", 64, 1024, log=True)
            phi_h1 = trial.suggest_int("slot_phi_h1", 64, 1024, log=True)
            head_h0 = trial.suggest_int("slot_head_h0", 64, 1024, log=True)
            head_h1 = trial.suggest_int("slot_head_h1", 64, 1024, log=True)
            dropout = trial.suggest_float("slot_dropout", 0.0, 0.5)
            model_spec.update({
                "slot_K": K,
                "slot_H": H,
                "slot_phi_hidden": [phi_h0, phi_h1],
                "slot_head_hidden": [head_h0, head_h1],
                "slot_dropout": dropout,
            })
        else:
            n_layers = trial.suggest_int("mlp_layers", 1, 5)
            hidden = [trial.suggest_int(f"mlp_h_{i}", 32, 2048, log=True) for i in range(n_layers)]
            model_spec.update({"mlp_hidden": hidden})

    # If deepset requested but no mask, fall back to mlp for the actual run.
    if model_spec["model_type"] == "deepset" and not use_mask:
        warnings.warn("Requested 'deepset' model but data loader yields fixed-vector batches; falling back to MLP.")
        model_spec = {"model_type": "mlp", "mlp_hidden": model_spec.get("mlp_hidden", [128, 64]), "use_train_model": model_spec.get("use_train_model", False)}

    payload = {
        "args": {k: v for k, v in vars(args).items()},
        "model_spec": model_spec,
        "lr": lr,
        "device": str(device),
        "trial_number": trial.number,
    }

    res = _run_trial_subprocess(payload)

    trial.set_user_attr("val_mse", float(res["val_mse"]))
    trial.set_user_attr("val_r2", float(res["val_r2"]))
    trial.set_user_attr("train_mse", float(res["train_mse"]))
    trial.set_user_attr("train_r2", float(res["train_r2"]))
    if "test_mse" in res:
        trial.set_user_attr("test_mse", float(res["test_mse"]))
    if "test_r2" in res:
        trial.set_user_attr("test_r2", float(res["test_r2"]))

    return float(res["val_mse"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, default="../data/camels_LH.hdf5")
    parser.add_argument("--snap", type=int, default=90)
    parser.add_argument("--param-keys", type=str, nargs="*", default=None)
    parser.add_argument("--use-smf", action="store_true", default=False)
    parser.add_argument("--data-field", type=str, default="SubhaloStellarMass")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction of data for training")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of data for validation")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of data for testing")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--model-type", type=str, choices=["deepset", "mlp", "slotsetpool"], default="deepset",
                        help="Choose model type to evaluate (not optimized)")
    parser.add_argument("--wandb", action="store_true", help="Log Optuna trials to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="deepset_optuna")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1, help="Run trials in parallel (n_jobs passed to optuna.study.optimize)")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///optuna.db) for distributed parallelization")
    parser.add_argument("--max-batches", type=int, default=50, help="max batches per epoch to speed trials; 0 or None for no limit")
    parser.add_argument("--study-name", type=str, default="deepset_optuna")
    parser.add_argument("--out-dir", type=str, default="../data/optuna")
    parser.add_argument("--normalize-input", choices=["none", "log", "log_std"], default="none")
    parser.add_argument("--normalize-output", choices=["none", "minmax"], default="none")
    parser.add_argument("--stats-sample", type=int, default=1000)
    parser.add_argument("--use-train-model", action="store_true", default=False,
                        help="Use the same model architecture defaults as `train.py` (for sanity checks)")
    parser.add_argument("--mem-debug", action="store_true", default=False,
                        help="Log CPU/GPU memory usage at key points in each trial")
    parser.add_argument("--mem-log-file", type=str, default=None,
                        help="Path to append memory logs (used when --mem-debug is set)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional: enable tracemalloc for Python allocation debugging
    if getattr(args, "mem_debug", False):
        try:
            import tracemalloc
            tracemalloc.start()
        except Exception:
            pass

    args.data_field = "smf/phi" if args.use_smf else "SubhaloStellarMass"

    base_ds, train_loader, val_loader, test_loader = get_data_loaders(
        args.h5_path,
        args.snap,
        args.param_keys,
        args.data_field,
        args.batch,
        args.train_frac,
        args.val_frac,
        args.test_frac,
        args.use_smf,
    )
    print(f"Split sizes -> train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset) if test_loader is not None else 0}")

    # Auto-detect parameter names and dataset size for convenience
    try:
        detected_nsims = len(base_ds)
        detected_param_names = getattr(base_ds, 'param_names', None) or getattr(base_ds, 'param_keys', None)

        if detected_param_names is not None:
            # If user passed integer indices in --param-keys, map them to names
            if args.param_keys is not None:
                mapped = []
                for p_or_idx in args.param_keys:
                    try:
                        idx = int(p_or_idx)
                        if 0 <= idx < len(detected_param_names):
                            mapped.append(detected_param_names[idx])
                        else:
                            raise ValueError(f"Index {idx} is out of bounds for detected parameters.")
                    except (ValueError, TypeError):
                        # Not an integer, assume it's a name
                        mapped.append(p_or_idx)
                args.param_keys = mapped
            else:
                # Default to all available parameters if none were provided
                args.param_keys = detected_param_names

            print(f"Detected dataset: nsims={detected_nsims}, ndim={len(detected_param_names)}")
        else:
            print(f"Detected dataset: nsims={detected_nsims}")

        total_frac = args.train_frac + args.val_frac + args.test_frac
        if total_frac <= 0:
            raise ValueError("train/val/test fractions must sum to a positive value")
        if total_frac > 1.0 + 1e-6:
            scale = 1.0 / total_frac
            args.train_frac *= scale
            args.val_frac *= scale
            args.test_frac *= scale
            print(f"Scaled fractions to sum to 1.0: train_frac={args.train_frac:.3f}, val_frac={args.val_frac:.3f}, test_frac={args.test_frac:.3f}")

        # Recreate loaders if we changed param_keys or adjusted fractions
        base_ds, train_loader, val_loader, test_loader = get_data_loaders(
            args.h5_path,
            args.snap,
            args.param_keys,
            args.data_field,
            args.batch,
            args.train_frac,
            args.val_frac,
            args.test_frac,
            args.use_smf,
        )
        print(f"Updated split sizes -> train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset) if test_loader is not None else 0}")
    except Exception:
        # non-fatal: continue with original loaders
        pass

    if args.max_batches == 0:
        args.max_batches = None

    # Compute normalization stats once (from the training split) and attach to args
    input_stats = None
    output_stats = None
    if args.normalize_input != "none" or args.normalize_output != "none":
        stats = compute_stats_from_dataset(train_loader.dataset, sample_limit=args.stats_sample)
        input_stats = {"input_mean": stats["input_mean"], "input_std": stats["input_std"]}
        output_stats = {"y_min": stats["y_min"], "y_max": stats["y_max"]}
    args._input_stats = input_stats
    args._output_stats = output_stats
    args._input_norm = args.normalize_input
    args._output_norm = args.normalize_output

    # Pre-create the W&B project in the main process to avoid
    # race conditions where many workers try to create the same project
    # simultaneously and trigger HTTP 409 errors.
    if getattr(args, "wandb", False):
        try:
            pre_name = f"{args.study_name}_setup"
            pre_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=pre_name,
                reinit=True,
                settings=wandb.Settings(start_method="fork"),
            )
            pre_run.finish()
        except Exception as e:
            # If the project already exists or a 409-like error occurs,
            # ignore it and continue — the goal is to ensure the project
            # exists before workers start.
            msg = str(e).lower()
            if ("project may already be created" in msg) or ("409" in msg) or ("already exists" in msg):
                pass
            else:
                raise

    # create or load a study; if storage is provided, use it (allows multi-process distributed parallelization)
    storage = args.storage if getattr(args, 'storage', None) else None
    study = optuna.create_study(direction="minimize", study_name=args.study_name,
                                sampler=optuna.samplers.TPESampler(), storage=storage, load_if_exists=True)
    func = lambda trial: objective(trial, args, base_ds, train_loader, val_loader, device)

    study.optimize(func, n_trials=args.trials, n_jobs=args.n_jobs)

    best = study.best_trial
    print(f"Best value: {best.value}")
    print("Best params:")
    print(best.params)

    out_path = os.path.join(args.out_dir, f"{args.study_name}_best.json")
    with open(out_path, "w") as f:
        json.dump({"value": best.value, "params": best.params, "user_attrs": best.user_attrs}, f, indent=2)
    print(f"Saved best params to {out_path}")


if __name__ == "__main__":
    main()
