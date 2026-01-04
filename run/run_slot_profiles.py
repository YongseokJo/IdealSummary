#!/usr/bin/env python3
"""Analyze how SlotSetPool slots attend to galaxies.

This script loads a SlotSetPool checkpoint (saved by src/train.py), runs the
model over a few batches, and summarizes for each slot:
- attention-weighted feature mean/std (what kinds of galaxies it focuses on)
- attention entropy (sharp vs diffuse assignment)
- distributions of top-k attended galaxies (per-slot histograms)

Outputs:
- JSON summary
- histogram plot(s)

Example:
  python run/run_slot_profiles.py --ckpt data/models/checkpoints/deepset_epoch100.pt \
    --h5-path data/camels_SB28.hdf5 --snap 90 --batch 8 --max-batches 20 \
    --out-dir analysis/slot_profiles
"""

import argparse
import json
import os
import sys

import torch

# ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.data import HDF5SetDataset, hdf5_collate
from src.setpooling import SlotSetPool
from tools.setpool_analysis import accumulate_slot_attention_over_loader, plot_slot_feature_histograms
import functools

try:
    import optuna
except Exception:
    optuna = None


def _build_slotsetpool_from_ckpt_args(args_dict: dict, input_dim: int, target_dim: int) -> SlotSetPool:
    # train.py uses argparse dest names like slot_K, slot_H, slot_dropout, slot_logm_idx
    K = int(args_dict.get("slot_K", args_dict.get("slot-K", 8)))
    H = int(args_dict.get("slot_H", args_dict.get("slot-H", 128)))
    dropout = float(args_dict.get("slot_dropout", args_dict.get("slot-dropout", 0.0)))
    logm_idx = int(args_dict.get("slot_logm_idx", args_dict.get("slot-logm-idx", 0)))

    # If checkpoint doesn't store phi/head hidden sizes, fall back to SlotSetPool defaults.
    return SlotSetPool(
        input_dim=input_dim,
        logm_idx=logm_idx,
        H=H,
        K=K,
        out_dim=target_dim,
        dropout=dropout,
    )


def _build_slotsetpool_from_optuna_params(params: dict, input_dim: int, target_dim: int) -> SlotSetPool:
    # Optuna params naming convention used in src/optuna_search.py
    if not ("slot_K" in params or "slot_H" in params or any(str(k).startswith("slot_") for k in params.keys())):
        raise ValueError("Optuna trial params do not look like a SlotSetPool trial (missing slot_* params)")

    K = int(params.get("slot_K", 8))
    H = int(params.get("slot_H", 128))
    dropout = float(params.get("slot_dropout", 0.0))
    logm_idx = int(params.get("slot_logm_idx", 0))

    return SlotSetPool(
        input_dim=input_dim,
        logm_idx=logm_idx,
        H=H,
        K=K,
        out_dim=target_dim,
        dropout=dropout,
    )


def _load_optuna_trial(optuna_db: str, study_name: str, trial_number: int | None):
    if optuna is None:
        raise RuntimeError("Optuna is not installed in this environment. Install with `pip install optuna`. ")
    storage_url = f"sqlite:///{os.path.abspath(optuna_db)}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    if trial_number is None:
        return study.best_trial
    for t in study.trials:
        if t.number == trial_number:
            return t
    raise KeyError(f"Trial number {trial_number} not found in study '{study_name}'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path containing trained weights (torch.save dict with model_state_dict). If provided, Optuna params are ignored.")
    p.add_argument("--optuna-db", type=str, default=None, help="Optuna sqlite DB path to load SlotSetPool trial params (required if --ckpt is not provided)")
    p.add_argument("--study-name", type=str, default=None, help="Optuna study name (required if --ckpt is not provided)")
    p.add_argument("--trial-number", type=int, default=None, help="Optional: use a specific trial number instead of best trial")
    p.add_argument("--h5-path", required=True)
    p.add_argument("--snap", type=int, default=90)
    p.add_argument("--data-field", type=str, default="SubhaloStellarMass")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=20)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out-dir", type=str, default="analysis_outputs/slot_profiles")
    p.add_argument("--param-keys", nargs="+", default=None, help="Optional subset of target params (names)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = None
    ckpt_args = {}
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    trial = None
    trial_params = None
    if args.ckpt is None:
        # must rely on Optuna params if no checkpoint provided
        if args.optuna_db is None or args.study_name is None:
            raise ValueError("Without --ckpt you must supply --optuna-db and --study-name to reconstruct the SlotSetPool architecture")
        trial = _load_optuna_trial(args.optuna_db, args.study_name, args.trial_number)
        trial_params = dict(trial.params)
    else:
        # ignore optuna if checkpoint is provided (per request)
        trial = None
        trial_params = None

    # Dataset / loader
    ds = HDF5SetDataset(h5_path=args.h5_path, snap=args.snap, param_keys=args.param_keys, data_field=args.data_field)
    collate_fn = functools.partial(hdf5_collate, max_size=ds.max_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    # infer dims
    Xb, maskb, yb = next(iter(loader))
    input_dim = int(Xb.shape[2])
    target_dim = 1 if yb.ndim == 1 else int(yb.shape[1])

    if trial_params is not None:
        model = _build_slotsetpool_from_optuna_params(trial_params, input_dim=input_dim, target_dim=target_dim)
        if ckpt is not None:
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=True)
    else:
        model = _build_slotsetpool_from_ckpt_args(ckpt_args, input_dim=input_dim, target_dim=target_dim)
        if ckpt is not None:
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    summary = accumulate_slot_attention_over_loader(
        model,
        loader,
        device=device,
        max_batches=args.max_batches,
        topk=args.topk,
    )
    if summary is None:
        raise RuntimeError("No batches processed; check dataset and max-batches")

    # Save JSON (convert numpy to python)
    out_json = {
        "ckpt": args.ckpt,
        "optuna_db": args.optuna_db,
        "study_name": args.study_name,
        "trial_number": (int(trial.number) if trial is not None else None),
        "trial_value": (float(trial.value) if trial is not None and trial.value is not None else None),
        "trial_params": trial_params,
        "h5_path": args.h5_path,
        "snap": int(args.snap),
        "data_field": args.data_field,
        "batch": int(args.batch),
        "max_batches": int(args.max_batches),
        "topk": int(args.topk),
        "feat_mean_w": summary["feat_mean_w"].tolist(),
        "feat_std_w": summary["feat_std_w"].tolist(),
        "entropy": summary["entropy"].tolist(),
    }

    with open(os.path.join(args.out_dir, "slot_profiles.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    # Plot histograms for feature 0 by default
    try:
        plot_slot_feature_histograms(
            summary["topk_values"],
            out_path=os.path.join(args.out_dir, "slot_topk_hist_feature0.png"),
            feature_idx=0,
            bins=50,
            title="SlotSetPool: top-k attended galaxies (feature 0)",
        )
    except Exception:
        pass

    print(f"Saved slot profile outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
