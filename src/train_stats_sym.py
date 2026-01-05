"""
Training script for Simplified Set-DSR (stats_sym.py).

This script trains the SimplifiedSetDSR model which:
1. Learns per-element symbolic transforms g(x) via GP search
2. Uses learnable weights w(x) for weighted statistics (trained by gradient descent)
3. Computes classical summary statistics (moments, cumulants, quantiles)
4. Selects top-K features
5. Uses MLP or linear head for final prediction

Usage:
    python train_stats_sym.py --h5-path ../data/camels_LH.hdf5 --n-generations 50 --wandb
"""

import os
import sys
import argparse
import math
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from data import HDF5SetDataset, hdf5_collate
from stats_sym import (
    SimplifiedSetDSR,
    TransformEvolver,
    SummaryStatistics,
    create_simplified_model,
)
from train import (
    compute_stats_from_dataset,
    normalize_batch,
    _inverse_output_norm,
    plot_pred_vs_true,
)


# =============================================================================
# Training Functions
# =============================================================================

def compute_summary_feature_stats(
    model: SimplifiedSetDSR,
    loader: DataLoader,
    device: torch.device,
    input_norm: str = "none",
    input_stats: dict = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute dataset-wide mean/std for summary statistics features."""
    model.eval()
    total = 0
    feat_sum = None
    feat_sumsq = None

    with torch.no_grad():
        for batch in loader:
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            X, mask, _ = normalize_batch(
                X, mask, y,
                input_norm, input_stats,
                output_norm="none", output_stats=None,
            )
            feats, _ = model.compute_summary_features(X, mask)
            if feat_sum is None:
                feat_sum = feats.sum(dim=0)
                feat_sumsq = (feats ** 2).sum(dim=0)
            else:
                feat_sum += feats.sum(dim=0)
                feat_sumsq += (feats ** 2).sum(dim=0)
            total += feats.shape[0]

    if total == 0:
        raise ValueError("Empty loader when computing summary feature statistics")

    mean = feat_sum / float(total)
    var = feat_sumsq / float(total) - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0)) + eps
    return mean, std


def apply_summary_feature_norm(
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (features - mean) / std


def train_weights_and_head(
    model: SimplifiedSetDSR,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
    lr_head: Optional[float] = None,
    lr_weight: Optional[float] = None,
    weight_decay: float = 1e-4,
    scheduler_name: str = "warmup_cosine",
    warmup_frac: float = 0.1,
    min_lr_scale: float = 0.05,
    profile: bool = False,
    profile_epochs: int = 2,
    profile_steps: bool = False,
    profile_steps_epochs: int = 2,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    input_norm: str = "none",
    input_stats: dict = None,
    output_norm: str = "none",
    output_stats: dict = None,
    wandb_run=None,
    epoch_offset: int = 0,
) -> Dict[str, List[float]]:
    """
    Train learnable weights and prediction head by gradient descent.
    
    The per-element transforms are frozen during this phase.
    """
    model = model.to(device)
    # Prepare a sample batch for logging feature distributions / weights to wandb
    sample_X = None
    sample_mask = None
    try:
        sample_batch = next(iter(train_loader))
        sample_X, sample_mask, _ = sample_batch
        sample_X = sample_X.to(device)
        sample_mask = sample_mask.to(device)
    except Exception:
        sample_X = None
        sample_mask = None
    
    # Only train weights and head parameters
    params_head = list(model.head.parameters())
    params_weight = []
    if model.weight_net is not None:
        params_weight = list(model.weight_net.parameters())
    if model.selector is not None and model.selector.method == "learnable":
        params_head.append(model.selector.gate_logits)

    def make_scheduler(optimizer, name, total_epochs, steps_per_epoch):
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs)
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=optimizer.defaults["lr"] * min_lr_scale,
            )
        if name == "warmup_cosine":
            total_steps = max(total_epochs * steps_per_epoch, 1)
            warmup_steps = max(int(total_steps * warmup_frac), 1)

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_scale + (1.0 - min_lr_scale) * cosine

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        raise ValueError(f"Unknown scheduler: {name}")

    steps_per_epoch = max(len(train_loader), 1)
    lr_head = lr if lr_head is None else lr_head
    lr_weight = lr if lr_weight is None else lr_weight

    optimizer_head = torch.optim.AdamW(params_head, lr=lr_head, weight_decay=weight_decay)
    scheduler_head = make_scheduler(optimizer_head, scheduler_name, n_epochs, steps_per_epoch)
    optimizer_weight = None
    scheduler_weight = None
    if len(params_weight) > 0:
        optimizer_weight = torch.optim.AdamW(params_weight, lr=lr_weight, weight_decay=weight_decay)
        scheduler_weight = make_scheduler(optimizer_weight, scheduler_name, n_epochs, steps_per_epoch)

    per_batch_scheduler = scheduler_name == "warmup_cosine"
    use_amp = use_amp and device.type == "cuda"
    amp_dtype_t = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype_t == torch.float16)
    
    history = {"train_loss": [], "val_loss": [], "val_r2": []}
    eps = 1e-8
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        do_profile = profile and epoch < profile_epochs
        if do_profile:
            t_epoch = time.perf_counter()
            t_load = 0.0
            t_norm = 0.0
            t_zero = 0.0
            t_fwd = 0.0
            t_loss = 0.0
            t_bwd = 0.0
            t_clip = 0.0
            t_opt = 0.0
            t_sched = 0.0
            t_other = 0.0
        if profile_steps and epoch < profile_steps_epochs:
            if hasattr(model, "reset_profile_stats"):
                model.reset_profile_stats()
        
        for batch in train_loader:
            t0 = time.perf_counter() if do_profile else None
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            if t0 is not None:
                t_load += time.perf_counter() - t0
            
            # Normalize
            t1 = time.perf_counter() if t0 is not None else None
            X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
            if t1 is not None:
                t_norm += time.perf_counter() - t1

            if do_profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_zero_start = time.perf_counter() if do_profile else None
            optimizer_head.zero_grad()
            if optimizer_weight is not None:
                optimizer_weight.zero_grad()
            if t_zero_start is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_zero += time.perf_counter() - t_zero_start

            if do_profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd_start = time.perf_counter() if do_profile else None
            with autocast("cuda", enabled=use_amp, dtype=amp_dtype_t):
                pred = model(X, mask)
            if t_fwd_start is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_fwd += time.perf_counter() - t_fwd_start

            if do_profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_loss_start = time.perf_counter() if do_profile else None
            loss = F.mse_loss(pred, y)
            if t_loss_start is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_loss += time.perf_counter() - t_loss_start

            if do_profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_bwd_start = time.perf_counter() if do_profile else None
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if t_bwd_start is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_bwd += time.perf_counter() - t_bwd_start

            if do_profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_clip_start = time.perf_counter() if do_profile else None
            if scaler.is_enabled():
                scaler.unscale_(optimizer_head)
                if optimizer_weight is not None:
                    scaler.unscale_(optimizer_weight)
            torch.nn.utils.clip_grad_norm_(params_head + params_weight, 1.0)
            if t_clip_start is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_clip += time.perf_counter() - t_clip_start

            if do_profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_opt_start = time.perf_counter() if do_profile else None
            if scaler.is_enabled():
                scaler.step(optimizer_head)
                if optimizer_weight is not None:
                    scaler.step(optimizer_weight)
                scaler.update()
            else:
                optimizer_head.step()
                if optimizer_weight is not None:
                    optimizer_weight.step()
            if t_opt_start is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_opt += time.perf_counter() - t_opt_start

            if per_batch_scheduler:
                if do_profile and device.type == "cuda":
                    torch.cuda.synchronize()
                t_sched_start = time.perf_counter() if do_profile else None
                scheduler_head.step()
                if scheduler_weight is not None:
                    scheduler_weight.step()
                if t_sched_start is not None:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_sched += time.perf_counter() - t_sched_start
            
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)
        if do_profile:
            accounted = t_load + t_norm + t_zero + t_fwd + t_loss + t_bwd + t_clip + t_opt + t_sched
            t_other = max((time.perf_counter() - t_epoch) - accounted, 0.0)
            epoch_time = time.perf_counter() - t_epoch
            print(
                "Profile epoch "
                f"{epoch_offset + epoch}: "
                f"total={epoch_time:.3f}s "
                f"load={t_load:.3f}s "
                f"norm={t_norm:.3f}s "
                f"zero={t_zero:.3f}s "
                f"fwd={t_fwd:.3f}s "
                f"loss={t_loss:.3f}s "
                f"bwd={t_bwd:.3f}s "
                f"clip={t_clip:.3f}s "
                f"opt={t_opt:.3f}s "
                f"sched={t_sched:.3f}s "
                f"other={t_other:.3f}s"
            )
        if profile_steps and epoch < profile_steps_epochs and hasattr(model, "profile_stats"):
            stats = model.profile_stats
            batches = max(int(stats.get("batches", 0)), 1)
            summary_total = stats.get("summary_total", 0.0)
            print(
                "Step profile epoch "
                f"{epoch_offset + epoch}: "
                f"summary_total={summary_total:.3f}s "
                f"weights={stats.get('weights', 0.0):.3f}s "
                f"transform={stats.get('transform', 0.0):.3f}s "
                f"stats={stats.get('stats', 0.0):.3f}s "
                f"selector={stats.get('selector', 0.0):.3f}s "
                f"head={stats.get('head', 0.0):.3f}s "
                f"per_batch={(summary_total / batches):.4f}s"
            )
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    X, mask, y = batch
                    X, mask, y = X.to(device), mask.to(device), y.to(device)
                    X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)

                    with autocast("cuda", enabled=use_amp, dtype=amp_dtype_t):
                        pred = model(X, mask)
                        batch_loss = F.mse_loss(pred, y)
                    val_loss += batch_loss.item()
                    all_preds.append(pred)
                    all_targets.append(y)
            
            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            if scheduler_name == "plateau":
                if do_profile and device.type == "cuda":
                    torch.cuda.synchronize()
                t_sched_start = time.perf_counter() if do_profile else None
                scheduler_head.step(val_loss)
                if scheduler_weight is not None:
                    scheduler_weight.step(val_loss)
                if t_sched_start is not None:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_sched += time.perf_counter() - t_sched_start
        if not per_batch_scheduler and scheduler_name != "plateau":
            if do_profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_sched_start = time.perf_counter() if do_profile else None
            scheduler_head.step()
            if scheduler_weight is not None:
                scheduler_weight.step()
            if t_sched_start is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_sched += time.perf_counter() - t_sched_start
        if not per_batch_scheduler and scheduler_name == "plateau" and val_loader is None:
            if do_profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_sched_start = time.perf_counter() if do_profile else None
            scheduler_head.step(avg_train_loss)
            if scheduler_weight is not None:
                scheduler_weight.step(avg_train_loss)
            if t_sched_start is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_sched += time.perf_counter() - t_sched_start

        # Compute R² (overall and per-target)
        if val_loader is not None:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            ss_res = ((all_targets - all_preds) ** 2).sum()
            ss_tot = ((all_targets - all_targets.mean(dim=0)) ** 2).sum() + eps
            r2 = (1 - ss_res / ss_tot).item()
            history["val_r2"].append(r2)

            # Per-target R²
            n_targets = all_targets.shape[1] if all_targets.dim() > 1 else 1
            r2_per_target = []
            for k in range(n_targets):
                if all_targets.dim() > 1:
                    y_k = all_targets[:, k]
                    p_k = all_preds[:, k]
                else:
                    y_k = all_targets
                    p_k = all_preds
                ss_res_k = ((y_k - p_k) ** 2).sum()
                ss_tot_k = ((y_k - y_k.mean()) ** 2).sum() + eps
                r2_per_target.append((1 - ss_res_k / ss_tot_k).item())

            # store per-target r2
            history.setdefault("val_r2_per_target", []).append(r2_per_target)

            # Log to wandb (overall + per-target) using parameter labels if available
            if wandb_run is not None:
                # try to obtain parameter names from wandb config
                param_names = None
                try:
                    # config behaves like a dict-like object
                    param_names = wandb_run.config.get("param_keys", None)
                except Exception:
                    try:
                        param_names = dict(wandb_run.config).get("param_keys", None)
                    except Exception:
                        param_names = None

                log_dict = {
                    "phase2/epoch": epoch_offset + epoch,
                    "phase2/train_loss": avg_train_loss,
                    "phase2/val_loss": val_loss,
                    "phase2/val_r2": r2,
                }
                # add per-target entries, using labels when available
                for i, r in enumerate(r2_per_target):
                    if param_names and i < len(param_names) and param_names[i] is not None:
                        label = str(param_names[i])
                    else:
                        label = f"target_{i}"
                    # sanitize label for wandb key
                    safe_label = label.replace("/", "_").replace(" ", "_")
                    log_dict[f"phase2/val_r2_{safe_label}"] = r

                wandb_run.log(log_dict)
            # Additionally log sample feature distributions and sample weights
            if wandb_run is not None and sample_X is not None:
                try:
                    model.eval()
                    # create dummy y for normalization
                    y_dummy = torch.zeros(sample_X.shape[0], device=sample_X.device)
                    Xn, maskn, _ = normalize_batch(sample_X, sample_mask, y_dummy, input_norm, input_stats, output_norm, output_stats)
                    with torch.no_grad():
                        feat_sample, weights_sample = model.compute_summary_features(Xn, maskn)

                    # feature-wise mean and std
                    feat_mean = feat_sample.mean(dim=0).cpu().numpy()
                    feat_std = feat_sample.std(dim=0).cpu().numpy()

                    log_dict2 = {"phase2/epoch": epoch_offset + epoch}
                    # Limit number of logged features to avoid huge logs
                    max_log_feats = min(50, feat_mean.shape[0])
                    if False:
                        for i in range(max_log_feats):
                            log_dict2[f"phase2/feat_mean_{i}"] = float(feat_mean[i])
                            log_dict2[f"phase2/feat_std_{i}"] = float(feat_std[i])

                    # log sample weights for first batch element if available
                    if False and weights_sample is not None:
                        w0 = weights_sample[0].cpu().numpy()
                        # store as a list under one key or one key per kernel
                        if w0.ndim == 1:
                            log_dict2["phase2/sample_weights_0"] = w0.tolist()
                        else:
                            max_kernels = min(4, w0.shape[1])
                            for k in range(max_kernels):
                                log_dict2[f"phase2/sample_weights_0_k{k}"] = w0[:, k].tolist()

                    wandb_run.log(log_dict2)
                except Exception:
                    pass
    
    return history


def evaluate_model(
    model: SimplifiedSetDSR,
    loader: DataLoader,
    device: torch.device,
    input_norm: str = "none",
    input_stats: dict = None,
    output_norm: str = "none",
    output_stats: dict = None,
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    eps = 1e-8
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
            
            pred = model(X, mask)
            loss = F.mse_loss(pred, y)
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(pred)
            all_targets.append(y)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Overall metrics
    mse = F.mse_loss(all_preds, all_targets).item()
    ss_res = ((all_targets - all_preds) ** 2).sum()
    ss_tot = ((all_targets - all_targets.mean(dim=0)) ** 2).sum() + eps
    r2 = (1 - ss_res / ss_tot).item()
    
    # Per-target R²
    n_targets = all_targets.shape[1] if all_targets.dim() > 1 else 1
    r2_per_target = []
    for k in range(n_targets):
        if all_targets.dim() > 1:
            y_k = all_targets[:, k]
            p_k = all_preds[:, k]
        else:
            y_k = all_targets
            p_k = all_preds
        ss_res_k = ((y_k - p_k) ** 2).sum()
        ss_tot_k = ((y_k - y_k.mean()) ** 2).sum() + eps
        r2_per_target.append((1 - ss_res_k / ss_tot_k).item())
    
    return {
        "loss": total_loss / max(n_batches, 1),
        "mse": mse,
        "r2": r2,
        "r2_per_target": r2_per_target,
    }


def collect_predictions(
    model: SimplifiedSetDSR,
    loader: DataLoader,
    device: torch.device,
    input_norm: str = "none",
    input_stats: dict = None,
    output_norm: str = "none",
    output_stats: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect predictions and targets in original units."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            
            # Keep original y for output
            y_orig = y.clone()
            
            # Normalize for prediction
            X_norm, mask, y_norm = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
            
            pred_norm = model(X_norm, mask)
            
            # Inverse transform predictions
            pred = _inverse_output_norm(pred_norm, output_norm, output_stats, device)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_orig.cpu().numpy())
    
    return np.concatenate(all_targets, axis=0), np.concatenate(all_preds, axis=0)


def train_simplified_dsr_full(
    model: SimplifiedSetDSR,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    n_generations: int = 50,
    n_weight_epochs: int = 20,
    lr: float = 1e-3,
    lr_head: Optional[float] = None,
    lr_weight: Optional[float] = None,
    weight_decay: float = 1e-4,
    complexity_weight: float = 0.01,
    log_interval: int = 10,
    weight_retrain_interval: int = 10,
    scheduler_name: str = "warmup_cosine",
    warmup_frac: float = 0.1,
    min_lr_scale: float = 0.05,
    profile: bool = False,
    profile_epochs: int = 2,
    profile_steps: bool = False,
    profile_steps_epochs: int = 2,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    input_norm: str = "none",
    input_stats: dict = None,
    output_norm: str = "none",
    output_stats: dict = None,
    wandb_run=None,
) -> Dict:
    """
    Full training loop for Simplified Set-DSR with wandb logging.
    
    If symbolic transforms are enabled:
        Alternates between:
        1. Evolving per-element transforms (GP search)
        2. Training weights and head (gradient descent)
    
    If symbolic transforms are disabled:
        Only trains weights and head via gradient descent.
    """
    model = model.to(device)
    
    history = {
        "best_fitness": [],
        "mean_fitness": [],
        "best_expressions": [],
    }
    
    if profile_steps and hasattr(model, "enable_step_profiling"):
        model.enable_step_profiling(True)

    # If symbolic transforms are disabled, just train weights and head
    if not model.use_symbolic_transforms:
        print(f"\n{'='*60}")
        print("Training Simplified Set-DSR (no symbolic transforms)")
        print(f"{'='*60}")
        print(f"Summary features: {model.n_summary_features}")
        print(f"Top-K: {model.top_k if model.use_top_k else 'disabled'}")
        print(f"{'='*60}\n")
        
        # Just train weights and head
        total_epochs = n_weight_epochs  # Use same total training time
        final_history = train_weights_and_head(
            model, train_loader, val_loader, device,
            n_epochs=total_epochs, lr=lr,
            lr_head=lr_head, lr_weight=lr_weight,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            warmup_frac=warmup_frac,
            min_lr_scale=min_lr_scale,
            profile=profile,
            profile_epochs=profile_epochs,
            profile_steps=profile_steps,
            profile_steps_epochs=profile_steps_epochs,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            input_norm=input_norm, input_stats=input_stats,
            output_norm=output_norm, output_stats=output_stats,
            wandb_run=wandb_run,
            epoch_offset=0,
        )
        
        history["final_train_loss"] = final_history["train_loss"]
        history["final_val_r2"] = final_history.get("val_r2", [])
        history["best_expressions"] = model.get_transform_expressions()
        history["selected_features"] = model.get_selected_feature_names()
        
        print(f"\nTraining complete!")
        if final_history.get("val_r2"):
            print(f"Final val R²: {final_history['val_r2'][-1]:.4f}")
        print(f"\nFeatures used: {history['best_expressions'][:5]}...")
        print(f"\nSelected features: {history['selected_features'][:10]}...")
        
        return history
    
    # Symbolic transforms enabled - use GP evolution
    # Concatenate all training data for GP evaluation
    all_X, all_mask, all_y = [], [], []
    for batch in train_loader:
        X, mask, y = batch
        all_X.append(X)
        all_mask.append(mask)
        all_y.append(y)
    
    X_train = torch.cat(all_X, dim=0).to(device)
    mask_train = torch.cat(all_mask, dim=0).to(device)
    y_train = torch.cat(all_y, dim=0).to(device)
    
    # Normalize training data for GP evaluation
    X_train_norm, mask_train, y_train_norm = normalize_batch(
        X_train, mask_train, y_train, input_norm, input_stats, output_norm, output_stats
    )
    
    # Initialize evolver
    evolver = TransformEvolver(model)
    
    print(f"\n{'='*60}")
    print("Training Simplified Set-DSR")
    print(f"{'='*60}")
    print(f"Population size: {evolver.population_size}")
    print(f"Generations: {n_generations}")
    print(f"Transforms: {model.n_transforms}")
    print(f"Summary features: {model.n_summary_features}")
    print(f"Top-K: {model.top_k if model.use_top_k else 'disabled'}")
    print(f"{'='*60}\n")
    
    for gen in range(n_generations):
        # Evaluate fitness
        if profile_steps and hasattr(model, "reset_profile_stats"):
            model.reset_profile_stats()
            if device.type == "cuda":
                torch.cuda.synchronize()
        t_gp = time.perf_counter() if profile_steps else None
        fitness_scores = evolver.evaluate_fitness(
            X_train_norm, mask_train, y_train_norm,
            complexity_weight=complexity_weight,
        )
        if t_gp is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            gp_time = time.perf_counter() - t_gp
            stats = getattr(model, "profile_stats", {})
            summary_total = stats.get("summary_total", 0.0)
            print(
                f"GP profile gen {gen}: "
                f"gp_total={gp_time:.3f}s "
                f"summary_total={summary_total:.3f}s "
                f"weights={stats.get('weights', 0.0):.3f}s "
                f"transform={stats.get('transform', 0.0):.3f}s "
                f"stats={stats.get('stats', 0.0):.3f}s"
            )
        
        best_expr, best_fitness = evolver.get_best()
        mean_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        
        history["best_fitness"].append(best_fitness)
        history["mean_fitness"].append(mean_fitness)
        
        # Log to wandb
        if wandb_run is not None:
            log_dict = {
                "evo/generation": gen,
                "evo/best_fitness": best_fitness,
                "evo/mean_fitness": mean_fitness,
                "evo/std_fitness": std_fitness,
            }
            wandb_run.log(log_dict)
        
        # Log to console
        if gen % log_interval == 0:
            print(f"Gen {gen:3d} | Best: {best_fitness:.4f} | Mean: {mean_fitness:.4f}")
            
            # Show best expressions
            model.transform.expressions = best_expr
            expr_strs = model.get_transform_expressions()
            for i, e in enumerate(expr_strs[:3]):  # show first 3
                print(f"  g{i}: {e}")
        
        # Periodically retrain weights
        if gen > 0 and gen % weight_retrain_interval == 0:
            print(f"  → Retraining weights and head...")
            model.transform.expressions = best_expr
            train_weights_and_head(
                model, train_loader, val_loader, device,
                n_epochs=n_weight_epochs, lr=lr,
                lr_head=lr_head, lr_weight=lr_weight,
                weight_decay=weight_decay,
                scheduler_name=scheduler_name,
                warmup_frac=warmup_frac,
                min_lr_scale=min_lr_scale,
                profile=profile,
                profile_epochs=profile_epochs,
                profile_steps=profile_steps,
                profile_steps_epochs=profile_steps_epochs,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                input_norm=input_norm, input_stats=input_stats,
                output_norm=output_norm, output_stats=output_stats,
                wandb_run=wandb_run,
                epoch_offset=gen * n_weight_epochs,
            )
            
            # Evaluate on validation
            if val_loader is not None:
                val_metrics = evaluate_model(
                    model, val_loader, device,
                    input_norm=input_norm, input_stats=input_stats,
                    output_norm=output_norm, output_stats=output_stats,
                )
                print(f"  → Val R²: {val_metrics['r2']:.4f}")
                
                if wandb_run is not None:
                    wandb_run.log({
                        "evo/generation": gen,
                        "evo/val_r2": val_metrics["r2"],
                        "evo/val_mse": val_metrics["mse"],
                    })
        
        # Evolve
        evolver.evolve_generation()
    
    # Final weight training
    print("\nFinal weight training...")
    best_expr, best_fitness = evolver.get_best()
    model.transform.expressions = best_expr
    final_history = train_weights_and_head(
        model, train_loader, val_loader, device,
        n_epochs=n_weight_epochs * 2, lr=lr,
        lr_head=lr_head, lr_weight=lr_weight,
        weight_decay=weight_decay,
        scheduler_name=scheduler_name,
        warmup_frac=warmup_frac,
        min_lr_scale=min_lr_scale,
        profile=profile,
        profile_epochs=profile_epochs,
        profile_steps=profile_steps,
        profile_steps_epochs=profile_steps_epochs,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        input_norm=input_norm, input_stats=input_stats,
        output_norm=output_norm, output_stats=output_stats,
        wandb_run=wandb_run,
        epoch_offset=n_generations * n_weight_epochs,
    )
    
    history["final_train_loss"] = final_history["train_loss"]
    history["final_val_r2"] = final_history.get("val_r2", [])
    history["best_expressions"] = model.get_transform_expressions()
    history["selected_features"] = model.get_selected_feature_names()
    
    print(f"\nTraining complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    if final_history.get("val_r2"):
        print(f"Final val R²: {final_history['val_r2'][-1]:.4f}")
    print(f"\nBest expressions:")
    for i, expr in enumerate(history["best_expressions"]):
        print(f"  g{i}: {expr}")
    print(f"\nSelected features: {history['selected_features'][:10]}...")  # show first 10
    
    return history


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Simplified Set-DSR model")
    
    # Data arguments
    parser.add_argument("--h5-path", type=str, default="../data/camels_LH.hdf5",
                        help="Path to CAMELS LH HDF5 file")
    parser.add_argument("--snap", type=int, default=90,
                        help="Snapshot number to read")
    parser.add_argument("--param-keys", nargs="+", default=None,
                        help="List of parameter keys to use as targets")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Fraction of data for training")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of data for validation")
    parser.add_argument("--test-frac", type=float, default=0.1,
                        help="Fraction of data for testing")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for gradient descent phases")
    
    # Normalization
    parser.add_argument("--normalize-input", choices=["none", "log", "log_std"],
                        default="log_std", help="Input normalization")
    parser.add_argument("--normalize-output", choices=["none", "minmax"],
                        default="minmax", help="Output normalization")
    
    # Model arguments
    parser.add_argument("--n-transforms", type=int, default=4,
                        help="Number of per-element symbolic transforms")
    parser.add_argument("--top-k", type=int, default=16,
                        help="Number of top features to select")
    parser.add_argument("--include-moments", action="store_true", default=False,
                        help="Include raw/central moments in statistics")
    parser.add_argument("--include-cumulants", action="store_true", default=False,
                        help="Include cumulants in statistics")
    parser.add_argument("--include-quantiles", action="store_true", default=False,
                        help="Include soft quantiles in statistics")
    parser.add_argument("--use-learnable-weights", action="store_true", default=False,
                        help="Use learnable per-element weights")
    parser.add_argument("--selection-method", choices=["learnable", "correlation", "variance", "fixed"],
                        default="learnable", help="Top-K selection method")
    parser.add_argument("--use-mlp-head", action="store_true", default=False,
                        help="Use MLP (vs linear) for final prediction")
    parser.add_argument("--use-symbolic-transforms", action="store_true", default=False,
                        help="Use symbolic per-element transforms (default: use raw features)")
    parser.add_argument("--use-top-k", action="store_true", default=False,
                        help="Use top-K feature selection (default: use all stats)")
    parser.add_argument("--head-hidden-dims", nargs="+", type=int, default=[128, 128],
                        help="Hidden dimensions for MLP head")
    parser.add_argument("--weight-hidden-dims", nargs="+", type=int, default=[32, 16],
                        help="Hidden dimensions for weight network")
    parser.add_argument("--n-weight-kernels", type=int, default=1,
                        help="Number of learnable weight kernels for summary stats")
    
    # Training arguments - Evolution
    parser.add_argument("--n-generations", type=int, default=50,
                        help="Number of GP evolution generations")
    parser.add_argument("--population-size", type=int, default=50,
                        help="Population size for GP")
    parser.add_argument("--complexity-weight", type=float, default=0.01,
                        help="Penalty weight for expression complexity")
    parser.add_argument("--weight-retrain-interval", type=int, default=10,
                        help="Retrain weights every N generations")
    
    # Training arguments - Gradient descent
    parser.add_argument("--n-weight-epochs", type=int, default=20,
                        help="Epochs for weight/head training per interval")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for gradient descent")
    parser.add_argument("--lr-head", type=float, default=None,
                        help="Learning rate for prediction head (defaults to --lr)")
    parser.add_argument("--lr-weight", type=float, default=None,
                        help="Learning rate for weight network (defaults to --lr)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--scheduler", choices=["cosine", "warmup_cosine", "plateau"],
                        default="warmup_cosine", help="LR scheduler type")
    parser.add_argument("--warmup-frac", type=float, default=0.1,
                        help="Warmup fraction for warmup_cosine scheduler")
    parser.add_argument("--min-lr-scale", type=float, default=0.05,
                        help="Minimum LR scale for warmup_cosine or plateau")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (CUDA only)")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16",
                        help="AMP dtype (fp16 or bf16)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile on the model (PyTorch 2.x)")
    parser.add_argument("--profile", action="store_true",
                        help="Print basic time profile for training epochs")
    parser.add_argument("--profile-epochs", type=int, default=2,
                        help="Number of epochs to profile")
    parser.add_argument("--profile-steps", action="store_true",
                        help="Print per-step timing for summary stats and head")
    parser.add_argument("--profile-steps-epochs", type=int, default=2,
                        help="Number of epochs to profile per-step timings")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Log every N generations")
    parser.add_argument("--wandb", action="store_true",
                        help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="simplified-set-dsr",
                        help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="WandB entity (team/user)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="WandB run name")
    
    # Saving
    parser.add_argument("--save-dir", type=str, default="data/models/stats_sym/",
                        help="Directory to save model and results")
    parser.add_argument("--save-model", action="store_true",
                        help="Save model checkpoint")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save prediction plots")
    
    args = parser.parse_args(argv)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================================================
    # Data Loading
    # ==========================================================================
    
    print("\nLoading data...")
    
    # Probe dataset for param names
    probe_ds = HDF5SetDataset(
        h5_path=args.h5_path,
        snap=args.snap,
        param_keys=None,
        data_field="SubhaloStellarMass",
    )
    detected_param_names = getattr(probe_ds, 'param_names', None)
    
    # Resolve param keys
    if args.param_keys is not None and detected_param_names is not None:
        mapped_keys = []
        for p_or_idx in args.param_keys:
            try:
                idx = int(p_or_idx)
                if 0 <= idx < len(detected_param_names):
                    mapped_keys.append(detected_param_names[idx])
                else:
                    mapped_keys.append(p_or_idx)
            except ValueError:
                mapped_keys.append(p_or_idx)
        args.param_keys = mapped_keys
    elif args.param_keys is None and detected_param_names is not None:
        args.param_keys = detected_param_names
    
    # Create dataset
    full_ds = HDF5SetDataset(
        h5_path=args.h5_path,
        snap=args.snap,
        param_keys=args.param_keys,
        data_field="SubhaloStellarMass",
    )
    
    n_total = len(full_ds)
    print(f"Dataset size: {n_total}")
    print(f"Target params: {args.param_keys}")
    
    # Compute splits
    train_size = int(round(args.train_frac * n_total))
    val_size = int(round(args.val_frac * n_total))
    test_size = int(round(args.test_frac * n_total))
    
    # Adjust for rounding
    total_assigned = train_size + val_size + test_size
    if total_assigned < n_total:
        train_size += n_total - total_assigned
    elif total_assigned > n_total:
        train_size -= total_assigned - n_total
    
    print(f"Split sizes -> train: {train_size}, val: {val_size}, test: {test_size}")
    
    # Create subsets
    indices = list(range(n_total))
    train_ds = Subset(full_ds, indices[:train_size])
    val_ds = Subset(full_ds, indices[train_size:train_size + val_size])
    test_ds = Subset(full_ds, indices[train_size + val_size:]) if test_size > 0 else None
    
    # Compute normalization stats
    input_stats = None
    output_stats = None
    if args.normalize_input != "none" or args.normalize_output != "none":
        print("Computing normalization statistics...")
        stats = compute_stats_from_dataset(train_ds)
        input_stats = {"input_mean": stats["input_mean"], "input_std": stats["input_std"]}
        output_stats = {"y_min": stats["y_min"], "y_max": stats["y_max"]}
    
    # Create collate function
    max_size = full_ds.max_size
    collate_fn = lambda batch: hdf5_collate(batch, max_size=max_size)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if test_ds else None
    
    # Infer dimensions
    sample_x, sample_y = full_ds[0]
    n_features = sample_x.shape[1] if sample_x.ndim == 2 else 1
    output_dim = sample_y.shape[0] if sample_y.ndim >= 1 else 1
    
    print(f"Input features: {n_features}, Output dim: {output_dim}")
    
    # ==========================================================================
    # Initialize WandB
    # ==========================================================================
    
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
            )
            if input_stats is not None:
                wandb.config.update({
                    "input_mean": input_stats["input_mean"].tolist(),
                    "input_std": input_stats["input_std"].tolist(),
                })
            if output_stats is not None:
                wandb.config.update({
                    "y_min": output_stats["y_min"].tolist(),
                    "y_max": output_stats["y_max"].tolist(),
                })
            print(f"WandB initialized: {wandb.run.name}")
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
            wandb_run = None
    
    # ==========================================================================
    # Create Model
    # ==========================================================================
    
    print("\nCreating model...")
    
    model = SimplifiedSetDSR(
        n_features=n_features,
        n_transforms=args.n_transforms,
        output_dim=output_dim,
        top_k=args.top_k,
        include_moments=args.include_moments,
        include_cumulants=args.include_cumulants,
        include_quantiles=args.include_quantiles,
        use_learnable_weights=args.use_learnable_weights,
        weight_hidden_dims=args.weight_hidden_dims,
        n_weight_kernels=args.n_weight_kernels,
        selection_method=args.selection_method,
        use_mlp_head=args.use_mlp_head,
        head_hidden_dims=args.head_hidden_dims,
        use_symbolic_transforms=args.use_symbolic_transforms,
        use_top_k=args.use_top_k,
    )
    
    model = model.to(device)

    print("Computing summary-statistics feature normalization...")
    orig_compute_summary_features = model.compute_summary_features
    summary_mean, summary_std = compute_summary_feature_stats(
        model=model,
        loader=train_loader,
        device=device,
        input_norm=args.normalize_input,
        input_stats=input_stats,
    )
    summary_mean = summary_mean.to(device)
    summary_std = summary_std.to(device)

    def _compute_summary_features_norm(X, mask):
        feats, weights = orig_compute_summary_features(X, mask)
        feats = apply_summary_feature_norm(feats, summary_mean, summary_std)
        return feats, weights

    model.compute_summary_features = _compute_summary_features_norm
    model.summary_feature_mean = summary_mean
    model.summary_feature_std = summary_std

    if args.compile:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as exc:
            print(f"Warning: torch.compile failed: {exc}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Summary features: {model.n_summary_features}")
    print(f"Statistics per transform: {model.summary_stats.n_stats}")
    
    # Log model info to wandb
    if wandb_run is not None:
        wandb.config.update({
            "n_params": n_params,
            "n_summary_features": model.n_summary_features,
            "n_stats_per_transform": model.summary_stats.n_stats,
        })
    
    # ==========================================================================
    # Train
    # ==========================================================================
    
    print("\nStarting training...")
    
    history = train_simplified_dsr_full(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_generations=args.n_generations,
        n_weight_epochs=args.n_weight_epochs,
        lr=args.lr,
        lr_head=args.lr_head,
        lr_weight=args.lr_weight,
        weight_decay=args.weight_decay,
        complexity_weight=args.complexity_weight,
        log_interval=args.log_interval,
        weight_retrain_interval=args.weight_retrain_interval,
        scheduler_name=args.scheduler,
        warmup_frac=args.warmup_frac,
        min_lr_scale=args.min_lr_scale,
        use_amp=args.amp,
        amp_dtype=args.amp_dtype,
        profile=args.profile,
        profile_epochs=args.profile_epochs,
        profile_steps=args.profile_steps,
        profile_steps_epochs=args.profile_steps_epochs,
        input_norm=args.normalize_input,
        input_stats=input_stats,
        output_norm=args.normalize_output,
        output_stats=output_stats,
        wandb_run=wandb_run,
    )
    
    # ==========================================================================
    # Final Evaluation
    # ==========================================================================
    
    print("\nFinal evaluation...")
    
    # Validation metrics
    val_metrics = evaluate_model(
        model, val_loader, device,
        input_norm=args.normalize_input, input_stats=input_stats,
        output_norm=args.normalize_output, output_stats=output_stats,
    )
    print(f"Validation - Loss: {val_metrics['loss']:.6f}, MSE: {val_metrics['mse']:.6e}, R²: {val_metrics['r2']:.4f}")
    
    # Per-target R²
    if args.param_keys is not None:
        for i, (key, r2) in enumerate(zip(args.param_keys, val_metrics['r2_per_target'])):
            print(f"  {key}: R² = {r2:.4f}")
    
    # Test metrics
    if test_loader is not None:
        test_metrics = evaluate_model(
            model, test_loader, device,
            input_norm=args.normalize_input, input_stats=input_stats,
            output_norm=args.normalize_output, output_stats=output_stats,
        )
        print(f"Test - Loss: {test_metrics['loss']:.6f}, MSE: {test_metrics['mse']:.6e}, R²: {test_metrics['r2']:.4f}")
        
        if args.param_keys is not None:
            for i, (key, r2) in enumerate(zip(args.param_keys, test_metrics['r2_per_target'])):
                print(f"  {key}: R² = {r2:.4f}")
    
    # Log final metrics to wandb
    if wandb_run is not None:
        final_log = {
            "final/val_loss": val_metrics["loss"],
            "final/val_mse": val_metrics["mse"],
            "final/val_r2": val_metrics["r2"],
        }
        if args.param_keys is not None:
            for i, (key, r2) in enumerate(zip(args.param_keys, val_metrics['r2_per_target'])):
                final_log[f"final/val_r2_{key}"] = r2
        
        if test_loader is not None:
            final_log["final/test_loss"] = test_metrics["loss"]
            final_log["final/test_mse"] = test_metrics["mse"]
            final_log["final/test_r2"] = test_metrics["r2"]
            if args.param_keys is not None:
                for i, (key, r2) in enumerate(zip(args.param_keys, test_metrics['r2_per_target'])):
                    final_log[f"final/test_r2_{key}"] = r2
        
        wandb.log(final_log)
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save model
    if args.save_model:
        model_path = os.path.join(args.save_dir, f"simplified_set_dsr_{timestamp}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "expressions": history["best_expressions"],
            "selected_features": history["selected_features"],
            "args": vars(args),
            "input_stats": input_stats,
            "output_stats": output_stats,
            "summary_feature_mean": getattr(model, "summary_feature_mean", None),
            "summary_feature_std": getattr(model, "summary_feature_std", None),
        }, model_path)
        print(f"Saved model to {model_path}")
        
        if wandb_run is not None:
            wandb.save(model_path)
    
    # Save history/expressions
    results = {
        "expressions": history["best_expressions"],
        "selected_features": history["selected_features"],
        "final_val_r2": history.get("final_val_r2", []),
        "best_fitness": history["best_fitness"],
        "param_keys": args.param_keys,
    }
    results_path = os.path.join(args.save_dir, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    # Save plots
    if args.save_plots:
        plot_dir = os.path.join(args.save_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Validation plot
        y_true_val, y_pred_val = collect_predictions(
            model, val_loader, device,
            input_norm=args.normalize_input, input_stats=input_stats,
            output_norm=args.normalize_output, output_stats=output_stats,
        )
        val_plot_path = os.path.join(plot_dir, f"val_pred_vs_true_{timestamp}.png")
        plot_pred_vs_true(y_true_val, y_pred_val, val_plot_path, title="Validation")
        print(f"Saved validation plot to {val_plot_path}")
        
        if wandb_run is not None:
            wandb.log({"plots/val_pred_vs_true": wandb.Image(val_plot_path)})
        
        # Test plot
        if test_loader is not None:
            y_true_test, y_pred_test = collect_predictions(
                model, test_loader, device,
                input_norm=args.normalize_input, input_stats=input_stats,
                output_norm=args.normalize_output, output_stats=output_stats,
            )
            test_plot_path = os.path.join(plot_dir, f"test_pred_vs_true_{timestamp}.png")
            plot_pred_vs_true(y_true_test, y_pred_test, test_plot_path, title="Test")
            print(f"Saved test plot to {test_plot_path}")
            
            if wandb_run is not None:
                wandb.log({"plots/test_pred_vs_true": wandb.Image(test_plot_path)})
    
    # Finish wandb
    if wandb_run is not None:
        wandb.finish()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
