"""
Training script for Set-DSR model.

Two-phase training:
1. Evolutionary search (GP or RL) to find good symbolic programs
2. Fine-tune MLP head with fixed programs

Aligned with train.py: uses HDF5SetDataset, wandb logging, normalization, plotting.

Advanced features (toggleable):
- Subtree mutation (--use-subtree-mutation)
- Constant optimization (--use-constant-optimization)
- RL-based search instead of GP (--use-rl-search)
- SymPy simplification (--use-sympy-simplify)
"""

import argparse
import functools
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm

from set_dsr import (
    SetDSR,
    SetDSREvolver,
    AdvancedSetDSREvolver,
    ProgramPolicyNetwork,
    RLProgramSearcher,
    ParameterizedSetDSR,
    extract_expressions,
    print_model_summary,
    simplify_all_expressions,
    expressions_to_latex,
)
from data import HDF5SetDataset, hdf5_collate


# =============================================================================
# Normalization utilities (matching train.py)
# =============================================================================

def compute_stats_from_dataset(dataset, sample_limit: int = 1000):
    """Estimate input (log) mean/std and output min/max from `dataset`."""
    n = len(dataset)
    if n == 0:
        raise ValueError("Empty dataset for stats computation")
    
    step = 1  # use entire dataset
    
    sum_x = None
    sum_x2 = None
    count = 0
    y_mins = []
    y_maxs = []
    
    for i in range(0, n, step):
        x, y = dataset[i]
        lx = np.log1p(x.astype(np.float64))
        s = lx.sum(axis=0)
        s2 = (lx ** 2).sum(axis=0)
        if sum_x is None:
            sum_x = s
            sum_x2 = s2
        else:
            sum_x += s
            sum_x2 += s2
        count += lx.shape[0]
        
        y_arr = np.array(y, dtype=np.float64)
        y_mins.append(np.min(y_arr, axis=0))
        y_maxs.append(np.max(y_arr, axis=0))
    
    mean = (sum_x / count).astype(np.float32)
    var = (sum_x2 / count - mean.astype(np.float64) ** 2).clip(min=1e-12)
    std = np.sqrt(var).astype(np.float32)
    
    y_min = np.min(np.stack(y_mins, axis=0), axis=0).astype(np.float32)
    y_max = np.max(np.stack(y_maxs, axis=0), axis=0).astype(np.float32)
    
    y_min = np.atleast_1d(y_min)
    y_max = np.atleast_1d(y_max)
    
    return {"input_mean": mean, "input_std": std, "y_min": y_min, "y_max": y_max}


def normalize_batch(X, mask, y, input_norm: str, input_stats: dict, 
                    output_norm: str, output_stats: dict, eps: float = 1e-8):
    """Apply normalizations to a batch of tensors."""
    if input_norm is None or input_norm == "none":
        pass
    else:
        if input_norm in ("log", "log_std"):
            X = torch.log1p(X)
        if input_norm == "log_std":
            if input_stats is None:
                raise ValueError("input_stats required for 'log_std' normalization")
            mean_np = np.atleast_1d(input_stats["input_mean"])
            std_np = np.atleast_1d(input_stats["input_std"])
            mean = torch.from_numpy(mean_np).to(X.dtype).to(X.device)
            std = torch.from_numpy(std_np).to(X.dtype).to(X.device)
            if X.ndim == 3:
                X = (X - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + eps)
            elif X.ndim == 2:
                X = (X - mean.view(1, -1)) / (std.view(1, -1) + eps)
            else:
                X = (X - mean) / (std + eps)
    
    if output_norm is None or output_norm == "none":
        pass
    else:
        if output_norm == "minmax":
            y_min = torch.from_numpy(output_stats["y_min"]).to(y.dtype).to(y.device)
            y_max = torch.from_numpy(output_stats["y_max"]).to(y.dtype).to(y.device)
            if y.dim() == 1:
                y = (y - y_min) / (y_max - y_min + eps)
            else:
                y = (y - y_min.view(1, -1)) / (y_max.view(1, -1) - y_min.view(1, -1) + eps)
    
    return X, mask, y


def inverse_output_norm(tensor: torch.Tensor, output_norm: str, output_stats: dict, 
                        device: torch.device, eps: float = 1e-8):
    """Inverse output normalization to get original scale."""
    if output_norm is None or output_norm == "none" or output_stats is None:
        return tensor
    if output_norm == "minmax":
        y_min = torch.from_numpy(output_stats["y_min"]).to(tensor.dtype).to(device)
        y_max = torch.from_numpy(output_stats["y_max"]).to(tensor.dtype).to(device)
        if tensor.dim() == 1 or (tensor.dim() == 2 and y_min.numel() == 1):
            return tensor * (y_max - y_min + eps) + y_min
        else:
            return tensor * (y_max.view(1, -1) - y_min.view(1, -1) + eps) + y_min.view(1, -1)
    return tensor


# =============================================================================
# Plotting utilities (matching train.py)
# =============================================================================

def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, 
                      title: str = "pred vs true", param_names: List[str] = None):
    """Generate scatter plots of predictions vs true values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    N, K = y_true.shape
    
    if K == 1:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true.ravel(), y_pred.ravel(), alpha=0.4, s=8)
        mn = float(min(np.nanmin(y_true), np.nanmin(y_pred)))
        mx = float(max(np.nanmax(y_true), np.nanmax(y_pred)))
        if math.isfinite(mn) and math.isfinite(mx):
            plt.plot([mn, mx], [mn, mx], 'k--')
        plt.xlabel('true')
        plt.ylabel('pred')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return
    
    cols = min(3, K)
    rows = math.ceil(K / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = np.array(axs).reshape(-1)
    
    for k in range(K):
        ax = axs[k]
        xt = y_true[:, k]
        xp = y_pred[:, k]
        ax.scatter(xt, xp, alpha=0.4, s=8)
        mn = float(min(np.nanmin(xt), np.nanmin(xp)))
        mx = float(max(np.nanmax(xt), np.nanmax(xp)))
        if math.isfinite(mn) and math.isfinite(mx):
            ax.plot([mn, mx], [mn, mx], 'k--')
        ax.set_xlabel('true')
        ax.set_ylabel('pred')
        label = param_names[k] if param_names and k < len(param_names) else f"target {k}"
        ax.set_title(f'{title} ({label})')
    
    for j in range(K, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def collect_predictions(model, loader, device,
                        input_norm: str = "none", input_stats: dict = None,
                        output_norm: str = "none", output_stats: dict = None):
    """Run model on loader and return (y_true_np, y_pred_np) in original units."""
    model.eval()
    y_trues = []
    y_preds = []
    
    with torch.no_grad():
        for batch in loader:
            X, mask, y = batch
            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device)
            
            y_true_np = y.cpu().numpy()
            if y_true_np.ndim == 1:
                y_true_np = y_true_np.reshape(-1, 1)
            
            Xn, maskn, yn = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
            pred_n = model(Xn, maskn)
            pred_orig = inverse_output_norm(pred_n, output_norm, output_stats, device)
            
            pred_np = pred_orig.cpu().numpy()
            if pred_np.ndim == 1:
                pred_np = pred_np.reshape(-1, 1)
            
            y_trues.append(y_true_np)
            y_preds.append(pred_np)
    
    return np.concatenate(y_trues, axis=0), np.concatenate(y_preds, axis=0)


# =============================================================================
# Metrics computation
# =============================================================================

def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> Dict:
    """Compute MSE, relative error, R2 for predictions."""
    B = y_true.shape[0]
    y_flat = y_true.view(B, -1)
    pred_flat = y_pred.view(B, -1)
    
    se = (pred_flat - y_flat) ** 2
    mse = se.mean().item()
    
    abs_rel = (torch.abs(pred_flat - y_flat) / (y_flat.abs() + eps)).mean().item()
    
    # R2
    ss_res = se.sum().item()
    mean_y = y_flat.mean().item()
    ss_tot = ((y_flat - mean_y) ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + eps)
    
    # Per-target metrics
    K = y_flat.shape[1]
    mse_per = (se.mean(dim=0)).cpu().numpy()
    
    mean_y_per = y_flat.mean(dim=0)
    ss_res_per = se.sum(dim=0)
    ss_tot_per = ((y_flat - mean_y_per.unsqueeze(0)) ** 2).sum(dim=0)
    r2_per = (1.0 - ss_res_per / (ss_tot_per + eps)).cpu().numpy()
    
    return {
        "mse": mse,
        "rel_err": abs_rel,
        "r2": r2,
        "mse_per_target": mse_per,
        "r2_per_target": r2_per,
    }


# =============================================================================
# Training loops
# =============================================================================

def train_evolutionary(
    model: SetDSR,
    evolver,  # Can be SetDSREvolver or AdvancedSetDSREvolver
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_generations: int = 100,
    complexity_weight: float = 0.01,
    curriculum_schedule: dict = None,
    log_interval: int = 10,
    input_norm: str = "none",
    input_stats: dict = None,
    output_norm: str = "none",
    output_stats: dict = None,
    wandb = None,
    param_names: List[str] = None,
):
    """Phase 1: Evolutionary search for symbolic programs (GP-based)."""
    print("\n" + "=" * 60)
    print("Phase 1: Evolutionary Search (GP)")
    print("=" * 60)
    
    # Collect all training data into tensors for GP evaluation
    X_all, mask_all, y_all = [], [], []
    for batch in train_loader:
        X, mask, y = batch
        X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
        X_all.append(X)
        mask_all.append(mask)
        y_all.append(y)
    
    X_train = torch.cat(X_all, dim=0).to(device)
    mask_train = torch.cat(mask_all, dim=0).to(device)
    y_train = torch.cat(y_all, dim=0).to(device)
    
    # Same for validation
    X_val_all, mask_val_all, y_val_all = [], [], []
    for batch in val_loader:
        X, mask, y = batch
        X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
        X_val_all.append(X)
        mask_val_all.append(mask)
        y_val_all.append(y)
    
    X_val = torch.cat(X_val_all, dim=0).to(device)
    mask_val = torch.cat(mask_val_all, dim=0).to(device)
    y_val = torch.cat(y_val_all, dim=0).to(device)
    
    best_fitness = float("inf")
    best_programs = None
    history = []
    
    for gen in range(n_generations):
        # Curriculum update
        if curriculum_schedule and gen in curriculum_schedule:
            new_level = curriculum_schedule[gen]
            print(f"\n[Gen {gen}] Advancing curriculum to level {new_level}")
            model.set_curriculum_level(new_level)
            evolver.model = model
        
        # Evolve one generation
        best_ind, fitness = evolver.evolve_generation(
            X_train, mask_train, y_train, complexity_weight
        )
        
        # Evaluate on validation set
        model.programs = best_ind
        with torch.no_grad():
            val_pred = model(X_val, mask_val)
            val_metrics = compute_metrics(y_val, val_pred)
        
        history.append({
            "generation": gen,
            "train_fitness": fitness,
            "val_mse": val_metrics["mse"],
            "val_r2": val_metrics["r2"],
            "complexity": model.total_complexity(),
        })
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_programs = [p for p in best_ind]
        
        if gen % log_interval == 0:
            print(f"[Gen {gen:4d}] fitness={fitness:.4f}, val_mse={val_metrics['mse']:.4e}, "
                  f"val_r2={val_metrics['r2']:.4f}, complexity={model.total_complexity():.2f}")
            
            if wandb is not None:
                log_dict = {
                    "evo/generation": gen,
                    "evo/train_fitness": fitness,
                    "evo/val_mse": val_metrics["mse"],
                    "evo/val_r2": val_metrics["r2"],
                    "evo/complexity": model.total_complexity(),
                }
                # Log per-parameter R²
                r2_per = val_metrics.get("r2_per_target", [])
                for i, r2_val in enumerate(r2_per):
                    pname = param_names[i] if param_names and i < len(param_names) else f"param_{i}"
                    log_dict[f"evo/val_r2_{pname}"] = float(r2_val)
                wandb.log(log_dict)
    
    model.programs = best_programs
    print(f"\nBest fitness: {best_fitness:.4f}")
    
    return history


def train_rl_search(
    model: SetDSR,
    searcher: RLProgramSearcher,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_generations: int = 100,
    n_samples_per_step: int = 10,
    complexity_weight: float = 0.01,
    curriculum_schedule: dict = None,
    log_interval: int = 10,
    input_norm: str = "none",
    input_stats: dict = None,
    output_norm: str = "none",
    output_stats: dict = None,
    wandb = None,
    param_names: List[str] = None,
):
    """Phase 1: RL-based search for symbolic programs."""
    print("\n" + "=" * 60)
    print("Phase 1: RL-based Program Search")
    print("=" * 60)
    
    # Collect all training data into tensors
    X_all, mask_all, y_all = [], [], []
    for batch in train_loader:
        X, mask, y = batch
        X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
        X_all.append(X)
        mask_all.append(mask)
        y_all.append(y)
    
    X_train = torch.cat(X_all, dim=0).to(device)
    mask_train = torch.cat(mask_all, dim=0).to(device)
    y_train = torch.cat(y_all, dim=0).to(device)
    
    # Same for validation
    X_val_all, mask_val_all, y_val_all = [], [], []
    for batch in val_loader:
        X, mask, y = batch
        X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
        X_val_all.append(X)
        mask_val_all.append(mask)
        y_val_all.append(y)
    
    X_val = torch.cat(X_val_all, dim=0).to(device)
    mask_val = torch.cat(mask_val_all, dim=0).to(device)
    y_val = torch.cat(y_val_all, dim=0).to(device)
    
    best_fitness = float("inf")
    best_programs = None
    history = []
    
    for gen in range(n_generations):
        # Curriculum update
        if curriculum_schedule and gen in curriculum_schedule:
            new_level = curriculum_schedule[gen]
            print(f"\n[Gen {gen}] Advancing curriculum to level {new_level}")
            model.set_curriculum_level(new_level)
            # Rebuild policy network vocabulary for new operators
            searcher.policy._build_vocabulary()
        
        # RL search step
        best_ind, fitness = searcher.search_step(
            X_train, mask_train, y_train, 
            n_samples=n_samples_per_step,
            complexity_weight=complexity_weight
        )
        
        # Evaluate on validation set
        model.programs = best_ind
        with torch.no_grad():
            val_pred = model(X_val, mask_val)
            val_metrics = compute_metrics(y_val, val_pred)
        
        history.append({
            "generation": gen,
            "train_fitness": fitness,
            "val_mse": val_metrics["mse"],
            "val_r2": val_metrics["r2"],
            "complexity": model.total_complexity(),
            "baseline": searcher.baseline,
        })
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_programs = [p for p in best_ind]
        
        if gen % log_interval == 0:
            print(f"[Gen {gen:4d}] fitness={fitness:.4f}, val_mse={val_metrics['mse']:.4e}, "
                  f"val_r2={val_metrics['r2']:.4f}, baseline={searcher.baseline:.4f}")
            
            if wandb is not None:
                log_dict = {
                    "rl/generation": gen,
                    "rl/train_fitness": fitness,
                    "rl/val_mse": val_metrics["mse"],
                    "rl/val_r2": val_metrics["r2"],
                    "rl/complexity": model.total_complexity(),
                    "rl/baseline": searcher.baseline,
                }
                # Log per-parameter R²
                r2_per = val_metrics.get("r2_per_target", [])
                for i, r2_val in enumerate(r2_per):
                    pname = param_names[i] if param_names and i < len(param_names) else f"param_{i}"
                    log_dict[f"rl/val_r2_{pname}"] = float(r2_val)
                wandb.log(log_dict)
    
    model.programs = best_programs
    print(f"\nBest fitness: {best_fitness:.4f}")
    
    return history


def train_mlp_head(
    model: SetDSR,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 100,
    lr: float = 1e-3,
    log_interval: int = 10,
    input_norm: str = "none",
    input_stats: dict = None,
    output_norm: str = "none",
    output_stats: dict = None,
    wandb = None,
    param_names: List[str] = None,
):
    """Phase 2: Fine-tune MLP head with fixed symbolic programs."""
    print("\n" + "=" * 60)
    print("Phase 2: MLP Head Fine-tuning")
    print("=" * 60)
    
    # Precompute summaries for all data
    S_train_list, y_train_list = [], []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            X, mask, y = batch
            X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            S = model.compute_summaries(X, mask)
            S = torch.nan_to_num(S, nan=0.0, posinf=1e6, neginf=-1e6)
            S_train_list.append(S.cpu())
            y_train_list.append(y.cpu())
    
    S_train = torch.cat(S_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    
    S_val_list, y_val_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            X, mask, y = batch
            X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            S = model.compute_summaries(X, mask)
            S = torch.nan_to_num(S, nan=0.0, posinf=1e6, neginf=-1e6)
            S_val_list.append(S.cpu())
            y_val_list.append(y.cpu())
    
    S_val = torch.cat(S_val_list, dim=0).to(device)
    y_val = torch.cat(y_val_list, dim=0).to(device)
    
    # DataLoader for summaries
    summary_dataset = TensorDataset(S_train, y_train)
    summary_loader = DataLoader(summary_dataset, batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(model.mlp_head.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    history = []
    best_val_loss = float("inf")
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        
        for S_batch, y_batch in summary_loader:
            S_batch, y_batch = S_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model.mlp_head(S_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * S_batch.shape[0]
        
        train_loss /= len(summary_dataset)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model.mlp_head(S_val)
            val_loss = criterion(val_pred, y_val).item()
            val_metrics = compute_metrics(y_val, val_pred)
        
        scheduler.step(val_loss)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_r2": val_metrics["r2"],
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if epoch % log_interval == 0:
            print(f"[Epoch {epoch:4d}] train_loss={train_loss:.4e}, val_loss={val_loss:.4e}, "
                  f"val_r2={val_metrics['r2']:.4f}")
            
            if wandb is not None:
                log_dict = {
                    "mlp/epoch": epoch,
                    "mlp/train_loss": train_loss,
                    "mlp/val_loss": val_loss,
                    "mlp/val_r2": val_metrics["r2"],
                }
                # Log per-parameter R²
                r2_per = val_metrics.get("r2_per_target", [])
                for i, r2_val in enumerate(r2_per):
                    pname = param_names[i] if param_names and i < len(param_names) else f"param_{i}"
                    log_dict[f"mlp/val_r2_{pname}"] = float(r2_val)
                wandb.log(log_dict)
    
    print(f"\nBest validation loss: {best_val_loss:.4e}")
    return history


def train_parameterized(
    model: SetDSR,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 100,
    lr: float = 1e-3,
    log_interval: int = 10,
    input_norm: str = "none",
    input_stats: dict = None,
    output_norm: str = "none",
    output_stats: dict = None,
    wandb = None,
    param_names: List[str] = None,
):
    """
    Phase 2b: Train MLP head AND learnable constants in symbolic programs jointly.
    
    Converts the model to ParameterizedSetDSR and trains all constants
    via gradient descent alongside the MLP head.
    """
    print("\n" + "=" * 60)
    print("Phase 2b: Joint Training (MLP + Learnable Program Constants)")
    print("=" * 60)
    
    # Convert to parameterized model
    param_model = model.parameterize_constants()
    param_model.to(device)
    
    n_consts = param_model.num_learnable_constants()
    print(f"Number of learnable constants in programs: {n_consts}")
    
    # Optimizer for both MLP and program constants
    optimizer = optim.Adam(param_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    history = []
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(n_epochs):
        param_model.train()
        train_loss = 0.0
        n_samples = 0
        
        for batch in train_loader:
            X, mask, y = batch
            X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = param_model(X, mask)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X.shape[0]
            n_samples += X.shape[0]
        
        train_loss /= n_samples
        
        # Validation
        param_model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for batch in val_loader:
                X, mask, y = batch
                X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
                X, mask, y = X.to(device), mask.to(device), y.to(device)
                pred = param_model(X, mask)
                val_loss += criterion(pred, y).item() * X.shape[0]
                val_preds.append(pred)
                val_trues.append(y)
        
        val_loss /= sum(p.shape[0] for p in val_preds)
        val_pred = torch.cat(val_preds, dim=0)
        val_true = torch.cat(val_trues, dim=0)
        val_metrics = compute_metrics(val_true, val_pred)
        
        scheduler.step(val_loss)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_r2": val_metrics["r2"],
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in param_model.state_dict().items()}
        
        if epoch % log_interval == 0:
            print(f"[Epoch {epoch:4d}] train_loss={train_loss:.4e}, val_loss={val_loss:.4e}, "
                  f"val_r2={val_metrics['r2']:.4f}")
            
            if wandb is not None:
                log_dict = {
                    "param/epoch": epoch,
                    "param/train_loss": train_loss,
                    "param/val_loss": val_loss,
                    "param/val_r2": val_metrics["r2"],
                }
                # Log per-parameter R²
                r2_per = val_metrics.get("r2_per_target", [])
                for i, r2_val in enumerate(r2_per):
                    pname = param_names[i] if param_names and i < len(param_names) else f"param_{i}"
                    log_dict[f"param/val_r2_{pname}"] = float(r2_val)
                wandb.log(log_dict)
    
    # Restore best state
    if best_state:
        param_model.load_state_dict(best_state)
    
    # Update base model constants with learned values
    for i, info in enumerate(param_model.param_info):
        prog = model.programs[info["prog_idx"]]
        node = param_model._get_node_at_path(prog, info["path"])
        node.constant = param_model.program_params[i].item()
    
    # Copy trained MLP back to base model
    model.mlp_head.load_state_dict(param_model.mlp_head.state_dict())
    
    print(f"\nBest validation loss: {best_val_loss:.4e}")
    print(f"Learned {n_consts} program constants via backprop")
    
    return history, param_model


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Set-DSR model")
    
    # Data arguments (matching train.py)
    parser.add_argument("--h5-path", type=str, default="data/camels_LH.hdf5", help="Path to CAMELS LH HDF5 file")
    parser.add_argument("--snap", type=int, default=90, help="Snapshot number to read")
    parser.add_argument("--param-keys", nargs="+", default=None, help="List of param keys to use as targets")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction of data for training")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of data for validation")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of data for testing")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    
    # Normalization (matching train.py)
    parser.add_argument("--normalize-input", choices=["none", "log", "log_std"], default="log_std")
    parser.add_argument("--normalize-output", choices=["none", "minmax"], default="minmax")
    
    # Model arguments
    parser.add_argument("--n-summaries", type=int, default=8, help="Number of symbolic summaries (K)")
    parser.add_argument("--mlp-hidden", nargs="+", type=int, default=[64, 64], help="MLP hidden layer sizes")
    parser.add_argument("--max-depth", type=int, default=5, help="Max depth of expression trees")
    parser.add_argument("--operator-scope", type=str, default="full",
                        choices=["simple", "intermediate", "full"],
                        help="Operator scope: simple (arithmetic only), intermediate (+log/exp/trig), full (all ops)")
    
    # Training arguments
    parser.add_argument("--n-generations", type=int, default=100, help="Number of evolutionary generations")
    parser.add_argument("--mlp-epochs", type=int, default=100, help="Number of MLP fine-tuning epochs")
    parser.add_argument("--population-size", type=int, default=100, help="Population size for GP")
    parser.add_argument("--complexity-weight", type=float, default=0.01, help="Complexity penalty weight")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for MLP head")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N generations/epochs")
    
    # Curriculum schedule
    parser.add_argument("--curriculum-schedule", type=str, default="0:1,30:2,60:3",
                        help="Curriculum schedule as 'gen:level,gen:level,...'")
    
    # Output arguments
    parser.add_argument("--save-path", type=str, default="run/data/models/set_dsr/", help="Directory to save checkpoints")
    parser.add_argument("--plot-path", type=str, default="run/data/models/plots/set_dsr/", help="Directory to save plots")
    
    # WandB arguments (matching train.py)
    parser.add_argument("--wandb", action="store_true", help="Log runs to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="set-dsr", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb-save-model", action="store_true", help="Save model to WandB")
    
    # Advanced features (toggleable)
    parser.add_argument("--use-subtree-mutation", action="store_true", default=True,
                        help="Enable subtree mutation in GP evolver")
    parser.add_argument("--no-subtree-mutation", action="store_false", dest="use_subtree_mutation",
                        help="Disable subtree mutation")
    parser.add_argument("--use-constant-optimization", action="store_true", default=True,
                        help="Enable constant optimization in GP evolver")
    parser.add_argument("--no-constant-optimization", action="store_false", dest="use_constant_optimization",
                        help="Disable constant optimization")
    parser.add_argument("--const-opt-steps", type=int, default=10,
                        help="Number of constant optimization steps per generation")
    parser.add_argument("--const-opt-lr", type=float, default=0.1,
                        help="Learning rate for constant optimization")
    parser.add_argument("--mlp-retrain-interval", type=int, default=10,
                        help="Retrain MLP every N generations (0 to disable)")
    parser.add_argument("--mlp-retrain-epochs", type=int, default=20,
                        help="Epochs per MLP retrain during evolution")
    parser.add_argument("--use-rl-search", action="store_true", default=False,
                        help="Use RL-based search instead of GP evolution")
    parser.add_argument("--rl-samples", type=int, default=10,
                        help="Number of samples per RL search step")
    parser.add_argument("--rl-hidden-size", type=int, default=128,
                        help="Hidden size for RL policy network")
    parser.add_argument("--rl-lr", type=float, default=1e-3,
                        help="Learning rate for RL policy network")
    parser.add_argument("--use-sympy-simplify", action="store_true", default=True,
                        help="Apply SymPy simplification to final expressions")
    parser.add_argument("--no-sympy-simplify", action="store_false", dest="use_sympy_simplify",
                        help="Disable SymPy simplification")
    parser.add_argument("--use-learnable-constants", action="store_true", default=False,
                        help="Train program constants via gradient descent in Phase 2")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args(argv)
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Parse curriculum schedule
    curriculum_schedule = {}
    if args.curriculum_schedule:
        for pair in args.curriculum_schedule.split(","):
            gen, level = pair.split(":")
            curriculum_schedule[int(gen)] = int(level)
    
    # Create output directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.plot_path, exist_ok=True)
    
    # Load dataset (matching train.py)
    print(f"Loading data from {args.h5_path}...")
    
    # Probe dataset to get parameter names
    probe_ds = HDF5SetDataset(h5_path=args.h5_path, snap=args.snap, param_keys=None, data_field="SubhaloStellarMass")
    detected_param_names = getattr(probe_ds, 'param_names', None) or getattr(probe_ds, 'param_keys', None)
    
    # Map param keys
    if args.param_keys is not None and detected_param_names is not None:
        mapped_keys = []
        for p_or_idx in args.param_keys:
            try:
                idx = int(p_or_idx)
                if 0 <= idx < len(detected_param_names):
                    mapped_keys.append(detected_param_names[idx])
                else:
                    raise ValueError(f"Index {idx} out of bounds")
            except (ValueError, TypeError):
                mapped_keys.append(p_or_idx)
        args.param_keys = mapped_keys
    elif args.param_keys is None and detected_param_names is not None:
        args.param_keys = detected_param_names
    
    # Create dataset
    full_ds = HDF5SetDataset(h5_path=args.h5_path, snap=args.snap, param_keys=args.param_keys, data_field="SubhaloStellarMass")
    n_total = len(full_ds)
    
    # Split dataset
    train_size = int(args.train_frac * n_total)
    val_size = int(args.val_frac * n_total)
    test_size = n_total - train_size - val_size
    
    indices = list(range(n_total))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx) if test_size > 0 else None
    
    print(f"Split sizes -> train: {train_size}, val: {val_size}, test: {test_size}")
    
    # Get sample to determine dimensions
    sample_x, sample_y = full_ds[0]
    input_dim = 1 if sample_x.ndim == 1 else sample_x.shape[1]
    target_dim = 1 if np.ndim(sample_y) == 0 else sample_y.shape[0]
    
    print(f"Input dim: {input_dim}, Target dim: {target_dim}")
    print(f"Parameters: {args.param_keys}")
    
    # Compute normalization stats
    input_stats = None
    output_stats = None
    if args.normalize_input != "none" or args.normalize_output != "none":
        stats = compute_stats_from_dataset(train_ds)
        input_stats = {"input_mean": stats["input_mean"], "input_std": stats["input_std"]}
        output_stats = {"y_min": stats["y_min"], "y_max": stats["y_max"]}
        print(f"Input stats: mean={input_stats['input_mean']}, std={input_stats['input_std']}")
        print(f"Output stats: min={output_stats['y_min']}, max={output_stats['y_max']}")
    
    # Initialize wandb
    wandb = None
    if args.wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args)
            )
            if input_stats:
                wandb.config.update({"input_mean": input_stats["input_mean"].tolist(),
                                     "input_std": input_stats["input_std"].tolist()})
            if output_stats:
                wandb.config.update({"y_min": output_stats["y_min"].tolist(),
                                     "y_max": output_stats["y_max"].tolist()})
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
            wandb = None
    
    # Create data loaders
    collate_fn = functools.partial(hdf5_collate, max_size=full_ds.max_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn) if test_ds else None
    
    # Create feature names
    feature_names = ["logM"] if input_dim == 1 else [f"x_{i}" for i in range(input_dim)]
    
    # Create model
    model = SetDSR(
        n_features=input_dim,
        feature_names=feature_names,
        n_summaries=args.n_summaries,
        n_params=target_dim,
        mlp_hidden=args.mlp_hidden,
        curriculum_level=1,
        max_depth=args.max_depth,
        operator_scope=args.operator_scope,
    )
    model.to(device)
    
    print(f"\nOperator scope: {args.operator_scope}")
    print("\nInitial model:")
    print_model_summary(model)
    
    start_time = time.time()
    
    # Phase 1: Symbolic search
    if args.use_rl_search:
        # RL-based search
        print("\nUsing RL-based program search")
        policy = ProgramPolicyNetwork(
            model.grammar,
            hidden_size=args.rl_hidden_size,
            n_summaries=args.n_summaries,
        )
        policy.to(device)
        
        searcher = RLProgramSearcher(
            model=model,
            policy=policy,
            lr=args.rl_lr,
        )
        
        evo_history = train_rl_search(
            model, searcher,
            train_loader, val_loader, device,
            n_generations=args.n_generations,
            n_samples_per_step=args.rl_samples,
            complexity_weight=args.complexity_weight,
            curriculum_schedule=curriculum_schedule,
            log_interval=args.log_interval,
            input_norm=args.normalize_input,
            input_stats=input_stats,
            output_norm=args.normalize_output,
            output_stats=output_stats,
            wandb=wandb,
            param_names=args.param_keys,
        )
    else:
        # GP-based evolution
        use_advanced = args.use_subtree_mutation or args.use_constant_optimization
        if use_advanced:
            print(f"\nUsing advanced GP evolver (subtree_mutation={args.use_subtree_mutation}, "
                  f"const_opt={args.use_constant_optimization})")
            evolver = AdvancedSetDSREvolver(
                model,
                population_size=args.population_size,
                elite_size=10,
                mutation_rate=0.3,
                crossover_rate=0.5,
                use_subtree_mutation=args.use_subtree_mutation,
                use_constant_optimization=args.use_constant_optimization,
                const_opt_steps=args.const_opt_steps,
                const_opt_lr=args.const_opt_lr,
                mlp_retrain_interval=args.mlp_retrain_interval,
                mlp_retrain_epochs=args.mlp_retrain_epochs,
            )
        else:
            print("\nUsing basic GP evolver")
            evolver = SetDSREvolver(
                model,
                population_size=args.population_size,
                elite_size=10,
                mutation_rate=0.3,
                crossover_rate=0.5,
                mlp_retrain_interval=args.mlp_retrain_interval,
                mlp_retrain_epochs=args.mlp_retrain_epochs,
            )
        
        evo_history = train_evolutionary(
            model, evolver,
            train_loader, val_loader, device,
            n_generations=args.n_generations,
            complexity_weight=args.complexity_weight,
            curriculum_schedule=curriculum_schedule,
            log_interval=args.log_interval,
            input_norm=args.normalize_input,
            input_stats=input_stats,
            output_norm=args.normalize_output,
            output_stats=output_stats,
            wandb=wandb,
            param_names=args.param_keys,
        )
    
    # Phase 2: Fine-tune MLP head (and optionally program constants)
    param_model = None
    if args.use_learnable_constants:
        mlp_history, param_model = train_parameterized(
            model,
            train_loader, val_loader, device,
            n_epochs=args.mlp_epochs,
            lr=args.lr,
            log_interval=args.log_interval,
            input_norm=args.normalize_input,
            input_stats=input_stats,
            output_norm=args.normalize_output,
            output_stats=output_stats,
            wandb=wandb,
            param_names=args.param_keys,
        )
    else:
        mlp_history = train_mlp_head(
            model,
            train_loader, val_loader, device,
            n_epochs=args.mlp_epochs,
            lr=args.lr,
            log_interval=args.log_interval,
            input_norm=args.normalize_input,
            input_stats=input_stats,
            output_norm=args.normalize_output,
            output_stats=output_stats,
            wandb=wandb,
            param_names=args.param_keys,
        )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    
    # Final model summary
    print("\nFinal model:")
    print_model_summary(model)
    
    # Final evaluation
    model.eval()
    
    # Collect predictions for plotting
    train_y_true, train_y_pred = collect_predictions(
        model, train_loader, device,
        input_norm=args.normalize_input, input_stats=input_stats,
        output_norm=args.normalize_output, output_stats=output_stats
    )
    val_y_true, val_y_pred = collect_predictions(
        model, val_loader, device,
        input_norm=args.normalize_input, input_stats=input_stats,
        output_norm=args.normalize_output, output_stats=output_stats
    )
    
    # Compute final metrics
    train_metrics = compute_metrics(
        torch.from_numpy(train_y_true).to(device),
        torch.from_numpy(train_y_pred).to(device)
    )
    val_metrics = compute_metrics(
        torch.from_numpy(val_y_true).to(device),
        torch.from_numpy(val_y_pred).to(device)
    )
    
    print(f"\nFinal Train MSE: {train_metrics['mse']:.4e}, R2: {train_metrics['r2']:.4f}")
    print(f"Final Val MSE: {val_metrics['mse']:.4e}, R2: {val_metrics['r2']:.4f}")
    
    # Test evaluation
    if test_loader:
        test_y_true, test_y_pred = collect_predictions(
            model, test_loader, device,
            input_norm=args.normalize_input, input_stats=input_stats,
            output_norm=args.normalize_output, output_stats=output_stats
        )
        test_metrics = compute_metrics(
            torch.from_numpy(test_y_true).to(device),
            torch.from_numpy(test_y_pred).to(device)
        )
        print(f"Final Test MSE: {test_metrics['mse']:.4e}, R2: {test_metrics['r2']:.4f}")
        
        # Test plot
        test_plot_path = os.path.join(args.plot_path, "pred_vs_true_test_final.png")
        plot_pred_vs_true(test_y_true, test_y_pred, test_plot_path, 
                          title="Test: pred vs true", param_names=args.param_keys)
        print(f"Saved test plot: {test_plot_path}")
        
        if wandb:
            log_dict = {
                "test_mse": test_metrics["mse"],
                "test_r2": test_metrics["r2"],
                "pred_vs_true_test": wandb.Image(test_plot_path),
            }
            # Log per-parameter R² for test set
            r2_per = test_metrics.get("r2_per_target", [])
            for i, r2_val in enumerate(r2_per):
                pname = args.param_keys[i] if args.param_keys and i < len(args.param_keys) else f"param_{i}"
                log_dict[f"test_r2_{pname}"] = float(r2_val)
            wandb.log(log_dict)
    
    # Save plots
    train_plot_path = os.path.join(args.plot_path, "pred_vs_true_train_final.png")
    val_plot_path = os.path.join(args.plot_path, "pred_vs_true_val_final.png")
    plot_pred_vs_true(train_y_true, train_y_pred, train_plot_path, 
                      title="Train: pred vs true", param_names=args.param_keys)
    plot_pred_vs_true(val_y_true, val_y_pred, val_plot_path, 
                      title="Val: pred vs true", param_names=args.param_keys)
    print(f"Saved plots: {train_plot_path}, {val_plot_path}")
    
    if wandb:
        wandb.log({
            "pred_vs_true_train": wandb.Image(train_plot_path),
            "pred_vs_true_val": wandb.Image(val_plot_path),
        })
    
    # Save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    expressions = extract_expressions(model)
    
    # Optionally simplify expressions with SymPy
    simplified_expressions = None
    latex_expressions = None
    if args.use_sympy_simplify:
        try:
            print("\nApplying SymPy simplification...")
            simplified_expressions = simplify_all_expressions(model)
            latex_expressions = expressions_to_latex(model)
            print("Simplified expressions:")
            for name, expr in simplified_expressions.items():
                print(f"  {name}: {expr}")
        except Exception as e:
            print(f"Warning: SymPy simplification failed: {e}")
            simplified_expressions = None
    
    # Save expressions as JSON
    expr_path = os.path.join(args.save_path, f"expressions_{timestamp}.json")
    expr_data = {
        "raw": expressions,
    }
    if simplified_expressions:
        expr_data["simplified"] = simplified_expressions
    if latex_expressions:
        expr_data["latex"] = latex_expressions
    with open(expr_path, "w") as f:
        json.dump(expr_data, f, indent=2)
    print(f"Saved expressions: {expr_path}")
    
    # Save model checkpoint
    ckpt_path = os.path.join(args.save_path, f"set_dsr_{timestamp}.pt")
    torch.save({
        "mlp_state_dict": model.mlp_head.state_dict(),
        "expressions": expressions,
        "simplified_expressions": simplified_expressions,
        "latex_expressions": latex_expressions,
        "config": vars(args),
        "input_stats": input_stats,
        "output_stats": output_stats,
        "n_features": input_dim,
        "n_summaries": args.n_summaries,
        "n_params": target_dim,
        "param_keys": args.param_keys,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    
    # Save history
    history = {"evolutionary": evo_history, "mlp": mlp_history}
    history_path = os.path.join(args.save_path, f"history_{timestamp}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history: {history_path}")
    
    if wandb and args.wandb_save_model:
        artifact = wandb.Artifact(name=f"set-dsr-{wandb.run.id}", type="model")
        artifact.add_file(ckpt_path)
        artifact.add_file(expr_path)
        wandb.log_artifact(artifact)
        print("Uploaded checkpoint to WandB")
    
    # Print final expressions
    print("\n" + "=" * 60)
    print("Learned Symbolic Summary Statistics:")
    print("=" * 60)
    if simplified_expressions:
        print("(Simplified with SymPy)")
        for name, expr in simplified_expressions.items():
            print(f"  {name}: {expr}")
    else:
        for name, expr in expressions.items():
            print(f"  {name}: {expr}")
    print("=" * 60)
    
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()