import argparse
import math
import random
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

import numpy as np
import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from deepset import DeepSet, MLP, DeepSetMultiPhi
from setpooling import SlotSetPool
from data import HDF5SetDataset, hdf5_collate
import functools


def stellar_mass_function(masses, bins: int = 3, mass_range: Tuple[float, float] = (8.0, 11.0), box_size: float = 25 / 0.6711):
    m = np.asarray(masses, dtype=np.float64)
    m = np.where(m <= 0, 1e6, m)  # avoid log10(0)
    logm = np.log10(m)

    hist, edges = np.histogram(logm, bins=bins, range=mass_range)
    dM = edges[1] - edges[0]
    centers = 0.5 * (edges[1:] + edges[:-1])

    # number density per dex
    phi = hist / dM / (box_size ** 3)
    return phi, centers


def apply_subhalo_mask_np(
    masses: np.ndarray,
    prob: float,
    bias_low: float,
    bias_high: float,
    bias_strength: float,
    keep_one: bool,
) -> np.ndarray:
    if prob <= 0 or masses.size == 0:
        return masses
    if masses.ndim != 1:
        raise ValueError("SMF masking expects 1D masses per simulation")
    m = np.where(masses > 0, masses, 1e-6).astype(np.float64, copy=False)
    logm = np.log10(m)
    prob_arr = np.full_like(logm, float(prob))
    if bias_strength > 0 and bias_high > bias_low:
        scaled = (logm - bias_low) / (bias_high - bias_low)
        scaled = np.clip(scaled, 0.0, 1.0)
        prob_arr = prob_arr * (1.0 + bias_strength * (1.0 - scaled))
    prob_arr = np.clip(prob_arr, 0.0, 1.0)
    keep = np.random.rand(masses.shape[0]) >= prob_arr
    if keep_one and not keep.any() and masses.shape[0] > 0:
        keep[np.argmax(masses)] = True
    return masses[keep]


def smf_collate_from_masses(
    batch,
    bins: int,
    mass_range: Tuple[float, float],
    box_size: float,
    mask_prob: float = 0.0,
    mask_bias_low: float = 8.0,
    mask_bias_high: float = 11.0,
    mask_bias_strength: float = 0.0,
    mask_keep_one: bool = True,
):
    Xs = []
    ys = []
    for masses, y in batch:
        masses = np.asarray(masses, dtype=np.float64)
        if mask_prob > 0:
            masses = apply_subhalo_mask_np(
                masses,
                prob=mask_prob,
                bias_low=mask_bias_low,
                bias_high=mask_bias_high,
                bias_strength=mask_bias_strength,
                keep_one=mask_keep_one,
            )
        phi, _ = stellar_mass_function(
            masses,
            bins=bins,
            mass_range=mass_range,
            box_size=box_size,
        )
        Xs.append(phi.astype(np.float32))
        ys.append(np.array(y, dtype=np.float32))
    X = np.stack(Xs, axis=0)
    y_arr = np.stack(ys, axis=0)
    return torch.from_numpy(X), torch.from_numpy(y_arr)


def compute_and_save_smf(
    h5_path: str,
    snap: int,
    bins: int,
    mass_range: Tuple[float, float],
    box_size: float,
    overwrite: bool = True,
):
    with h5py.File(h5_path, "a") as f:
        g = f[f"snap_{snap:03d}"]
        mstar_vlen = g["SubhaloStellarMass"]
        nsims = mstar_vlen.shape[0]

        smf_grp = g.require_group("smf")
        if overwrite:
            for name in ("phi", "logM"):
                if name in smf_grp:
                    del smf_grp[name]

        phi_all = np.zeros((nsims, bins), dtype=np.float32)

        centers_ref = None
        for sim in range(nsims):
            phi, centers = stellar_mass_function(mstar_vlen[sim], bins=bins, mass_range=mass_range, box_size=box_size)
            phi_all[sim, :] = phi
            if centers_ref is None:
                centers_ref = centers

        smf_grp.create_dataset("phi", data=phi_all)
        if centers_ref is not None:
            smf_grp.create_dataset("logM", data=np.asarray(centers_ref, dtype=np.float32))


class SMFOnTheFlyDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        snap: int,
        param_keys: Optional[List[str]] = None,
        bins: int = 13,
        mass_range: Tuple[float, float] = (8.0, 12.0),
        box_size: float = 25 / 0.6711,
        data_field="SubhaloStellarMass",
        mass_min=None,
        mass_max=None,
        mass_feature_idx=None,
        mass_feature_name=None,
        feature_names=None,
    ):
        if isinstance(data_field, (list, tuple, np.ndarray)):
            if len(data_field) != 1:
                raise ValueError("SMF dataset expects a single data_field entry")
            data_field = data_field[0]
        self.data_field = data_field
        self.bins = int(bins)
        self.mass_range = (float(mass_range[0]), float(mass_range[1]))
        self.box_size = float(box_size)
        if self.bins <= 0:
            raise ValueError("bins must be positive")
        if self.mass_range[0] >= self.mass_range[1]:
            raise ValueError("mass_range must be (low, high)")
        if self.box_size <= 0:
            raise ValueError("box_size must be positive")

        self.base = HDF5SetDataset(
            h5_path=h5_path,
            snap=snap,
            param_keys=param_keys,
            data_field=data_field,
            mass_min=mass_min,
            mass_max=mass_max,
            mass_feature_idx=mass_feature_idx,
            mass_feature_name=mass_feature_name,
            feature_names=feature_names,
        )
        self.param_names = getattr(self.base, "param_names", None)
        self.param_keys = self.base.param_keys
        self.seeds = getattr(self.base, "seeds", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        masses, y = self.base[idx]
        return np.asarray(masses), y


def compute_stats_from_dataset(dataset, sample_limit: int = 1000):
    """Estimate input (log) mean/std and output min/max from `dataset`.

    Returns a dict with keys: 'input_mean','input_std','y_min','y_max'.
    """
    # Determine length
    n = len(dataset)
    if n == 0:
        raise ValueError("Empty dataset for stats computation")

    #step = max(1, n // sample_limit) For now, I would use the entire dataset to normalize

    step = 1

    sum_x = None
    sum_x2 = None
    count = 0
    y_mins = []
    y_maxs = []

    for i in range(0, n, step):
        x, y = dataset[i]
        # x: (Ni, D)
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

    # Ensure y_min/y_max are at least 1-D arrays so downstream code
    # can safely call `torch.from_numpy`.
    y_min = np.atleast_1d(y_min)
    y_max = np.atleast_1d(y_max)

    return {"input_mean": mean, "input_std": std, "y_min": y_min, "y_max": y_max}


def compute_stats_from_loader(loader):
    sum_x = None
    sum_x2 = None
    count = 0
    y_mins = []
    y_maxs = []

    for batch in loader:
        if len(batch) == 2:
            X, y = batch
        else:
            X, _, y = batch
        X_np = np.array(X, dtype=np.float64)
        lx = np.log1p(X_np)
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


def normalize_batch(X, mask, y, input_norm: str, input_stats: dict, output_norm: str, output_stats: dict, eps: float = 1e-8):
    """
    Apply normalizations to a batch of tensors (on CPU or device).
    X: tensor (B,N,D), mask: (B,N), y: (B,) or (B,K)
    input_norm: 'none'|'log'|'log_std'
    output_norm: 'none'|'minmax'
    input_stats/output_stats: dicts as returned by `compute_stats_from_dataset` (numpy arrays)
    """
    if input_norm is None or input_norm == "none":
        pass
    else:
        if input_norm in ("log", "log_std"):
            X = torch.log1p(X)
        if input_norm == "log_std":
            # subtract mean and divide by std
            if input_stats is None:
                raise ValueError("input_stats required for 'log_std' normalization")
            # ensure stats are numpy arrays (handle scalar -> 1-D)
            mean_np = np.atleast_1d(input_stats["input_mean"])
            std_np = np.atleast_1d(input_stats["input_std"])
            mean = torch.from_numpy(mean_np).to(X.dtype).to(X.device)
            std = torch.from_numpy(std_np).to(X.dtype).to(X.device)
            # broadcast mean/std according to X dimensions
            if X.ndim == 3:
                X = (X - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + eps)
            elif X.ndim == 2:
                X = (X - mean.view(1, -1)) / (std.view(1, -1) + eps)
            elif X.ndim == 1:
                X = (X - mean) / (std + eps)
            else:
                X = (X - mean) / (std + eps)

    if output_norm is None or output_norm == "none":
        pass
    else:
        if output_norm == "minmax":
            y_min = torch.from_numpy(output_stats["y_min"]).to(y.dtype).to(y.device)
            y_max = torch.from_numpy(output_stats["y_max"]).to(y.dtype).to(y.device)
            # handle scalar target
            if y.dim() == 1:
                y = (y - y_min) / (y_max - y_min + eps)
            else:
                y = (y - y_min.view(1, -1)) / (y_max.view(1, -1) - y_min.view(1, -1) + eps)

    return X, mask, y


def resolve_feature_index(feature_names, feature_idx, feature_name, fallback_name="SubhaloStellarMass"):
    if feature_idx is not None and feature_name is not None:
        raise ValueError("Specify only one of mask_feature_idx or mask_feature_name")
    if feature_idx is not None:
        return int(feature_idx)
    if feature_name is not None:
        if feature_names is None:
            raise ValueError("mask_feature_name requires feature_names to be available")
        if feature_name not in feature_names:
            raise ValueError(f"Unknown mask_feature_name '{feature_name}'; available: {feature_names}")
        return feature_names.index(feature_name)
    if feature_names:
        if fallback_name in feature_names:
            return feature_names.index(fallback_name)
        return 0
    return 0


def apply_subhalo_mask(
    X,
    mask,
    prob: float,
    feature_idx: int,
    bias_low: float,
    bias_high: float,
    bias_strength: float,
    keep_one: bool,
):
    if mask is None or prob is None or prob <= 0:
        return X, mask
    if X.ndim == 2:
        mass = X
    else:
        if feature_idx < 0 or feature_idx >= X.shape[2]:
            raise ValueError(f"mask_feature_idx {feature_idx} out of bounds for data with {X.shape[2]} features")
        mass = X[:, :, feature_idx]
    valid = mask > 0
    prob_t = torch.full_like(mass, float(prob))
    if bias_strength > 0 and bias_high > bias_low:
        logm = torch.log10(torch.clamp(mass, min=1e-12))
        scaled = (logm - bias_low) / (bias_high - bias_low)
        scaled = torch.clamp(scaled, 0.0, 1.0)
        prob_t = prob_t * (1.0 + bias_strength * (1.0 - scaled))
    prob_t = torch.clamp(prob_t, 0.0, 1.0)
    prob_t = prob_t * valid.to(prob_t.dtype)
    drop = torch.rand_like(prob_t) < prob_t
    new_mask = mask * (~drop).to(mask.dtype)
    if keep_one:
        valid_counts = valid.sum(dim=1)
        new_counts = new_mask.sum(dim=1)
        need_fix = (valid_counts > 0) & (new_counts == 0)
        if need_fix.any():
            mass_safe = mass.clone()
            mass_safe[~valid] = -float("inf")
            keep_idx = mass_safe.argmax(dim=1)
            batch_idx = torch.nonzero(need_fix, as_tuple=False).squeeze(1)
            new_mask[batch_idx, keep_idx[batch_idx]] = 1.0
    if X.ndim == 2:
        X = X * new_mask
    else:
        X = X * new_mask.unsqueeze(-1)
    return X, new_mask


def _inverse_output_norm(tensor: torch.Tensor, output_norm: str, output_stats: dict, device: torch.device, eps: float = 1e-8):
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


def run_model(model, X, mask=None):
    """Call model with (X,mask) if provided and return only the primary prediction tensor.

    This unwraps models like `SlotSetPool` that return `(y, attn)`.
    """
    out = model(X) if mask is None else model(X, mask)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def collect_predictions(model, loader, device,
                        input_norm: str = "none", input_stats: dict = None,
                        output_norm: str = "none", output_stats: dict = None,
                        predict_fn=None):
    """Run model on `loader` and return (y_true_np, y_pred_np) in original output units."""
    model.eval()
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                X, mask, y = batch
            else:
                X, y = batch
                mask = None

            X = X.to(device)
            if mask is not None:
                mask = mask.to(device)
            y = y.to(device)

            # keep a CPU copy of true y in original units
            y_true_np = y.cpu().numpy()
            y_true_np = np.squeeze(y_true_np)
            if y_true_np.ndim == 1:
                y_true_np = y_true_np.reshape(-1, 1)

            Xn, maskn, yn = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)
            if predict_fn is not None:
                pred_n = predict_fn(model, Xn, maskn)
            else:
                pred_n = run_model(model, Xn, maskn)

            # inverse transform predictions to original output units
            pred_orig = _inverse_output_norm(pred_n, output_norm, output_stats, device)

            pred_np = pred_orig.cpu().numpy()
            pred_np = np.squeeze(pred_np)
            if pred_np.ndim == 1:
                pred_np = pred_np.reshape(-1, 1)
            
            y_trues.append(y_true_np)
            y_preds.append(pred_np)

    y_trues = np.concatenate(y_trues, axis=0)
    y_preds = np.concatenate(y_preds, axis=0)
    return y_trues, y_preds


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str = "pred vs true"):
    # Convert inputs to 2D arrays (N, K)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # If shapes mismatch but are compatible (e.g. one is (N,) and other is (N,1)), try to reshape
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred have different number of samples: {y_true.shape[0]} vs {y_pred.shape[0]}")

    # Ensure same number of targets
    if y_true.shape[1] != y_pred.shape[1]:
        # try to broadcast single-target to multi-target if possible
        if y_true.shape[1] == 1 and y_pred.shape[1] > 1:
            y_true = np.repeat(y_true, y_pred.shape[1], axis=1)
        elif y_pred.shape[1] == 1 and y_true.shape[1] > 1:
            y_pred = np.repeat(y_pred, y_true.shape[1], axis=1)
        else:
            raise ValueError(f"y_true and y_pred have different target dims: {y_true.shape[1]} vs {y_pred.shape[1]}")

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

    # multiple targets: grid of subplots
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
        ax.set_title(f'{title} (target {k})')

    # hide unused axes
    for j in range(K, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def train_epoch(model, loader, opt, loss_fn, device,
                input_norm: str = "none", input_stats: dict = None,
                output_norm: str = "none", output_stats: dict = None, eps: float = 1e-8,
                predict_fn=None, mask_cfg: dict = None):
    """Train for one epoch and compute metrics.

    Returns (avg_loss, metrics_dict) where metrics_dict contains keys
    'mse', 'rel_err', 'r2'. All metrics are computed per-output-element.
    """
    model.train()
    total_loss = 0.0

    ss_res = 0.0
    abs_rel_sum = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    total_elements = 0
    # per-target accumulators (initialized on first batch)
    ss_res_per = None
    sum_y_per = None
    sum_y2_per = None

    for batch in loader:
        # batch may be (X, mask, y) for DeepSet, or (X, y) for SMF/MLP
        if len(batch) == 3:
            X, mask, y = batch
        else:
            X, y = batch
            mask = None

        X = X.to(device)
        if mask is not None:
            mask = mask.to(device)
        y = y.to(device)

        if mask is not None and mask_cfg is not None:
            X, mask = apply_subhalo_mask(
                X,
                mask,
                prob=mask_cfg["prob"],
                feature_idx=mask_cfg["feature_idx"],
                bias_low=mask_cfg["bias_low"],
                bias_high=mask_cfg["bias_high"],
                bias_strength=mask_cfg["bias_strength"],
                keep_one=mask_cfg["keep_one"],
            )

        X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)

        opt.zero_grad()
        if predict_fn is not None:
            pred = predict_fn(model, X, mask)
        else:
            pred = run_model(model, X, mask)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

        B = y.shape[0]
        # Ensure shapes (B, K)
        pred_flat = pred.view(B, -1)
        y_flat = y.view(B, -1)

        se = (pred_flat - y_flat) ** 2
        ss_res += float(se.sum().item())
        abs_rel_sum += float((torch.abs((pred_flat - y_flat) / (y_flat + eps))).sum().item())
        sum_y += float(y_flat.sum().item())
        sum_y2 += float((y_flat ** 2).sum().item())
        total_elements += int(y_flat.numel())

        # per-target sums
        K = pred_flat.shape[1]
        if ss_res_per is None:
            ss_res_per = torch.zeros(K, dtype=se.dtype, device=se.device)
            sum_y_per = torch.zeros(K, dtype=y_flat.dtype, device=y_flat.device)
            sum_y2_per = torch.zeros(K, dtype=y_flat.dtype, device=y_flat.device)
        # sum over batch dimension to get per-target sums
        ss_res_per += se.sum(dim=0)
        sum_y_per += y_flat.sum(dim=0)
        sum_y2_per += (y_flat ** 2).sum(dim=0)

        total_loss += float(loss.item()) * B

    avg_loss = total_loss / len(loader.dataset)
    mse = ss_res / (total_elements + eps)
    rel_err = abs_rel_sum / (total_elements + eps)

    # Compute per-target metrics
    if ss_res_per is None:
        # No data
        mse_per = np.array([])
        r2_per = np.array([])
    else:
        n_samples = total_elements // ss_res_per.numel()
        mse_per = (ss_res_per.detach().cpu().numpy() / (n_samples + eps)).astype(float)
        mean_y_per = (sum_y_per.detach().cpu().numpy() / (n_samples + eps)).astype(float)
        ss_tot_per = (sum_y2_per.detach().cpu().numpy() - n_samples * (mean_y_per ** 2)).astype(float)
        r2_per = (1.0 - (ss_res_per.detach().cpu().numpy() / (ss_tot_per + eps))).astype(float)

    mean_y = sum_y / (total_elements + eps)
    ss_tot = sum_y2 - total_elements * (mean_y ** 2)
    r2 = 1.0 - ss_res / (ss_tot + eps)

    metrics = {"mse": mse, "rel_err": rel_err, "r2": r2,
               "mse_per_target": mse_per, "r2_per_target": r2_per}
    return avg_loss, metrics


def eval_model(model, loader, loss_fn, device,
               input_norm: str = "none", input_stats: dict = None,
               output_norm: str = "none", output_stats: dict = None, eps: float = 1e-8,
               predict_fn=None):
    model.eval()
    total_loss = 0.0

    ss_res = 0.0
    abs_rel_sum = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    total_elements = 0
    # per-target accumulators
    ss_res_per = None
    sum_y_per = None
    sum_y2_per = None

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                X, mask, y = batch
            else:
                X, y = batch
                mask = None

            X = X.to(device)
            if mask is not None:
                mask = mask.to(device)
            y = y.to(device)

            X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm, output_stats)

            if predict_fn is not None:
                pred = predict_fn(model, X, mask)
            else:
                pred = run_model(model, X, mask)
            loss = loss_fn(pred, y)

            B = y.shape[0]
            pred_flat = pred.view(B, -1)
            y_flat = y.view(B, -1)

            se = (pred_flat - y_flat) ** 2
            ss_res += float(se.sum().item())
            abs_rel_sum += float((torch.abs((pred_flat - y_flat) / (y_flat + eps))).sum().item())
            sum_y += float(y_flat.sum().item())
            sum_y2 += float((y_flat ** 2).sum().item())
            total_elements += int(y_flat.numel())

            # per-target sums
            K = pred_flat.shape[1]
            if ss_res_per is None:
                ss_res_per = torch.zeros(K, dtype=se.dtype, device=se.device)
                sum_y_per = torch.zeros(K, dtype=y_flat.dtype, device=y_flat.device)
                sum_y2_per = torch.zeros(K, dtype=y_flat.dtype, device=y_flat.device)
            ss_res_per += se.sum(dim=0)
            sum_y_per += y_flat.sum(dim=0)
            sum_y2_per += (y_flat ** 2).sum(dim=0)

            total_loss += float(loss.item()) * B

    avg_loss = total_loss / len(loader.dataset)
    mse = ss_res / (total_elements + eps)
    rel_err = abs_rel_sum / (total_elements + eps)

    # per-target metrics
    if ss_res_per is None:
        mse_per = np.array([])
        r2_per = np.array([])
    else:
        n_samples = total_elements // ss_res_per.numel()
        mse_per = (ss_res_per.detach().cpu().numpy() / (n_samples + eps)).astype(float)
        mean_y_per = (sum_y_per.detach().cpu().numpy() / (n_samples + eps)).astype(float)
        ss_tot_per = (sum_y2_per.detach().cpu().numpy() - n_samples * (mean_y_per ** 2)).astype(float)
        r2_per = (1.0 - (ss_res_per.detach().cpu().numpy() / (ss_tot_per + eps))).astype(float)

    mean_y = sum_y / (total_elements + eps)
    ss_tot = sum_y2 - total_elements * (mean_y ** 2)
    r2 = 1.0 - ss_res / (ss_tot + eps)

    metrics = {"mse": mse, "rel_err": rel_err, "r2": r2,
               "mse_per_target": mse_per, "r2_per_target": r2_per}
    return avg_loss, metrics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 for single-process)")
    parser.add_argument("--pin-memory", action="store_true",
                        help="Pin CPU memory for faster H2D transfers")
    parser.add_argument("--persistent-workers", action="store_true",
                        help="Keep DataLoader workers alive between epochs")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Batches prefetched per worker (requires num_workers > 0)")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction of data for training")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of data for validation")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of data for held-out testing")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-hdf5", action="store_true", help="Use CAMELS HDF5 dataset instead of synthetic data")
    parser.add_argument("--h5-path", type=str, default="data/camels_LH.hdf5", help="Path to CAMELS LH HDF5 file")
    parser.add_argument("--snap", type=int, default=90, help="Snapshot number to read (e.g. 90)")
    parser.add_argument("--data-field", nargs="+", default=None, help="HDF5 data field(s) under the snapshot group")
    parser.add_argument("--param-keys", nargs="+", default=None, help="List of params keys to use as targets")
    parser.add_argument("--mass-min", type=float, nargs="+", default=None, help="Lower mass cut(s) (inclusive); can provide multiple values")
    parser.add_argument("--mass-max", type=float, nargs="+", default=None, help="Upper mass cut(s) (inclusive); can provide multiple values")
    parser.add_argument("--mass-feature-idx", type=int, nargs="+", default=None, help="Feature index/indices to apply mass cuts when inputs are multi-dimensional")
    parser.add_argument("--mass-feature-name", type=str, nargs="+", default=None, help="Feature name(s) to apply mass cuts (uses feature-names or data-field names)")
    parser.add_argument("--feature-names", type=str, nargs="+", default=None, help="Optional list of input feature names for mapping mass-feature-name to indices")
    parser.add_argument("--mask-prob", type=float, default=0.0, help="Random subhalo drop probability (0 disables)")
    parser.add_argument("--mask-feature-idx", type=int, default=None, help="Feature index for masking bias model")
    parser.add_argument("--mask-feature-name", type=str, default=None, help="Feature name for masking bias model")
    parser.add_argument("--mask-bias-low", type=float, default=8.0, help="Low log10 mass for higher masking probability")
    parser.add_argument("--mask-bias-high", type=float, default=11.0, help="High log10 mass for lower masking probability")
    parser.add_argument("--mask-bias-strength", type=float, default=0.0, help="Strength of low-mass bias for masking")
    parser.add_argument("--mask-allow-empty", action="store_true", help="Allow all subhalos to be masked for a simulation")
    parser.add_argument("--normalize-input", choices=["none","log","log_std"], default="none", help="Input normalization: log, or log+standardize")
    parser.add_argument("--normalize-output", choices=["none","minmax"], default="none", help="Output normalization: minmax scaling")
    parser.add_argument("--stats-sample", type=int, default=1000, help="Number of samples to use when estimating normalization stats")
    parser.add_argument("--wandb", action="store_true", help="Log runs to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="deepset-reg", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity (team/user)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb-save-model", action="store_true", help="Save model checkpoint to WandB at end of training")
    parser.add_argument("--save-model", action="store_true", help="Save model checkpoint locally at end of training")
    parser.add_argument("--save-path", type=str, default="/u/gkerex/projects/IdealSummary/data/models/checkpoints/", help="Local directory to save model checkpoints")
    parser.add_argument("--plot-path", type=str, default="/u/gkerex/projects/IdealSummary/data/plots/", help="Directory to save prediction plots (separate from checkpoints)")
    parser.add_argument("--plot-every", type=int, default=1, help="Save pred-vs-true plots every N epochs (0 disables)")
    parser.add_argument("--use-smf", action="store_true", help="Compute SMF from SubhaloStellarMass and use fixed-size inputs")
    parser.add_argument("--smf-bins", type=int, default=13, help="Number of SMF bins for --use-smf")
    parser.add_argument("--smf-mass-range", type=float, nargs=2, default=[8.0, 12.0], help="Log10 mass range (min max) for SMF bins")
    parser.add_argument("--smf-box-size", type=float, default=25/0.6711, help="Box size for SMF number density")
    parser.add_argument("--multi-phi", action="store_true", help="Use DeepSetMultiPhi with multiple phi networks (set-based inputs)")
    parser.add_argument("--model-type", type=str, choices=["deepset", "mlp", "slotsetpool"], default="deepset",
                        help="Model type to construct: deepset (set-based), mlp (fixed-vector), or slotsetpool")
    parser.add_argument("--mlp-structure", choices=["shallow", "intermediate", "deep"], default=None,
                        help="Preset MLP widths for fixed-vector inputs; overridden by --mlp-hidden")
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=None,
                        help="Hidden sizes for MLP (fixed-vector inputs), e.g. --mlp-hidden 256 512 256")
    parser.add_argument("--slot-K", type=int, default=32, help="Number of learnable slots for SlotSetPool")
    parser.add_argument("--slot-H", type=int, default=128, help="Per-slot hidden dim (H) for SlotSetPool")
    parser.add_argument("--slot-dropout", type=float, default=0.0, help="Dropout for SlotSetPool MLPs")
    parser.add_argument("--slot-logm-idx", type=int, default=0, help="Index of logM feature in input for SlotSetPool key augmentation")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.use_hdf5:
        raise ValueError("This training script requires HDF5 inputs. Run with --use-hdf5 (and optionally --use-smf).")

    # optional wandb init (deferred until after stats computed)
    wandb = None

    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    else:
        if args.persistent_workers:
            print("Warning: --persistent-workers requires --num-workers > 0; disabling.")
        if args.prefetch_factor != 2:
            print("Warning: --prefetch-factor requires --num-workers > 0; ignoring.")
        loader_kwargs["persistent_workers"] = False

    # Determine target_dim from param_keys (applies to both SMF and set modes)
    if args.param_keys is not None:
        target_dim = len(args.param_keys)
    else:
        target_dim = None
    
    mask_cfg = None
    if args.use_hdf5:
        if args.use_smf:
            if args.data_field is None:
                data_fields = ["SubhaloStellarMass"]
            else:
                data_fields = args.data_field
                if len(data_fields) != 1:
                    raise ValueError("--use-smf expects a single --data-field entry")
                if data_fields[0] != "SubhaloStellarMass":
                    raise ValueError("--use-smf computes SMF from SubhaloStellarMass; set --data-field SubhaloStellarMass")
        else:
            data_fields = args.data_field if args.data_field is not None else ["SubhaloStellarMass"]

        data_field = data_fields[0] if len(data_fields) == 1 else data_fields

        smf_mass_range = tuple(args.smf_mass_range)

        # Probe dataset first to get all available parameter names
        if args.use_smf:
            probe_ds = SMFOnTheFlyDataset(
                h5_path=args.h5_path,
                snap=args.snap,
                param_keys=None,
                bins=args.smf_bins,
                mass_range=smf_mass_range,
                box_size=args.smf_box_size,
                data_field=data_field,
                mass_min=args.mass_min,
                mass_max=args.mass_max,
                mass_feature_idx=args.mass_feature_idx,
                mass_feature_name=args.mass_feature_name,
                feature_names=args.feature_names,
            )
        else:
            probe_ds = HDF5SetDataset(
                h5_path=args.h5_path,
                snap=args.snap,
                param_keys=None,
                data_field=data_field,
                mass_min=args.mass_min,
                mass_max=args.mass_max,
                mass_feature_idx=args.mass_feature_idx,
                mass_feature_name=args.mass_feature_name,
                feature_names=args.feature_names,
            )
        detected_param_names = getattr(probe_ds, 'param_names', None) or getattr(probe_ds, 'param_keys', None)

        # If user passed integer indices, map them to names. If no keys, default to all.
        if args.param_keys is not None and detected_param_names is not None:
            mapped_keys = []
            for p_or_idx in args.param_keys:
                try:
                    idx = int(p_or_idx)
                    if 0 <= idx < len(detected_param_names):
                        mapped_keys.append(detected_param_names[idx])
                    else:
                        raise ValueError(f"Index {idx} is out of bounds for detected parameters.")
                except (ValueError, TypeError):
                    # Not an integer, assume it's a name
                    mapped_keys.append(p_or_idx)
            args.param_keys = mapped_keys
        elif args.param_keys is None and detected_param_names is not None:
            # Default to all available parameters
            args.param_keys = detected_param_names

        # Now, create the definitive dataset with the resolved parameter keys
        if args.use_smf:
            full_ds = SMFOnTheFlyDataset(
                h5_path=args.h5_path,
                snap=args.snap,
                param_keys=args.param_keys,
                bins=args.smf_bins,
                mass_range=smf_mass_range,
                box_size=args.smf_box_size,
                data_field=data_field,
                mass_min=args.mass_min,
                mass_max=args.mass_max,
                mass_feature_idx=args.mass_feature_idx,
                mass_feature_name=args.mass_feature_name,
                feature_names=args.feature_names,
            )
        else:
            full_ds = HDF5SetDataset(
                h5_path=args.h5_path,
                snap=args.snap,
                param_keys=args.param_keys,
                data_field=data_field,
                mass_min=args.mass_min,
                mass_max=args.mass_max,
                mass_feature_idx=args.mass_feature_idx,
                mass_feature_name=args.mass_feature_name,
                feature_names=args.feature_names,
            )
        n_total = len(full_ds)
        if args.mask_prob < 0 or args.mask_prob > 1:
            raise ValueError("--mask-prob must be between 0 and 1")
        if args.mask_bias_strength < 0:
            raise ValueError("--mask-bias-strength must be >= 0")
        if args.mask_prob > 0:
            if args.mask_bias_high <= args.mask_bias_low:
                raise ValueError("--mask-bias-high must be > --mask-bias-low")
            if not args.use_smf:
                feature_names = getattr(full_ds, "feature_names", None)
                if feature_names is None and isinstance(data_field, (list, tuple, np.ndarray)):
                    feature_names = list(data_field)
                mask_feature_idx = resolve_feature_index(
                    feature_names,
                    args.mask_feature_idx,
                    args.mask_feature_name,
                )
            else:
                mask_feature_idx = 0
            mask_cfg = {
                "prob": float(args.mask_prob),
                "feature_idx": int(mask_feature_idx),
                "bias_low": float(args.mask_bias_low),
                "bias_high": float(args.mask_bias_high),
                "bias_strength": float(args.mask_bias_strength),
                "keep_one": not args.mask_allow_empty,
            }

        total_frac = args.train_frac + args.val_frac + args.test_frac
        if total_frac <= 0:
            raise ValueError("train/val/test fractions must sum to a positive value")
        if total_frac > 1.0 + 1e-6:
            raise ValueError("train/val/test fractions must sum to <= 1")

        train_size = int(round(args.train_frac * n_total))
        val_size = int(round(args.val_frac * n_total))
        test_size = int(round(args.test_frac * n_total))

        # Ensure we don't exceed available samples due to rounding
        total_assigned = train_size + val_size + test_size
        if total_assigned == 0:
            raise ValueError("Computed zero samples for all splits; adjust fractions")
        if total_assigned != n_total:
            # Assign any remainder to the training set (positive or negative)
            train_size = max(1, train_size + (n_total - total_assigned))
            total_assigned = train_size + val_size + test_size
        if total_assigned > n_total:
            raise ValueError("Split sizes exceed dataset size after rounding; adjust fractions")

        # Guarantee at least one sample for val/test if a non-zero fraction was requested and data permits
        if args.val_frac > 0 and val_size == 0 and n_total >= 2:
            val_size = 1
            train_size = max(1, train_size - 1)
        if args.test_frac > 0 and test_size == 0 and n_total - train_size - val_size >= 1:
            test_size = 1
            train_size = max(1, train_size - 1)

        # Final adjustment to ensure the splits sum exactly to n_total
        total_assigned = train_size + val_size + test_size
        if total_assigned < n_total:
            train_size += n_total - total_assigned
        elif total_assigned > n_total:
            # Trim from training first
            overflow = total_assigned - n_total
            trim_train = min(overflow, train_size - 1)
            train_size -= trim_train
            overflow -= trim_train
            val_size = max(0, val_size - overflow)

        if train_size <= 0:
            raise ValueError("Training split is empty; increase train_frac")
        if val_size <= 0:
            raise ValueError("Validation split is empty; increase val_frac")
        
        indices = list(range(n_total))
        train_idx = indices[: train_size]
        val_idx = indices[train_size : train_size + val_size]
        test_idx = indices[train_size + val_size : train_size + val_size + test_size]
        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
        test_ds = Subset(full_ds, test_idx) if test_size > 0 else None
        stats_ds = train_ds
        print(f"Split sizes -> train: {train_size}, val: {val_size}, test: {test_size}")
        
        # Get sample to determine dimensions
        sample_x, sample_y = full_ds[0]
        
        # Set target_dim if not already set
        if target_dim is None:
            target_dim = 1 if np.ndim(sample_y) == 0 else int(np.shape(sample_y)[0])
        
        # Choose collate function based on --use-smf flag.
        # The model is constructed after creating the DataLoader by
        # sampling a single batch to reliably infer input dimensions.
        if args.use_smf:
            collate_fn = functools.partial(
                smf_collate_from_masses,
                bins=args.smf_bins,
                mass_range=smf_mass_range,
                box_size=args.smf_box_size,
                mask_prob=args.mask_prob,
                mask_bias_low=args.mask_bias_low,
                mask_bias_high=args.mask_bias_high,
                mask_bias_strength=args.mask_bias_strength,
                mask_keep_one=not args.mask_allow_empty,
            )
        else:
            # use a fixed max_size (dataset.max_size) for padding/truncation
            try:
                collate_fn = functools.partial(hdf5_collate, max_size=full_ds.max_size)
            except Exception:
                collate_fn = hdf5_collate
        print(f"Dataset: {'SMF' if args.use_smf else 'Set-based'}, inferred target_dim: {target_dim}")

    # Estimate normalization stats from the training dataset if requested
    input_stats = None
    output_stats = None
    if args.normalize_input != "none" or args.normalize_output != "none":
        if args.use_smf:
            stats_collate = functools.partial(
                smf_collate_from_masses,
                bins=args.smf_bins,
                mass_range=smf_mass_range,
                box_size=args.smf_box_size,
                mask_prob=0.0,
                mask_bias_low=args.mask_bias_low,
                mask_bias_high=args.mask_bias_high,
                mask_bias_strength=args.mask_bias_strength,
                mask_keep_one=not args.mask_allow_empty,
            )
            stats_loader = DataLoader(
                stats_ds,
                batch_size=args.batch,
                shuffle=False,
                collate_fn=stats_collate,
                **loader_kwargs,
            )
            stats = compute_stats_from_loader(stats_loader)
        else:
            stats = compute_stats_from_dataset(stats_ds, sample_limit=args.stats_sample)
        input_stats = {"input_mean": stats["input_mean"], "input_std": stats["input_std"]}
        output_stats = {"y_min": stats["y_min"], "y_max": stats["y_max"]}

    # Initialize wandb if requested
    if args.wandb:
        try:
            import wandb as _wandb
        except Exception as e:
            raise RuntimeError("wandb requested but not installed. Install with `pip install wandb`." ) from e
        wandb = _wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))
        # log normalization stats (small arrays are OK)
        if input_stats is not None:
            wandb.config.update({"input_mean": input_stats["input_mean"].tolist(), "input_std": input_stats["input_std"].tolist()})
        if output_stats is not None:
            wandb.config.update({"y_min": output_stats["y_min"].tolist(), "y_max": output_stats["y_max"].tolist()})

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, **loader_kwargs) if test_ds is not None else None

    # Build model after inspecting a single batch from the training loader
    sample_batch = next(iter(train_loader))
    if len(sample_batch) == 3:
        Xb, maskb, yb = sample_batch
        use_mask = True
    else:
        Xb, yb = sample_batch
        maskb = None
        use_mask = False

    # Infer input and target dims from the sample batch
    if use_mask:
        # Xb: (B, N, D)
        print("Xb shape:", Xb.shape)
        print("mask shape:", maskb.shape)
        input_dim = int(Xb.shape[2])
    else:
        # Xb: (B, D)
        input_dim = int(Xb.shape[1])

    if target_dim is None:
        target_dim = 1 if yb.ndim == 1 else int(yb.shape[1])

    # Resolve MLP hidden sizes for fixed-vector inputs (e.g., SMF).
    if args.mlp_hidden is not None:
        mlp_hidden = [int(h) for h in args.mlp_hidden]
    elif args.mlp_structure is not None:
        if args.mlp_structure == "shallow":
            mlp_hidden = [256, 256]
        elif args.mlp_structure == "intermediate":
            mlp_hidden = [256, 512, 256]
        elif args.mlp_structure == "deep":
            mlp_hidden = [256, 1024, 1024, 256]
        else:
            raise ValueError(f"Unknown mlp_structure: {args.mlp_structure}")
    else:
        mlp_hidden = [256, 1024, 1024, 256]

    # Build model according to requested type and input modality
    if use_mask:
        if args.model_type == "slotsetpool":
            model = SlotSetPool(input_dim=input_dim, logm_idx=args.slot_logm_idx, H=args.slot_H, K=args.slot_K,
                                out_dim=target_dim, dropout=args.slot_dropout)
        elif getattr(args, 'multi_phi', False):
            # default to two φ networks with hidden sizes [64,64] each
            phi_hiddens = [[64, 64]]*13
            model = DeepSetMultiPhi(input_dim=input_dim, phi_hiddens=phi_hiddens, rho_hidden=[128, 128], agg="mean", out_dim=target_dim)
        else:
            # default DeepSet
            model = DeepSet(input_dim=input_dim, phi_hidden=[256,128,512], rho_hidden=[1024,1024,128], agg="mean", out_dim=target_dim)
    else:
        # fixed-vector inputs: always use MLP
        model = MLP(in_dim=input_dim, hidden=mlp_hidden, out_dim=target_dim)

    model_name = args.model_type if use_mask else 'MLP'
    print(f"Model: {model_name}, input_dim: {input_dim}, target_dim: {target_dim}")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Set up prediction function based on model type
    if use_mask:
        predict_fn = lambda m, X, mask: run_model(m, X, mask)
    else:
        predict_fn = lambda m, X, mask: run_model(m, X, mask)
    
    for ep in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, opt, loss_fn, device,
                                                input_norm=args.normalize_input, input_stats=input_stats,
                                                output_norm=args.normalize_output, output_stats=output_stats,
                                                predict_fn=predict_fn, mask_cfg=mask_cfg)
        val_loss, val_metrics = eval_model(model, val_loader, loss_fn, device,
                                          input_norm=args.normalize_input, input_stats=input_stats,
                                          output_norm=args.normalize_output, output_stats=output_stats,
                                          predict_fn=predict_fn)
        # optional per-epoch test evaluation
        test_loss = None
        test_metrics = None
        if test_loader is not None:
            test_loss, test_metrics = eval_model(model, test_loader, loss_fn, device,
                                                 input_norm=args.normalize_input, input_stats=input_stats,
                                                 output_norm=args.normalize_output, output_stats=output_stats,
                                                 predict_fn=predict_fn)
        print(
            f"Epoch {ep:03d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"train_mse={train_metrics['mse']:.6e}  val_mse={val_metrics['mse']:.6e}  "
            f"train_rel={train_metrics['rel_err']:.6e}  val_rel={val_metrics['rel_err']:.6e}  "
            f"train_r2={train_metrics['r2']:.6f}  val_r2={val_metrics['r2']:.6f}"
            + (f"  test_loss={test_loss:.6f}  test_mse={test_metrics['mse']:.6e}  test_r2={test_metrics['r2']:.6f}" if test_metrics is not None else "")
        )

        if wandb is not None:
            log_dict = {
                "epoch": ep,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_mse": float(train_metrics["mse"]),
                "val_mse": float(val_metrics["mse"]),
                "train_rel_err": float(train_metrics["rel_err"]),
                "val_rel_err": float(val_metrics["rel_err"]),
                "train_r2": float(train_metrics["r2"]),
                "val_r2": float(val_metrics["r2"]),
            }
            if test_metrics is not None:
                log_dict.update({
                    "test_loss": float(test_loss),
                    "test_mse": float(test_metrics["mse"]),
                    "test_rel_err": float(test_metrics["rel_err"]),
                    "test_r2": float(test_metrics["r2"]),
                })
            # add per-target metrics as individual entries
            try:
                # derive labels from dataset if available
                labels = None
                try:
                    # train_loader.dataset may be a Subset
                    ds_obj = getattr(train_loader, 'dataset', None)
                    if hasattr(ds_obj, 'dataset'):
                        ds_real = ds_obj.dataset
                    else:
                        ds_real = ds_obj
                    labels = getattr(ds_real, 'param_keys', None)
                except Exception:
                    labels = None

                mse_per = train_metrics.get("mse_per_target", None)
                r2_per = train_metrics.get("r2_per_target", None)
                if mse_per is not None and r2_per is not None and len(mse_per) == len(r2_per):
                    for i in range(len(mse_per)):
                        label = (labels[i] if labels is not None and i < len(labels) else f"target_{i}")
                        safe_label = str(label).replace('/', '_').replace(' ', '_')
                        log_dict[f"train/mse/{safe_label}"] = float(mse_per[i])
                        log_dict[f"train/r2/{safe_label}"] = float(r2_per[i])

                mse_per_v = val_metrics.get("mse_per_target", None)
                r2_per_v = val_metrics.get("r2_per_target", None)
                if mse_per_v is not None and r2_per_v is not None and len(mse_per_v) == len(r2_per_v):
                    for i in range(len(mse_per_v)):
                        label = (labels[i] if labels is not None and i < len(labels) else f"target_{i}")
                        safe_label = str(label).replace('/', '_').replace(' ', '_')
                        log_dict[f"val/mse/{safe_label}"] = float(mse_per_v[i])
                        log_dict[f"val/r2/{safe_label}"] = float(r2_per_v[i])
                # per-target test metrics
                try:
                    if test_metrics is not None:
                        mse_per_t = test_metrics.get("mse_per_target", None)
                        r2_per_t = test_metrics.get("r2_per_target", None)
                        if mse_per_t is not None and r2_per_t is not None and len(mse_per_t) == len(r2_per_t):
                            for i in range(len(mse_per_t)):
                                label = (labels[i] if labels is not None and i < len(labels) else f"target_{i}")
                                safe_label = str(label).replace('/', '_').replace(' ', '_')
                                log_dict[f"test/mse/{safe_label}"] = float(mse_per_t[i])
                                log_dict[f"test/r2/{safe_label}"] = float(r2_per_t[i])
                except Exception:
                    pass
            except Exception:
                # be conservative: don't let logging failures break training
                pass
            wandb.log(log_dict)

        if args.plot_every > 0 and (ep % args.plot_every == 0):
            # produce 1:1 scatter plots for train and validation predictions
            try:
                os.makedirs(args.plot_path, exist_ok=True)
                train_y_true, train_y_pred = collect_predictions(model, train_loader, device,
                                                                 input_norm=args.normalize_input, input_stats=input_stats,
                                                                 output_norm=args.normalize_output, output_stats=output_stats,
                                                                 predict_fn=predict_fn)
                val_y_true, val_y_pred = collect_predictions(model, val_loader, device,
                                                             input_norm=args.normalize_input, input_stats=input_stats,
                                                             output_norm=args.normalize_output, output_stats=output_stats,
                                                             predict_fn=predict_fn)

                train_plot_path = os.path.join(args.plot_path, f"pred_vs_true_train_ep{ep}.png")
                val_plot_path = os.path.join(args.plot_path, f"pred_vs_true_val_ep{ep}.png")
                plot_pred_vs_true(train_y_true, train_y_pred, train_plot_path, title=f"Train: pred vs true (ep {ep})")
                plot_pred_vs_true(val_y_true, val_y_pred, val_plot_path, title=f"Val: pred vs true (ep {ep})")
                # per-epoch test plot
                if test_loader is not None:
                    try:
                        test_y_true, test_y_pred = collect_predictions(model, test_loader, device,
                                                                       input_norm=args.normalize_input, input_stats=input_stats,
                                                                       output_norm=args.normalize_output, output_stats=output_stats,
                                                                       predict_fn=predict_fn)
                        test_plot_path = os.path.join(args.plot_path, f"pred_vs_true_test_ep{ep}.png")
                        plot_pred_vs_true(test_y_true, test_y_pred, test_plot_path, title=f"Test: pred vs true (ep {ep})")
                        if wandb is not None:
                            wandb.log({f"pred_vs_true_test_ep{ep}": wandb.Image(test_plot_path)})
                    except Exception as e:
                        print(f"Warning: failed to produce per-epoch test plot: {e}")
                print(f"Saved pred-vs-true plots: {train_plot_path}, {val_plot_path}")
                if wandb is not None:
                    wandb.log({"pred_vs_true_train": wandb.Image(train_plot_path), "pred_vs_true_val": wandb.Image(val_plot_path)})
            except Exception as e:
                print(f"Warning: failed to save pred-vs-true plots: {e}")

    # Final evaluation on held-out test set (if available)
    test_loss = None
    test_metrics = None
    if test_loader is not None:
        test_loss, test_metrics = eval_model(model, test_loader, loss_fn, device,
                                             input_norm=args.normalize_input, input_stats=input_stats,
                                             output_norm=args.normalize_output, output_stats=output_stats,
                                             predict_fn=predict_fn)
        print(
            f"Test  loss={test_loss:.6f}  test_mse={test_metrics['mse']:.6e}  "
            f"test_rel={test_metrics['rel_err']:.6e}  test_r2={test_metrics['r2']:.6f}")
        if wandb is not None:
            wandb.log({
                "test_loss": float(test_loss),
                "test_mse": float(test_metrics["mse"]),
                "test_rel_err": float(test_metrics["rel_err"]),
                "test_r2": float(test_metrics["r2"]),
            })
            try:
                run = getattr(wandb, 'run', None)
                if run is not None:
                    run.summary["test_loss"] = float(test_loss)
                    run.summary["test_mse"] = float(test_metrics["mse"])
                    run.summary["test_r2"] = float(test_metrics["r2"])
                    # per-target test metrics
                    try:
                        mse_per_t = test_metrics.get("mse_per_target", None)
                        r2_per_t = test_metrics.get("r2_per_target", None)
                        if mse_per_t is not None and r2_per_t is not None and len(mse_per_t) == len(r2_per_t):
                            # derive labels from dataset if available
                            labels = None
                            try:
                                ds_obj = getattr(test_loader, 'dataset', None)
                                if hasattr(ds_obj, 'dataset'):
                                    ds_real = ds_obj.dataset
                                else:
                                    ds_real = ds_obj
                                labels = getattr(ds_real, 'param_keys', None)
                            except Exception:
                                labels = None

                            d = {}
                            for i in range(len(mse_per_t)):
                                label = (labels[i] if labels is not None and i < len(labels) else f"target_{i}")
                                safe_label = str(label).replace('/', '_').replace(' ', '_')
                                d[f"test/mse/{safe_label}"] = float(mse_per_t[i])
                                d[f"test/r2/{safe_label}"] = float(r2_per_t[i])
                            wandb.log(d)
                    except Exception:
                        pass
            except Exception:
                pass
        if args.plot_every > 0:
            try:
                os.makedirs(args.plot_path, exist_ok=True)
                test_y_true, test_y_pred = collect_predictions(model, test_loader, device,
                                                               input_norm=args.normalize_input, input_stats=input_stats,
                                                               output_norm=args.normalize_output, output_stats=output_stats,
                                                               predict_fn=predict_fn)
                test_plot_path = os.path.join(args.plot_path, "pred_vs_true_test_final.png")
                plot_pred_vs_true(test_y_true, test_y_pred, test_plot_path, title="Test: pred vs true (final)")
                print(f"Saved test pred-vs-true plot: {test_plot_path}")
                if wandb is not None:
                    wandb.log({"pred_vs_true_test": wandb.Image(test_plot_path)})
            except Exception as e:
                print(f"Warning: failed to save test pred-vs-true plot: {e}")

    # Save checkpoint locally and upload to WandB if requested
    if args.save_model:
        os.makedirs(args.save_path, exist_ok=True)
        ckpt_path = os.path.join(args.save_path, f"deepset_epoch{args.epochs}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "args": vars(args),
            "input_stats": input_stats,
            "output_stats": output_stats,
        }, ckpt_path)
        print(f"Saved model checkpoint to {ckpt_path}")
        if wandb is not None and args.wandb_save_model:
            artifact = wandb.Artifact(name=f"deepset-checkpoint-{wandb.run.id}", type="model")
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)
            print("Uploaded checkpoint to WandB as artifact")


if __name__ == "__main__":
    main()
