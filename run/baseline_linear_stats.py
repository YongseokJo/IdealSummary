import os
import sys
import argparse
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Ensure imports work (set_dsr, stats_sym, data)
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, REPO_ROOT)

from data import HDF5SetDataset, hdf5_collate
from stats_sym import SummaryStatistics, SimplifiedSetDSR


def ridge_fit_closed_form(X: np.ndarray, Y: np.ndarray, alpha: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """Fit multi-target ridge regression with intercept.

    Solves: min_B ||X B - Y||^2 + alpha ||B||^2

    Args:
        X: (N, P)
        Y: (N, K)
        alpha: ridge penalty

    Returns:
        W: (P, K)
        b: (K,)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if Y.ndim == 1:
        Y = Y[:, None]

    N, P = X.shape
    K = Y.shape[1]

    # Add bias column
    Xb = np.concatenate([X, np.ones((N, 1), dtype=X.dtype)], axis=1)  # (N, P+1)

    # Regularize weights but not bias
    I = np.eye(P + 1, dtype=X.dtype)
    I[-1, -1] = 0.0

    A = Xb.T @ Xb + alpha * I
    B = Xb.T @ Y
    coef = np.linalg.solve(A, B)  # (P+1, K)

    W = coef[:-1, :]
    b = coef[-1, :]
    return W, b


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2) + eps
    return float(1.0 - ss_res / ss_tot)


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> List[float]:
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    out = []
    for k in range(y_true.shape[1]):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2) + eps
        out.append(float(1.0 - ss_res / ss_tot))
    return out


def compute_stat_features(
    loader: DataLoader,
    summary: SummaryStatistics,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute classical summary stats feature vectors for all samples in loader."""
    all_X = []
    all_y = []

    summary = summary.to(device)
    summary.eval()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            X, mask, y = batch
            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device)

            B, N, D = X.shape
            per_feat_stats = []
            for d in range(D):
                vals = X[:, :, d]
                stats = summary(vals, mask, weights=None)  # (B, n_stats)
                per_feat_stats.append(stats)
            feats = torch.cat(per_feat_stats, dim=1)  # (B, D*n_stats)

            all_X.append(feats.cpu().numpy())
            all_y.append(y.cpu().numpy())

    X_np = np.concatenate(all_X, axis=0)
    y_np = np.concatenate(all_y, axis=0)
    return X_np, y_np


def standardize_fit(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    Xn = (X - mu) / (sd + eps)
    return Xn, mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (X - mu) / (sd + eps)


def main(argv=None):
    p = argparse.ArgumentParser(description="Baseline: classical summary stats + Ridge regression")
    p.add_argument("--h5-path", type=str, default="../data/camels_LH.hdf5")
    p.add_argument("--snap", type=int, default=90)
    p.add_argument("--data-field", type=str, default="SubhaloStellarMass")
    p.add_argument("--param-keys", nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=1e-2, help="Ridge penalty")
    p.add_argument("--max-samples", type=int, default=0, help="If >0, subsample that many total samples for speed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-model-features", action="store_true", default=False,
                   help="Compute summary features via SimplifiedSetDSR.compute_summary_features instead of SummaryStatistics")
    args = p.parse_args(argv)

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    device = torch.device("cpu")

    ds = HDF5SetDataset(
        h5_path=args.h5_path,
        snap=args.snap,
        param_keys=args.param_keys,
        data_field=args.data_field,
    )

    n_total = len(ds)
    rng = np.random.default_rng(args.seed)

    indices = np.arange(n_total)
    rng.shuffle(indices)

    if args.max_samples and args.max_samples > 0:
        indices = indices[: min(args.max_samples, n_total)]

    n_total = len(indices)
    n_train = int(round(args.train_frac * n_total))
    n_val = int(round(args.val_frac * n_total))
    n_test = max(0, n_total - n_train - n_val)

    idx_train = indices[:n_train]
    idx_val = indices[n_train : n_train + n_val]
    idx_test = indices[n_train + n_val :]

    train_ds = Subset(ds, idx_train.tolist())
    val_ds = Subset(ds, idx_val.tolist())
    test_ds = Subset(ds, idx_test.tolist()) if n_test > 0 else None

    collate_fn = lambda batch: hdf5_collate(batch, max_size=ds.max_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if test_ds else None

    # SummaryStatistics: classical stats, no quantiles by default
    summary = SummaryStatistics(include_moments=True, include_cumulants=True, include_quantiles=False)

    print(f"Dataset samples: {n_total} (train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds) if test_ds else 0})")
    if hasattr(ds, "param_names") and ds.param_names is not None:
        print(f"Targets: {ds.param_keys}")

    if args.use_model_features:
        print("Computing summary-stat features via `SimplifiedSetDSR.compute_summary_features`...")
        # Build a model that uses raw features (no symbolic transforms), no weights, no top-k
        model = SimplifiedSetDSR(
            n_features=1 if ds.max_size > 0 and getattr(ds, 'max_size', 0) else 1,
            n_transforms=1,
            output_dim=1,
            top_k=16,
            include_moments=True,
            include_cumulants=True,
            include_quantiles=False,
            use_learnable_weights=False,
            selection_method='fixed',
            use_mlp_head=False,
            use_symbolic_transforms=False,
            use_top_k=False,
        )
        model = model.to(device)

        def compute_from_model(loader):
            all_X = []
            all_y = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    Xb, maskb, yb = batch
                    Xb = Xb.to(device)
                    maskb = maskb.to(device)
                    # compute features via model
                    feats, _ = model.compute_summary_features(Xb, maskb)
                    all_X.append(feats.cpu().numpy())
                    all_y.append(yb.numpy())
            return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

        X_train, y_train = compute_from_model(train_loader)
        X_val, y_val = compute_from_model(val_loader)
        if test_loader is not None:
            X_test, y_test = compute_from_model(test_loader)
        else:
            X_test, y_test = None, None
    else:
        print("Computing summary-stat features via `SummaryStatistics`...")
        X_train, y_train = compute_stat_features(train_loader, summary, device)
        X_val, y_val = compute_stat_features(val_loader, summary, device)
        if test_loader is not None:
            X_test, y_test = compute_stat_features(test_loader, summary, device)
        else:
            X_test, y_test = None, None

    # Ensure y is (N,K)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_val = y_val[:, None]
        if y_test is not None:
            y_test = y_test[:, None]

    # Standardize features for ridge
    X_train_n, mu, sd = standardize_fit(X_train)
    X_val_n = standardize_apply(X_val, mu, sd)
    X_test_n = standardize_apply(X_test, mu, sd) if X_test is not None else None

    print(f"Feature dim: {X_train_n.shape[1]}")

    # Fit ridge
    W, b = ridge_fit_closed_form(X_train_n, y_train, alpha=args.alpha)

    def predict(Xn: np.ndarray) -> np.ndarray:
        return Xn @ W + b[None, :]

    yhat_train = predict(X_train_n)
    yhat_val = predict(X_val_n)
    print("\nR² (train):", r2_score(y_train, yhat_train))
    print("R² per target (train):")
    for name, score in zip(ds.param_keys, r2_per_target(y_train, yhat_train)):
        print(f"  {name}: {score:.4f}")

    print("\nR² (val):", r2_score(y_val, yhat_val))
    print("R² per target (val):")
    for name, score in zip(ds.param_keys, r2_per_target(y_val, yhat_val)):
        print(f"  {name}: {score:.4f}")

    if X_test_n is not None:
        yhat_test = predict(X_test_n)
        print("\nR² (test):", r2_score(y_test, yhat_test))
        print("R² per target (test):")
        for name, score in zip(ds.param_keys, r2_per_target(y_test, yhat_test)):
            print(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    main()
