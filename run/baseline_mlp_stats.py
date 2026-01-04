import os
import sys
import argparse
from typing import Optional, List, Tuple
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

# Ensure imports work (set_dsr, stats_sym, data)
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, REPO_ROOT)

from data import HDF5SetDataset, hdf5_collate
from stats_sym import SummaryStatistics, SimplifiedSetDSR, PredictionHead

# Now that src is on sys.path, import training helpers from src/train.py
from train import normalize_batch, compute_stats_from_dataset


def ridge_fit_closed_form(X: np.ndarray, Y: np.ndarray, alpha: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """Fit multi-target ridge regression with intercept.

    Solves: min_B ||X B - Y||^2 + alpha ||B||^2
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
    input_norm: str = "none",
    input_stats: dict = None,
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

            # apply per-element input normalization (e.g. log_std) before computing per-element stats
            if input_norm is not None and input_norm != "none":
                X, mask, y = normalize_batch(X, mask, y, input_norm, input_stats, output_norm="none", output_stats=None)

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


def train_mlp_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    hidden_dims: List[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    n_epochs: int,
    device: torch.device,
    seed: int,
) -> PredictionHead:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).float()
    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
    )

    if X_val is not None and y_val is not None:
        Xva = torch.from_numpy(X_val).float()
        yva = torch.from_numpy(y_val).float()
        val_loader = DataLoader(
            TensorDataset(Xva, yva),
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    model = PredictionHead(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dims=hidden_dims,
        use_mlp=True,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_state = None
    best_val_r2 = -1e9

    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            preds = []
            targets = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds.append(model(xb).cpu().numpy())
                    targets.append(yb.cpu().numpy())
            y_pred = np.concatenate(preds, axis=0)
            y_true = np.concatenate(targets, axis=0)
            val_r2 = r2_score(y_true, y_pred)
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def main(argv=None):
    p = argparse.ArgumentParser(description="Baseline: classical summary stats + MLP head (compare to ridge)")
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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    p.add_argument(
        "--input-norm",
        type=str,
        default="none",
        choices=["none", "log", "log_std"],
        help="Per-element input normalization applied BEFORE computing summary stats",
    )
    p.add_argument(
        "--use-model-features",
        action="store_true",
        default=False,
        help="Compute summary features via SimplifiedSetDSR.compute_summary_features instead of SummaryStatistics",
    )
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
    idx_val = indices[n_train: n_train + n_val]
    idx_test = indices[n_train + n_val:]

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

    # Build model once if needed
    model = None
    if args.use_model_features:
        print("Will compute summary-stat features via `SimplifiedSetDSR.compute_summary_features` (raw features, no symbolic transforms)...")
        model = SimplifiedSetDSR(
            n_features=1 if ds.max_size > 0 and getattr(ds, "max_size", 0) else 1,
            n_transforms=1,
            output_dim=1,
            top_k=16,
            include_moments=True,
            include_cumulants=True,
            include_quantiles=False,
            use_learnable_weights=False,
            selection_method="fixed",
            use_mlp_head=False,
            use_symbolic_transforms=False,
            use_top_k=False,
        ).to(device)

    # helper to compute input_stats for a given norm choice (cache by name)
    stats_cache = {}

    def get_input_stats(norm_choice: str):
        if norm_choice not in ("log_std",):
            return None
        if norm_choice in stats_cache:
            return stats_cache[norm_choice]
        print(f"Estimating input statistics from training set for input normalization '{norm_choice}'...")
        stats_cache[norm_choice] = compute_stats_from_dataset(train_ds)
        return stats_cache[norm_choice]

    def compute_features_for_norm(norm_choice: str):
        stats = get_input_stats(norm_choice)
        if args.use_model_features:
            # compute via model
            all_X = []
            all_y = []
            model.eval()
            with torch.no_grad():
                for batch in train_loader:
                    Xb, maskb, yb = batch
                    Xb = Xb.to(device)
                    maskb = maskb.to(device)
                    yb = yb.to(device)
                    if norm_choice is not None and norm_choice != "none":
                        Xb, maskb, yb = normalize_batch(Xb, maskb, yb, norm_choice, stats, output_norm="none", output_stats=None)
                    feats, _ = model.compute_summary_features(Xb, maskb)
                    all_X.append(feats.cpu().numpy())
                    all_y.append(yb.cpu().numpy())
            X_train_np = np.concatenate(all_X, axis=0)
            y_train_np = np.concatenate(all_y, axis=0)

            def _compute_split(loader):
                if loader is None:
                    return None, None
                all_Xs, all_ys = [], []
                with torch.no_grad():
                    for batch in loader:
                        Xb, maskb, yb = batch
                        Xb = Xb.to(device)
                        maskb = maskb.to(device)
                        yb = yb.to(device)
                        if norm_choice is not None and norm_choice != "none":
                            Xb, maskb, yb = normalize_batch(Xb, maskb, yb, norm_choice, stats, output_norm="none", output_stats=None)
                        feats, _ = model.compute_summary_features(Xb, maskb)
                        all_Xs.append(feats.cpu().numpy())
                        all_ys.append(yb.cpu().numpy())
                return np.concatenate(all_Xs, axis=0), np.concatenate(all_ys, axis=0)

            X_val_np, y_val_np = _compute_split(val_loader)
            X_test_np, y_test_np = _compute_split(test_loader) if test_loader is not None else (None, None)
        else:
            # compute via SummaryStatistics
            X_train_np, y_train_np = compute_stat_features(train_loader, summary, device, input_norm=norm_choice, input_stats=stats)
            X_val_np, y_val_np = compute_stat_features(val_loader, summary, device, input_norm=norm_choice, input_stats=stats)
            if test_loader is not None:
                X_test_np, y_test_np = compute_stat_features(test_loader, summary, device, input_norm=norm_choice, input_stats=stats)
            else:
                X_test_np, y_test_np = None, None

        return (X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np)

    def eval_baselines(label: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, Xte: Optional[np.ndarray], yte: Optional[np.ndarray]):
        print(f"\n=== {label} features ===")
        print(f"Feature dim: {Xtr.shape[1]}")

        # Fit ridge
        W, b = ridge_fit_closed_form(Xtr, ytr, alpha=args.alpha)

        def ridge_predict(Xn: np.ndarray) -> np.ndarray:
            return Xn @ W + b[None, :]

        yhat_train = ridge_predict(Xtr)
        yhat_val = ridge_predict(Xva)
        print("\nR² (ridge/train):", r2_score(ytr, yhat_train))
        print("R² per target (ridge/train):")
        for name, score in zip(ds.param_keys, r2_per_target(ytr, yhat_train)):
            print(f"  {name}: {score:.4f}")

        print("\nR² (ridge/val):", r2_score(yva, yhat_val))
        print("R² per target (ridge/val):")
        for name, score in zip(ds.param_keys, r2_per_target(yva, yhat_val)):
            print(f"  {name}: {score:.4f}")

        if Xte is not None and yte is not None:
            yhat_test = ridge_predict(Xte)
            print("\nR² (ridge/test):", r2_score(yte, yhat_test))
            print("R² per target (ridge/test):")
            for name, score in zip(ds.param_keys, r2_per_target(yte, yhat_test)):
                print(f"  {name}: {score:.4f}")

        # Train MLP head
        print("\nTraining MLP head...")
        mlp = train_mlp_head(
            X_train=Xtr,
            y_train=ytr,
            X_val=Xva,
            y_val=yva,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            device=device,
            seed=args.seed,
        )

        with torch.no_grad():
            yhat_train_mlp = mlp(torch.from_numpy(Xtr).float().to(device)).cpu().numpy()
            yhat_val_mlp = mlp(torch.from_numpy(Xva).float().to(device)).cpu().numpy()
            print("\nR² (mlp/train):", r2_score(ytr, yhat_train_mlp))
            print("R² per target (mlp/train):")
            for name, score in zip(ds.param_keys, r2_per_target(ytr, yhat_train_mlp)):
                print(f"  {name}: {score:.4f}")

            print("\nR² (mlp/val):", r2_score(yva, yhat_val_mlp))
            print("R² per target (mlp/val):")
            for name, score in zip(ds.param_keys, r2_per_target(yva, yhat_val_mlp)):
                print(f"  {name}: {score:.4f}")

            if Xte is not None and yte is not None:
                yhat_test_mlp = mlp(torch.from_numpy(Xte).float().to(device)).cpu().numpy()
                print("\nR² (mlp/test):", r2_score(yte, yhat_test_mlp))
                print("R² per target (mlp/test):")
                for name, score in zip(ds.param_keys, r2_per_target(yte, yhat_test_mlp)):
                    print(f"  {name}: {score:.4f}")

    # Run four combinations: input_norm on/off × feature standardization on/off.
    # Build the list of input_norm choices: always include "none"; if user specified something else, include that too.
    input_norm_choices = ["none"]
    if args.input_norm != "none":
        input_norm_choices.append(args.input_norm)

    feat_norm_choices = [False, True]  # False = no standardization, True = standardize

    for input_norm_choice in input_norm_choices:
        X_tr, y_tr, X_va, y_va, X_te, y_te = compute_features_for_norm(input_norm_choice)

        # Ensure y shapes are (N,K)
        if y_tr.ndim == 1:
            y_tr = y_tr[:, None]
            y_va = y_va[:, None]
            if y_te is not None:
                y_te = y_te[:, None]

        for do_feat_norm in feat_norm_choices:
            if do_feat_norm:
                X_tr_use, mu, sd = standardize_fit(X_tr)
                X_va_use = standardize_apply(X_va, mu, sd)
                X_te_use = standardize_apply(X_te, mu, sd) if X_te is not None else None
                feat_label = "standardize"
            else:
                X_tr_use, X_va_use, X_te_use = X_tr, X_va, X_te
                feat_label = "none"

            label = f"input_norm={input_norm_choice}, feat_norm={feat_label}"
            eval_baselines(label, X_tr_use, y_tr, X_va_use, y_va, X_te_use, y_te)


if __name__ == "__main__":
    main()
