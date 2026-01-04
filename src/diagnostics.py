"""Dataset & normalization sanity checks for CAMELS HDF5.

Run to inspect targets and inputs, check for constant targets,
and produce simple plots saved to `data/models/plots`.

Example:
    python src/diagnostics.py --h5-path data/camels_LH.hdf5 --snap 90 --sample 500
"""
import argparse
import os
import numpy as np
import math
from collections import defaultdict

import matplotlib.pyplot as plt

from data import HDF5SetDataset, hdf5_collate, smf_collate
from torch.utils.data import DataLoader


def summarize_targets(ys: np.ndarray):
    # ys: (M,) or (M,K)
    ys = np.atleast_2d(ys)
    M, K = ys.shape
    lines = []
    for k in range(K):
        arr = ys[:, k]
        lines.append(f"target[{k}]: n={M} min={arr.min():.6g} max={arr.max():.6g} mean={arr.mean():.6g} std={arr.std():.6g} uniq={np.unique(arr).size}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, default="../data/camels_LH.hdf5")
    parser.add_argument("--snap", type=int, default=90)
    parser.add_argument("--param-keys", nargs="+", default=None)
    parser.add_argument("--use-smf", action="store_true")
    parser.add_argument("--sample", type=int, default=1000, help="Number of samples to inspect (max dataset length)")
    parser.add_argument("--plot-path", type=str, default="../data/plots")
    args = parser.parse_args(argv)

    ds = HDF5SetDataset(h5_path=args.h5_path, snap=args.snap, param_keys=args.param_keys, data_field=("smf/phi" if args.use_smf else "SubhaloStellarMass"))
    n = len(ds)
    print(f"Dataset length: {n}")
    M = min(n, args.sample)

    ys = []
    x_means = []
    x_vecs = []
    for i in range(M):
        x, y = ds[i]
        y_arr = np.atleast_1d(np.array(y, dtype=np.float64))
        ys.append(y_arr)

        x_np = np.array(x, dtype=np.float64)
        if x_np.ndim == 1:
            # fixed length vector (SMF)
            x_means.append(x_np.mean())
            x_vecs.append(x_np)
        else:
            # set of per-halo features; compute mean across halos and features
            x_means.append(x_np.mean())
            # also capture first-10 features aggregated if available
            x_vecs.append(x_np.reshape(-1))

    ys = np.stack(ys, axis=0)
    x_means = np.array(x_means)

    os.makedirs(args.plot_path, exist_ok=True)

    # Print target labels (param keys) if available
    labels = getattr(ds, 'param_keys', None)
    if labels is None:
        label_str = '[unknown]'
    else:
        label_str = ', '.join(labels)
    print(f"\nTarget labels: {label_str}")
    print("\nTarget summary:")
    print(summarize_targets(ys))

    # Check for constant targets
    if np.allclose(ys.std(axis=0), 0.0):
        print("WARNING: targets appear constant (zero std) for all requested keys.")
    else:
        const_keys = np.where(ys.std(axis=0) == 0.0)[0]
        if const_keys.size > 0:
            print(f"WARNING: targets constant for dimensions: {const_keys.tolist()}")

    # histogram of first target
    try:
        plt.figure(figsize=(6, 4))
        plt.hist(ys[:, 0], bins=40)
        plt.title("Target[0] distribution")
        plt.tight_layout()
        pth = os.path.join(args.plot_path, "target0_hist.png")
        plt.savefig(pth)
        plt.close()
        print(f"Saved {pth}")
    except Exception as e:
        print(f"Failed to save target histogram: {e}")

    # correlation between input mean and first target
    try:
        if x_means.size == ys.shape[0]:
            corr = np.corrcoef(x_means, ys[:, 0])[0, 1]
            print(f"Correlation between per-sample mean(input) and target[0]: {corr:.4f}")
            plt.figure(figsize=(5, 4))
            plt.scatter(x_means, ys[:, 0], s=6, alpha=0.4)
            plt.xlabel("mean(input)")
            plt.ylabel("target[0]")
            plt.title(f"corr={corr:.3f}")
            pth = os.path.join(args.plot_path, "input_mean_vs_target0.png")
            plt.tight_layout()
            plt.savefig(pth)
            plt.close()
            print(f"Saved {pth}")
    except Exception as e:
        print(f"Failed to compute input-target correlation: {e}")

    # Save per-target histograms and input_mean vs target scatter for all targets
    K = ys.shape[1]
    input_mean_corrs = []
    for k in range(K):
        try:
            plt.figure(figsize=(6, 4))
            plt.hist(ys[:, k], bins=40)
            plt.title(f"Target[{k}] distribution")
            plt.tight_layout()
            pth = os.path.join(args.plot_path, f"target{k}_hist.png")
            plt.savefig(pth)
            plt.close()
            print(f"Saved {pth}")
        except Exception as e:
            print(f"Failed to save histogram for target {k}: {e}")

        try:
            if x_means.size == ys.shape[0]:
                corr = np.corrcoef(x_means, ys[:, k])[0, 1]
                input_mean_corrs.append(corr)
                plt.figure(figsize=(5, 4))
                plt.scatter(x_means, ys[:, k], s=6, alpha=0.4)
                plt.xlabel("mean(input)")
                plt.ylabel(f"target[{k}]")
                plt.title(f"corr={corr:.3f}")
                pth = os.path.join(args.plot_path, f"input_mean_vs_target{k}.png")
                plt.tight_layout()
                plt.savefig(pth)
                plt.close()
                print(f"Saved {pth}")
            else:
                input_mean_corrs.append(np.nan)
        except Exception as e:
            input_mean_corrs.append(np.nan)
            print(f"Failed to save input_mean vs target {k}: {e}")

    # Print and save input_mean vs target correlations for all targets
    if len(input_mean_corrs) > 0:
        print("\nInput-mean vs target correlations:")
        for k, c in enumerate(input_mean_corrs):
            lab = labels[k] if labels is not None and k < len(labels) else f"target[{k}]"
            print(f" {lab}: corr={c:.6f}")
        # save CSV
        try:
            import csv
            csv_path = os.path.join(args.plot_path, 'input_mean_target_correlations.csv')
            with open(csv_path, 'w', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(['target_index', 'target_label', 'corr_input_mean'])
                for k, c in enumerate(input_mean_corrs):
                    lab = labels[k] if labels is not None and k < len(labels) else f"target[{k}]"
                    writer.writerow([k, lab, float(c) if not np.isnan(c) else 'nan'])
            print(f"Saved {csv_path}")
        except Exception as e:
            print(f"Failed to save input-mean correlations CSV: {e}")

    # If we have per-feature vectors (SMF), compute feature-target correlations
    try:
        if len(x_vecs) > 0:
            # Check if all vectors have same length
            lens = [v.shape[0] for v in x_vecs]
            if len(set(lens)) == 1:
                Xmat = np.stack(x_vecs, axis=0)  # (M, D)
                D = Xmat.shape[1]
                K = ys.shape[1]
                corr_mat = np.zeros((D, K), dtype=float)
                for d in range(D):
                    for k in range(K):
                        vx = Xmat[:, d]
                        vy = ys[:, k]
                        if np.std(vx) == 0 or np.std(vy) == 0:
                            corr_mat[d, k] = 0.0
                        else:
                            corr_mat[d, k] = np.corrcoef(vx, vy)[0, 1]

                # plot heatmap
                plt.figure(figsize=(6, max(3, K * 0.5)))
                plt.imshow(corr_mat.T, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
                plt.colorbar(label='pearson r')
                plt.xlabel('feature (SMF bin)')
                plt.ylabel('target index')
                pth = os.path.join(args.plot_path, 'feature_target_corr.png')
                plt.tight_layout()
                plt.savefig(pth)
                plt.close()
                print(f"Saved {pth}")
            else:
                print("Skipping per-feature correlations: variable-length feature vectors present.")
    except Exception as e:
        print(f"Failed to compute per-feature correlations: {e}")

    # Optional quick baseline: train small MLP on SMF vectors or input means
    parser = None
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except Exception:
        torch = None

    # If user passed --baseline in argv, run a quick training to see if loss decreases
    import sys
    if '--baseline' in sys.argv:
        if torch is None:
            print("Skipping baseline: PyTorch not available in this environment.")
        else:
            print("Running quick baseline MLP training (this may take a moment)...")
            # Prepare features: prefer per-feature SMF if available and consistent, else use mean
            if len(x_vecs) > 0 and len(set([v.shape[0] for v in x_vecs])) == 1:
                Xmat = np.stack(x_vecs, axis=0)
            else:
                Xmat = x_means.reshape(-1, 1)

            # Use first 80% for train, rest val
            M = Xmat.shape[0]
            ntrain = int(M * 0.8)
            X_train = torch.from_numpy(Xmat[:ntrain].astype(np.float32))
            y_train = torch.from_numpy(ys[:ntrain].astype(np.float32))
            X_val = torch.from_numpy(Xmat[ntrain:].astype(np.float32))
            y_val = torch.from_numpy(ys[ntrain:].astype(np.float32))

            train_ds = TensorDataset(X_train, y_train)
            val_ds = TensorDataset(X_val, y_val)
            tr_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            va_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

            # small MLP
            nin = X_train.shape[1]
            nout = y_train.shape[1]
            net = nn.Sequential(nn.Linear(nin, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, nout))
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            net.train()
            for ep in range(1, 51):
                total = 0.0
                for xb, yb in tr_loader:
                    opt.zero_grad()
                    pred = net(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
                    total += float(loss.item()) * xb.shape[0]
                if ep % 10 == 0:
                    avg = total / len(train_ds)
                    # val
                    net.eval()
                    with torch.no_grad():
                        vloss = 0.0
                        for xb, yb in va_loader:
                            vpred = net(xb)
                            vloss += float(loss_fn(vpred, yb).item()) * xb.shape[0]
                    vavg = vloss / len(val_ds)
                    print(f"Baseline ep {ep:02d} train_loss={avg:.6f} val_loss={vavg:.6f}")
                    net.train()

    # show shapes from a single collated batch for both collate fns (if applicable)
    try:
        # take first 8 indices
        sample_idx = list(range(min(8, n)))
        sub_samples = [ds[i] for i in sample_idx]
        xb, yb = smf_collate(sub_samples)
        print(f"smf_collate -> X.shape={xb.shape}, y.shape={yb.shape}")
    except Exception:
        pass

    try:
        batch = sub_samples
        Xp, maskp, yp = hdf5_collate(batch)
        print(f"hdf5_collate -> X.shape={Xp.shape}, mask.shape={maskp.shape}, y.shape={yp.shape}")
    except Exception:
        pass

    print("\nDiagnostics complete. If targets look constant or uncorrelated with inputs, check the HDF5 file layout and requested `param_keys`.")


if __name__ == "__main__":
    main()
