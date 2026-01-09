#!/usr/bin/env python
"""
Extract per-parameter best validation metrics from Optuna studies.

This script:
1. Loads best hyperparameters from each Optuna study
2. Optionally re-trains the model with those hyperparameters 
3. Evaluates on validation set to get per-parameter MSE/R²
4. Produces summary tables and bar plots

Usage:
    # Basic summary (no per-param metrics, fast):
    python tools/optuna_perparam_analysis.py --out-dir data/analysis/optuna_perparam

    # With retraining to get per-param metrics (slow):
    python tools/optuna_perparam_analysis.py --out-dir data/analysis/optuna_perparam --retrain --epochs 100
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:
    print("optuna not installed. pip install optuna")
    sys.exit(1)


# Lazy torch imports for when retraining
def get_torch_modules():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from data import HDF5SetDataset, hdf5_collate, smf_collate
    from deepset import DeepSet, MLP
    from setpooling import SlotSetPool
    return torch, nn, DataLoader, Subset, HDF5SetDataset, hdf5_collate, smf_collate, DeepSet, MLP, SlotSetPool


def build_model(model_type: str, params: dict, input_dim: int, output_dim: int, torch_modules):
    """Build model from Optuna best hyperparameters."""
    torch, nn, DataLoader, Subset, HDF5SetDataset, hdf5_collate, smf_collate, DeepSet, MLP, SlotSetPool = torch_modules
    
    if model_type == "smf":
        mlp_layers = params.get("mlp_layers", 1)
        hidden_sizes = [params.get(f"mlp_h_{i}", 128) for i in range(mlp_layers)]
        return MLP(in_dim=input_dim, hidden=hidden_sizes, out_dim=output_dim)
    
    elif model_type == "deepset":
        phi_layers = params.get("phi_layers", 2)
        rho_layers = params.get("rho_layers", 2)
        phi_hidden = [params.get(f"phi_h_{i}", 64) for i in range(phi_layers)]
        rho_hidden = [params.get(f"rho_h_{i}", 64) for i in range(rho_layers)]
        agg = params.get("agg", "mean")
        return DeepSet(input_dim=input_dim, phi_hidden=phi_hidden, rho_hidden=rho_hidden, agg=agg, out_dim=output_dim)
    
    elif model_type == "slotsetpool":
        return SlotSetPool(
            input_dim=input_dim,
            K=params.get("slot_K", 10),
            H=params.get("slot_H", 128),
            out_dim=output_dim,
            phi_hidden=(params.get("slot_phi_h0", 64), params.get("slot_phi_h1", 64)),
            head_hidden=(params.get("slot_head_h0", 64), params.get("slot_head_h1", 64)),
            dropout=params.get("slot_dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def eval_perparam_metrics(model, loader, device, use_smf: bool):
    """Evaluate model and return per-parameter MSE and R²."""
    import torch
    model.eval()
    y_trues, y_preds = [], []
    
    with torch.no_grad():
        for batch in loader:
            if use_smf:
                X, y = batch
                X = X.to(device)
                mask = None
            elif len(batch) == 3:
                X, mask, y = batch
                X = X.to(device)
                mask = mask.to(device)
            else:
                X, y = batch
                X = X.to(device)
                mask = None
            
            # Forward pass
            if mask is not None:
                pred = model(X, mask)
            else:
                pred = model(X)
            
            y_np = y.cpu().numpy()
            p_np = pred.detach().cpu().numpy()
            y_np = y_np.reshape(y_np.shape[0], -1) if y_np.ndim == 1 else y_np
            p_np = p_np.reshape(p_np.shape[0], -1) if p_np.ndim == 1 else p_np
            y_trues.append(y_np)
            y_preds.append(p_np)
    
    if len(y_trues) == 0:
        return {}, {}
    
    y_trues = np.concatenate(y_trues, axis=0)
    y_preds = np.concatenate(y_preds, axis=0)
    
    # Per-parameter metrics
    K = y_trues.shape[1]
    mse_per = []
    r2_per = []
    for k in range(K):
        yt = y_trues[:, k]
        yp = y_preds[:, k]
        mse_k = float(np.mean((yt - yp) ** 2))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_k = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        mse_per.append(mse_k)
        r2_per.append(float(r2_k))
    
    return np.array(mse_per), np.array(r2_per)


def train_and_eval_perparam(
    model_type: str,
    params: dict,
    dataset_name: str,
    use_smf: bool,
    epochs: int = 100,
    batch_size: int = 64,
    seed: int = 42,
    device_str: str = "cuda",
):
    """Train model with best hyperparameters and get per-parameter metrics."""
    torch_mods = get_torch_modules()
    torch, nn, DataLoader, Subset, HDF5SetDataset, hdf5_collate, smf_collate, DeepSet, MLP, SlotSetPool = torch_mods
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # Load dataset - SMF uses different data field
    data_path = Path(__file__).parent.parent / "data" / f"camels_{dataset_name}.hdf5"
    data_field = "smf/phi" if use_smf else "SubhaloStellarMass"
    ds = HDF5SetDataset(str(data_path), snap=90, data_field=data_field)
    
    # Split (80/20)
    total = len(ds)
    train_size = int(0.8 * total)
    val_size = total - train_size
    
    indices = np.arange(total)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    
    if use_smf:
        collate_fn = smf_collate
    else:
        from functools import partial
        collate_fn = partial(hdf5_collate, max_size=ds.max_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Determine dimensions
    sample = next(iter(train_loader))
    if use_smf:
        X, y = sample
        input_dim = X.shape[1]
    elif len(sample) == 3:
        X, mask, y = sample
        input_dim = X.shape[2]
    else:
        X, y = sample
        input_dim = X.shape[1]
    
    output_dim = y.shape[1] if y.ndim > 1 else 1
    
    # Build model
    model = build_model(model_type, params, input_dim, output_dim, torch_mods)
    model = model.to(device)
    
    # Training
    lr = params.get("lr", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_mse_per = None
    best_r2_per = None
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if use_smf:
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                mask = None
            elif len(batch) == 3:
                X, mask, y = batch
                X = X.to(device)
                mask = mask.to(device)
                y = y.to(device)
            else:
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                mask = None
            
            optimizer.zero_grad()
            if mask is not None:
                pred = model(X, mask)
            else:
                pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        mse_per, r2_per = eval_perparam_metrics(model, val_loader, device, use_smf)
        val_loss = float(np.mean(mse_per))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mse_per = mse_per.copy()
            best_r2_per = r2_per.copy()
    
    return {
        "val_mse": float(np.mean(best_mse_per)),
        "val_r2": float(np.mean(best_r2_per)),
        "mse_per_target": best_mse_per.tolist(),
        "r2_per_target": best_r2_per.tolist(),
    }


def get_param_names(dataset_name: str) -> list:
    """Get parameter names from HDF5 file."""
    data_path = Path(__file__).parent.parent / "data" / f"camels_{dataset_name}.hdf5"
    with h5py.File(data_path, 'r') as f:
        param_names = f['parameters']['param_names'][:]
        if hasattr(param_names[0], 'decode'):
            param_names = [n.decode() for n in param_names]
    return list(param_names)


def load_optuna_best(db_path: str, study_name: str) -> dict:
    """Load best trial from Optuna study."""
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    best = study.best_trial
    return {
        "value": best.value,
        "params": best.params,
        "user_attrs": best.user_attrs,
        "n_trials": len(study.trials),
    }


def plot_overall_comparison(results: dict, category: str, out_dir: Path):
    """Create bar plots comparing overall metrics across models."""
    if not results:
        return
    
    models = list(results.keys())
    val_mse = [results[m]["val_mse"] for m in models]
    val_r2 = [results[m]["val_r2"] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.tab10(np.arange(len(models)))
    
    # MSE comparison
    bars = axes[0].bar(models, val_mse, color=colors)
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_title(f"{category} - Validation MSE Comparison")
    axes[0].tick_params(axis='x', rotation=45)
    # Add value labels on bars
    for bar, v in zip(bars, val_mse):
        height = bar.get_height()
        if height < 1:
            label = f'{v:.4f}'
        else:
            label = f'{v:.1f}'
        axes[0].annotate(label, xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # R² comparison
    bars = axes[1].bar(models, val_r2, color=colors)
    axes[1].set_ylabel("Validation R²")
    axes[1].set_title(f"{category} - Validation R² Comparison")
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    # Add value labels
    for bar, v in zip(bars, val_r2):
        height = bar.get_height()
        label = f'{v:.4f}'
        ypos = height if height >= 0 else height - 0.05
        va = 'bottom' if height >= 0 else 'top'
        axes[1].annotate(label, xy=(bar.get_x() + bar.get_width() / 2, ypos),
                        xytext=(0, 3 if height >= 0 else -3), textcoords="offset points",
                        ha='center', va=va, fontsize=9)
    
    plt.tight_layout()
    plot_path = out_dir / f"{category.lower().replace(' ', '_').replace('(', '').replace(')', '')}_overall_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def plot_perparam_comparison(
    results: dict,
    param_names: list,
    dataset_label: str,
    out_path: str,
    metric: str = "r2"
):
    """Create grouped bar plot comparing models per parameter."""
    models = list(results.keys())
    n_params = len(param_names)
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=(max(12, n_params * 0.8), 6))
    
    x = np.arange(n_params)
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    for i, model in enumerate(models):
        key = f"{metric}_per_target"
        values = results[model].get(key)
        if values is None:
            continue
        values = np.array(values)
        if len(values) < n_params:
            values = np.pad(values, (0, n_params - len(values)), constant_values=np.nan)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, color=colors[i])
    
    ax.set_xlabel("Parameter")
    ax.set_ylabel(f"Validation {metric.upper()}")
    ax.set_title(f"Per-Parameter {metric.upper()} Comparison - {dataset_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-parameter Optuna analysis")
    parser.add_argument("--out-dir", type=str, default="data/analysis/optuna_perparam",
                        help="Output directory")
    parser.add_argument("--retrain", action="store_true",
                        help="Re-train models to get per-param metrics (slow)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs if retraining")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training (cuda/cpu)")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    db_dir = Path(__file__).parent.parent / "data" / "optuna"
    
    # Configuration: (db_file, study_name, model_type, dataset_name, category)
    configs = [
        ("optuna_smf.db", "smf_optuna", "smf", "LH", "LH"),
        ("optuna_deepset.db", "deepset_optuna", "deepset", "LH", "LH"),
        ("optuna_slotsetpool.db", "slotsetpool_optuna", "slotsetpool", "LH", "LH"),
        ("optuna_smf_sb28.db", "smf_optuna", "smf", "SB28", "SB28"),
        ("optuna_slotsetpool_sb28.db", "slotsetpool_optuna", "slotsetpool", "SB28", "SB28"),
    ]
    
    # Get param names for each dataset
    lh_params = get_param_names("LH")
    sb28_params = get_param_names("SB28")
    
    print("=" * 80)
    print("LH Parameter Names:", lh_params)
    print("SB28 Parameter Names:", sb28_params[:10], "..." if len(sb28_params) > 10 else "")
    print("=" * 80)
    
    results = {"LH": {}, "SB28": {}}
    
    for db_file, study_name, model_type, dataset_name, category in configs:
        db_path = db_dir / db_file
        if not db_path.exists():
            print(f"Skipping {db_file} (not found)")
            continue
        
        print(f"\nProcessing: {db_file} -> {model_type}")
        
        # Load best hyperparameters
        best = load_optuna_best(str(db_path), study_name)
        print(f"  Best value: {best['value']:.6f}, N trials: {best['n_trials']}")
        print(f"  Best params: {best['params']}")
        
        model_key = model_type.upper()
        if model_type == "slotsetpool":
            model_key = "SlotSetPool"
        elif model_type == "deepset":
            model_key = "DeepSet"
        elif model_type == "smf":
            model_key = "SMF"
        
        # Determine if SMF model (uses different collate)
        use_smf = (model_type == "smf")
        
        if args.retrain:
            # Re-train and get per-parameter metrics
            print(f"  Re-training for {args.epochs} epochs...")
            metrics = train_and_eval_perparam(
                model_type=model_type,
                params=best["params"],
                dataset_name=dataset_name,
                use_smf=use_smf,
                epochs=args.epochs,
                device_str=args.device,
            )
            results[category][model_key] = {
                "val_mse": metrics["val_mse"],
                "val_r2": metrics["val_r2"],
                "train_mse": best["user_attrs"].get("train_mse", 0),
                "train_r2": best["user_attrs"].get("train_r2", 0),
                "params": best["params"],
                "n_trials": best["n_trials"],
                "mse_per_target": metrics["mse_per_target"],
                "r2_per_target": metrics["r2_per_target"],
            }
            print(f"  Val MSE: {metrics['val_mse']:.4f}, Val R²: {metrics['val_r2']:.4f}")
            print(f"  Per-param R²: {[f'{r:.3f}' for r in metrics['r2_per_target'][:6]]}{'...' if len(metrics['r2_per_target']) > 6 else ''}")
        else:
            # Store overall metrics from Optuna
            results[category][model_key] = {
                "val_mse": best["user_attrs"].get("val_mse", best["value"]),
                "val_r2": best["user_attrs"].get("val_r2", 0),
                "train_mse": best["user_attrs"].get("train_mse", 0),
                "train_r2": best["user_attrs"].get("train_r2", 0),
                "params": best["params"],
                "n_trials": best["n_trials"],
                "mse_per_target": None,
                "r2_per_target": None,
            }
    
    # Print summary tables
    print("\n" + "=" * 80)
    print("                    SUMMARY: LH (6 Parameters)")
    print("=" * 80)
    lh_rows = []
    for model, data in results["LH"].items():
        row = {
            "Model": model,
            "Val MSE": f"{data['val_mse']:.6f}",
            "Val R²": f"{data['val_r2']:.4f}",
            "Train MSE": f"{data['train_mse']:.6f}",
            "Train R²": f"{data['train_r2']:.4f}",
            "N Trials": data["n_trials"],
        }
        lh_rows.append(row)
    
    lh_df = pd.DataFrame(lh_rows)
    print(lh_df.to_string(index=False))
    lh_df.to_csv(out_dir / "lh_summary.csv", index=False)
    
    print("\n" + "=" * 80)
    print("                    SUMMARY: SB28 (28 Parameters)")
    print("=" * 80)
    sb28_rows = []
    for model, data in results["SB28"].items():
        row = {
            "Model": model,
            "Val MSE": f"{data['val_mse']:.2f}",
            "Val R²": f"{data['val_r2']:.4f}",
            "Train MSE": f"{data['train_mse']:.2f}",
            "Train R²": f"{data['train_r2']:.4f}",
            "N Trials": data["n_trials"],
        }
        sb28_rows.append(row)
    
    sb28_df = pd.DataFrame(sb28_rows)
    print(sb28_df.to_string(index=False))
    sb28_df.to_csv(out_dir / "sb28_summary.csv", index=False)
    
    # Generate overall comparison plots
    plot_overall_comparison(results["LH"], "LH (6 params)", out_dir)
    plot_overall_comparison(results["SB28"], "SB28 (28 params)", out_dir)
    
    # Generate per-parameter plots if retraining was done
    if args.retrain:
        # Check if any model has per-param data
        has_lh_perparam = any(results["LH"][m].get("r2_per_target") is not None for m in results["LH"])
        has_sb28_perparam = any(results["SB28"][m].get("r2_per_target") is not None for m in results["SB28"])
        
        if has_lh_perparam:
            plot_perparam_comparison(results["LH"], lh_params, "LH (6 params)",
                                     str(out_dir / "lh_perparam_r2.png"), metric="r2")
            plot_perparam_comparison(results["LH"], lh_params, "LH (6 params)",
                                     str(out_dir / "lh_perparam_mse.png"), metric="mse")
        
        if has_sb28_perparam:
            plot_perparam_comparison(results["SB28"], sb28_params, "SB28 (28 params)",
                                     str(out_dir / "sb28_perparam_r2.png"), metric="r2")
            plot_perparam_comparison(results["SB28"], sb28_params, "SB28 (28 params)",
                                     str(out_dir / "sb28_perparam_mse.png"), metric="mse")
        
        # Also create per-parameter summary tables
        if has_lh_perparam:
            perparam_rows = []
            for i, pname in enumerate(lh_params):
                row = {"Parameter": pname}
                for model in results["LH"]:
                    r2_vals = results["LH"][model].get("r2_per_target")
                    if r2_vals and i < len(r2_vals):
                        row[f"{model}_R2"] = f"{r2_vals[i]:.4f}"
                perparam_rows.append(row)
            perparam_df = pd.DataFrame(perparam_rows)
            perparam_df.to_csv(out_dir / "lh_perparam_r2.csv", index=False)
            print(f"\nLH Per-Parameter R²:")
            print(perparam_df.to_string(index=False))
        
        if has_sb28_perparam:
            perparam_rows = []
            for i, pname in enumerate(sb28_params):
                row = {"Parameter": pname}
                for model in results["SB28"]:
                    r2_vals = results["SB28"][model].get("r2_per_target")
                    if r2_vals and i < len(r2_vals):
                        row[f"{model}_R2"] = f"{r2_vals[i]:.4f}"
                perparam_rows.append(row)
            perparam_df = pd.DataFrame(perparam_rows)
            perparam_df.to_csv(out_dir / "sb28_perparam_r2.csv", index=False)
            print(f"\nSB28 Per-Parameter R² (first 10):")
            print(perparam_df.head(10).to_string(index=False))
    else:
        print("\n⚠️  Per-parameter metrics require model retraining.")
        print("   Run with --retrain to train models and get per-parameter R²/MSE.")
    
    # Save full results as JSON
    results_json = {}
    for cat in results:
        results_json[cat] = {}
        for model in results[cat]:
            d = results[cat][model].copy()
            # Convert numpy arrays to lists for JSON
            for k in ["mse_per_target", "r2_per_target"]:
                if d.get(k) is not None and hasattr(d[k], 'tolist'):
                    d[k] = d[k].tolist()
            results_json[cat][model] = d
    
    with open(out_dir / "optuna_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: {out_dir / 'optuna_results.json'}")
    
    # Print best model per category
    print("\n" + "=" * 80)
    print("                         BEST MODELS")
    print("=" * 80)
    
    for cat in ["LH", "SB28"]:
        if results[cat]:
            best_model = min(results[cat].keys(), key=lambda m: results[cat][m]["val_mse"])
            best_mse = results[cat][best_model]["val_mse"]
            best_r2 = results[cat][best_model]["val_r2"]
            print(f"{cat}: {best_model} (Val MSE: {best_mse:.4f}, Val R²: {best_r2:.4f})")


if __name__ == "__main__":
    main()
