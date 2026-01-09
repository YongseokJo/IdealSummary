#!/usr/bin/env python3
"""
Plot comparison of Optuna results across models for LH and SB28.
Uses overall metrics from Optuna (no retraining needed).
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _extract_per_param_r2(model_data, param_names=None):
    r2 = model_data.get("r2_per_target")
    if r2 is None:
        return {}
    if isinstance(r2, dict):
        return {str(k): float(v) for k, v in r2.items()}
    if isinstance(r2, list):
        if param_names is None:
            param_names = model_data.get("param_names")
        if param_names is None:
            param_names = [f"param_{i}" for i in range(len(r2))]
        return {str(name): float(val) for name, val in zip(param_names, r2)}
    return {}


def _plot_best_per_param(out_path, title, per_model, colors):
    # per_model: {model_name: {param_name: r2}}
    if not per_model:
        return False
    # collect parameters in a stable order
    ordered_params = []
    seen = set()
    for model_map in per_model.values():
        for pname in model_map.keys():
            if pname not in seen:
                seen.add(pname)
                ordered_params.append(pname)
    if not ordered_params:
        return False

    best_vals = []
    best_models = []
    for pname in ordered_params:
        best_model = None
        best_val = None
        for model_name, model_map in per_model.items():
            if pname not in model_map:
                continue
            val = model_map[pname]
            if best_val is None or val > best_val:
                best_val = val
                best_model = model_name
        if best_val is None:
            best_val = float("nan")
            best_model = "N/A"
        best_vals.append(best_val)
        best_models.append(best_model)

    x = np.arange(len(ordered_params))
    fig_w = max(8.0, 0.5 * len(ordered_params))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    bar_colors = [colors.get(m, "#7f8c8d") for m in best_models]
    bars = ax.bar(x, best_vals, color=bar_colors, edgecolor="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_params, rotation=55, ha="right", fontsize=10)
    ax.set_ylabel("Validation R²", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)

    for bar, model_name, val in zip(bars, best_models, best_vals):
        if np.isnan(val):
            label = f"{model_name}"
        else:
            label = f"{model_name}\n{val:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                label, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", type=str, 
                        default="data/analysis/optuna_perparam/optuna_results.json")
    parser.add_argument("--out-dir", type=str, default="data/analysis/optuna_perparam")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.results_json) as f:
        results = json.load(f)
    
    colors = {
        "SMF": "#2ecc71",
        "DeepSet": "#3498db",
        "SlotSetPool": "#9b59b6",
    }

    # ===== LH Dataset =====
    lh_data = results.get("LH", {})
    if lh_data:
        models = list(lh_data.keys())
        val_r2 = [lh_data[m]["val_r2"] for m in models]
        val_mse = [lh_data[m]["val_mse"] for m in models]
        n_trials = [lh_data[m]["n_trials"] for m in models]
        
        # Bar plot for LH - R²
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.arange(len(models))
        
        # R² plot
        ax = axes[0]
        bars = ax.bar(x, val_r2, color=[colors.get(m, "#7f8c8d") for m in models],
                      edgecolor='black', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylabel("Validation R²", fontsize=12)
        ax.set_title("LH Dataset (6 params) - Validation R²", fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, max(val_r2) * 1.15 if max(val_r2) > 0 else 1)
        
        # Add value labels
        for bar, v, n in zip(bars, val_r2, n_trials):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}\n({n} trials)', ha='center', va='bottom', fontsize=10)
        
        # MSE plot
        ax = axes[1]
        bars = ax.bar(x, val_mse, color=[colors.get(m, "#7f8c8d") for m in models],
                      edgecolor='black', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylabel("Validation MSE", fontsize=12)
        ax.set_title("LH Dataset (6 params) - Validation MSE", fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(val_mse) * 1.15)
        
        for bar, v in zip(bars, val_mse):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(out_dir / "lh_overall_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_dir / 'lh_overall_comparison.png'}")

        # Best model per parameter (LH)
        lh_param_names = lh_data.get("param_names")
        per_model = {}
        for m in models:
            param_map = _extract_per_param_r2(lh_data[m], param_names=lh_param_names)
            if param_map:
                per_model[m] = param_map
        out_path = out_dir / "lh_best_per_param.png"
        if _plot_best_per_param(out_path, "LH Dataset (6 params) - Best model per parameter", per_model, colors):
            print(f"Saved: {out_path}")
    
    # ===== SB28 Dataset (without DeepSet) =====
    sb28_data = results.get("SB28", {})
    if sb28_data:
        models = list(sb28_data.keys())
        val_r2 = [sb28_data[m]["val_r2"] for m in models]
        val_mse = [sb28_data[m]["val_mse"] for m in models]
        n_trials = [sb28_data[m]["n_trials"] for m in models]
        
        # For SB28, R² values are extremely negative, so let's plot MSE and log-scale
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.arange(len(models))
        
        # MSE plot
        ax = axes[0]
        bars = ax.bar(x, val_mse, color=[colors.get(m, "#7f8c8d") for m in models],
                      edgecolor='black', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylabel("Validation MSE", fontsize=12)
        ax.set_title("SB28 Dataset (28 params) - Validation MSE\n(DeepSet SB28 not available)", 
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(val_mse) * 1.15)
        
        for bar, v, n in zip(bars, val_mse, n_trials):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{v:.0f}\n({n} trials)', ha='center', va='bottom', fontsize=10)
        
        # R² plot (will be very negative)
        ax = axes[1]
        # Clip for visualization
        r2_display = [max(r, -2) for r in val_r2]  # Clip at -2 for display
        bars = ax.bar(x, r2_display, color=[colors.get(m, "#7f8c8d") for m in models],
                      edgecolor='black', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylabel("Validation R² (clipped)", fontsize=12)
        ax.set_title("SB28 Dataset (28 params) - Validation R²\n(Very negative = poor fit)", 
                     fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        for bar, v in zip(bars, val_r2):
            label = f'{v:.0f}' if abs(v) < 1000 else f'{v:.2e}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.1,
                    label, ha='center', va='top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(out_dir / "sb28_overall_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_dir / 'sb28_overall_comparison.png'}")

        # Best model per parameter (SB28)
        sb28_param_names = sb28_data.get("param_names")
        per_model = {}
        for m in models:
            param_map = _extract_per_param_r2(sb28_data[m], param_names=sb28_param_names)
            if param_map:
                per_model[m] = param_map
        out_path = out_dir / "sb28_best_per_param.png"
        if _plot_best_per_param(out_path, "SB28 Dataset (28 params) - Best model per parameter", per_model, colors):
            print(f"Saved: {out_path}")
    
    # ===== Combined summary table =====
    print("\n" + "="*70)
    print("OPTUNA RESULTS SUMMARY")
    print("="*70)
    
    print("\nLH Dataset (6 cosmological parameters):")
    print("-"*50)
    print(f"{'Model':<15} {'Val R²':>10} {'Val MSE':>12} {'N Trials':>10}")
    print("-"*50)
    for m in lh_data:
        print(f"{m:<15} {lh_data[m]['val_r2']:>10.4f} {lh_data[m]['val_mse']:>12.4f} {lh_data[m]['n_trials']:>10}")
    
    print("\nSB28 Dataset (28 cosmological parameters):")
    print("-"*50)
    print(f"{'Model':<15} {'Val R²':>15} {'Val MSE':>12} {'N Trials':>10}")
    print("-"*50)
    for m in sb28_data:
        r2_str = f"{sb28_data[m]['val_r2']:.2e}"
        print(f"{m:<15} {r2_str:>15} {sb28_data[m]['val_mse']:>12.1f} {sb28_data[m]['n_trials']:>10}")
    
    print("\nNote: DeepSet SB28 database not available.")
    print("="*70)


if __name__ == "__main__":
    main()
