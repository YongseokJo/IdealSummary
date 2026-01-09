#!/usr/bin/env python3
"""
Analyze local W&B runs for grid-search results and visualize best val_r2 rankings.

Example:
  python tools/analyze_wandb_grid.py --wandb-dir wandb --out-dir run/grid_analysis \
    --program train.py --project SB28_SMF_grid
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())


def _parse_args_list(args_list: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    parsed["param_keys"] = None

    def _set_from_kv(flag: str, key: str, i: int) -> int:
        if i + 1 < len(args_list):
            parsed[key] = args_list[i + 1]
            return i + 2
        return i + 1

    i = 0
    while i < len(args_list):
        raw = args_list[i]
        if raw.startswith("--") and "=" in raw:
            flag, val = raw.split("=", 1)
            args_list = args_list[:i] + [flag, val] + args_list[i + 1:]
            raw = flag

        if raw == "--param-keys":
            i += 1
            keys = []
            while i < len(args_list) and not args_list[i].startswith("--"):
                keys.append(args_list[i])
                i += 1
            parsed["param_keys"] = keys
            continue

        if raw == "--lr":
            i = _set_from_kv(raw, "lr", i)
            continue
        if raw == "--mlp-structure":
            i = _set_from_kv(raw, "mlp_structure", i)
            continue
        if raw == "--wandb-project":
            i = _set_from_kv(raw, "wandb_project", i)
            continue
        if raw == "--wandb-run-name":
            i = _set_from_kv(raw, "wandb_run_name", i)
            continue
        if raw == "--model-type":
            i = _set_from_kv(raw, "model_type", i)
            continue
        if raw == "--h5-path":
            i = _set_from_kv(raw, "h5_path", i)
            continue
        if raw == "--epochs":
            i = _set_from_kv(raw, "epochs", i)
            continue
        if raw in ("--use-smf", "--wandb", "--pin-memory", "--persistent-workers"):
            parsed[raw.lstrip("-").replace("-", "_")] = True
            i += 1
            continue

        i += 1

    return parsed


def _summary_val_r2(summary: Dict[str, Any]) -> Optional[float]:
    for key in ("val_r2", "val/r2", "mlp/val_r2", "evo/val_r2"):
        if key in summary:
            try:
                return float(summary[key])
            except Exception:
                return None
    for key, val in summary.items():
        if key.endswith("val_r2"):
            try:
                return float(val)
            except Exception:
                return None
    return None


def _parse_epoch_series(output_log_path: str) -> List[Tuple[int, float]]:
    if not os.path.exists(output_log_path):
        return []
    series: List[Tuple[int, float]] = []
    epoch_re = re.compile(r"Epoch\s+(\d+).*?val_r2=([\-0-9.eE]+)")
    gen_re = re.compile(r"\[Gen\s+(\d+)\].*?val_r2=([\-0-9.eE]+)")
    try:
        with open(output_log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = epoch_re.search(line)
                if m:
                    series.append((int(m.group(1)), float(m.group(2))))
                    continue
                m = gen_re.search(line)
                if m:
                    series.append((int(m.group(1)), float(m.group(2))))
    except Exception:
        return []
    return series


def _best_from_series(series: List[Tuple[int, float]]) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[float]]:
    if not series:
        return None, None, None, None
    series_sorted = sorted(series, key=lambda x: x[0])
    last_epoch, last_val = series_sorted[-1]
    best_epoch, best_val = max(series_sorted, key=lambda x: x[1])
    return best_epoch, best_val, last_epoch, last_val


def _status_from_series(
    best_epoch: Optional[int],
    best_val: Optional[float],
    last_epoch: Optional[int],
    last_val: Optional[float],
    target_epochs: Optional[int],
    improve_epochs: int,
    improve_eps: float,
    overfit_drop: float,
) -> str:
    if best_epoch is None or best_val is None or last_epoch is None or last_val is None:
        return "unknown"
    if target_epochs is not None:
        try:
            if last_epoch < int(target_epochs):
                return "incomplete"
        except Exception:
            pass
    if last_epoch - best_epoch <= improve_epochs and last_val >= best_val - improve_eps:
        return "still_improving"
    if last_val <= best_val - overfit_drop:
        return "overfit"
    return "converged"


def _parse_lr(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _structure_order(structures: List[str]) -> List[str]:
    preferred = ["shallow", "intermediate", "deep"]
    ordered = [s for s in preferred if s in structures]
    leftovers = sorted([s for s in structures if s not in preferred])
    return ordered + leftovers


def _plot_heatmap(
    out_path: str,
    title: str,
    structures: List[str],
    lrs: List[float],
    data: Dict[Tuple[str, float], Tuple[float, Optional[int]]],
) -> None:
    struct_order = _structure_order(structures)
    lr_order = sorted(lrs)

    matrix = np.full((len(struct_order), len(lr_order)), np.nan, dtype=float)
    for i, s in enumerate(struct_order):
        for j, lr in enumerate(lr_order):
            val = data.get((s, lr))
            if val is not None:
                matrix[i, j] = val[0]

    fig, ax = plt.subplots(figsize=(1.8 + 1.2 * len(lr_order), 1.6 + 0.6 * len(struct_order)))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(lr_order)), [f"{lr:.0e}" for lr in lr_order])
    ax.set_yticks(range(len(struct_order)), struct_order)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("structure")
    plt.colorbar(im, ax=ax, label="val_r2")

    for i in range(len(struct_order)):
        for j in range(len(lr_order)):
            val = matrix[i, j]
            if not math.isnan(val):
                epoch = None
                entry = data.get((struct_order[i], lr_order[j]))
                if entry is not None:
                    epoch = entry[1]
                label = f"{val:.3f}" if epoch is None else f"{val:.3f}\n@{epoch}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-dir", type=str, default="wandb", help="Local wandb directory")
    parser.add_argument("--out-dir", type=str, default="run/grid_analysis", help="Output directory for reports/plots")
    parser.add_argument("--program", type=str, default="train.py", help="Filter runs by program basename")
    parser.add_argument("--project", type=str, default=None, help="Filter by wandb project name (from args)")
    parser.add_argument("--improve-epochs", type=int, default=3, help="Epoch window to mark 'still_improving'")
    parser.add_argument("--improve-eps", type=float, default=0.002, help="Tolerance from best val_r2 for improving")
    parser.add_argument("--overfit-drop", type=float, default=0.01, help="Drop from best val_r2 to flag overfit")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for entry in sorted(os.listdir(args.wandb_dir)):
        if not entry.startswith("run-"):
            continue
        run_dir = os.path.join(args.wandb_dir, entry)
        summary_path = os.path.join(run_dir, "files", "wandb-summary.json")
        meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")

        summary = _load_json(summary_path)
        meta = _load_json(meta_path)
        if summary is None or meta is None:
            continue

        program = os.path.basename(meta.get("program", ""))
        if args.program and program != args.program:
            continue

        parsed_args = _parse_args_list(meta.get("args", []))
        if args.project and parsed_args.get("wandb_project") != args.project:
            continue

        output_log = os.path.join(run_dir, "files", "output.log")
        series = _parse_epoch_series(output_log)
        best_epoch, best_val, last_epoch, last_val = _best_from_series(series)
        if best_val is None:
            best_val = _summary_val_r2(summary)
        if best_val is None:
            continue

        lr = _parse_lr(parsed_args.get("lr"))
        structure = parsed_args.get("mlp_structure", None)
        target_epochs = parsed_args.get("epochs")

        param_keys = parsed_args.get("param_keys")
        if not param_keys:
            param_label = "all"
        elif len(param_keys) == 1:
            param_label = str(param_keys[0])
        else:
            param_label = "all"

        run_id = entry.split("-")[-1]
        status = _status_from_series(
            best_epoch, best_val, last_epoch, last_val, target_epochs,
            args.improve_epochs, args.improve_eps, args.overfit_drop,
        )
        runs.append({
            "run_dir": run_dir,
            "run_id": run_id,
            "param_label": param_label,
            "param_keys": param_keys,
            "lr": lr,
            "structure": structure,
            "val_r2": float(best_val),
            "best_epoch": best_epoch,
            "last_epoch": last_epoch,
            "last_val_r2": last_val,
            "target_epochs": target_epochs,
            "status": status,
            "wandb_project": parsed_args.get("wandb_project"),
            "wandb_run_name": parsed_args.get("wandb_run_name"),
        })

    if not runs:
        print("No matching runs found.")
        return

    # Group by parameter label
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["param_label"], []).append(run)

    # Combined summary CSV
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "param_label", "structure", "lr", "val_r2", "best_epoch",
            "last_epoch", "last_val_r2", "target_epochs", "status",
            "run_id", "wandb_project", "wandb_run_name",
        ])
        for run in sorted(runs, key=lambda r: (r["param_label"], -(r["val_r2"] or 0.0))):
            writer.writerow([
                run["param_label"],
                run["structure"],
                run["lr"],
                run["val_r2"],
                run["best_epoch"],
                run["last_epoch"],
                run["last_val_r2"],
                run["target_epochs"],
                run["status"],
                run["run_id"],
                run.get("wandb_project"),
                run.get("wandb_run_name"),
            ])

    # Per-parameter rankings + plots
    for param_label, entries in grouped.items():
        best_by_combo: Dict[Tuple[str, float], Dict[str, Any]] = {}
        for run in entries:
            if run["structure"] is None or run["lr"] is None:
                continue
            key = (run["structure"], run["lr"])
            prev = best_by_combo.get(key)
            if prev is None or run["val_r2"] > prev["val_r2"]:
                best_by_combo[key] = run

        if not best_by_combo:
            continue

        ranking = sorted(best_by_combo.values(), key=lambda r: r["val_r2"], reverse=True)
        label_safe = _sanitize_label(param_label)
        rank_csv = os.path.join(args.out_dir, f"param_{label_safe}_ranking.csv")
        with open(rank_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "structure", "lr", "val_r2", "best_epoch",
                "last_epoch", "last_val_r2", "target_epochs", "status", "run_id",
            ])
            for idx, run in enumerate(ranking, start=1):
                writer.writerow([
                    idx, run["structure"], run["lr"], run["val_r2"], run["best_epoch"],
                    run["last_epoch"], run["last_val_r2"], run["target_epochs"], run["status"],
                    run["run_id"],
                ])

        structures = sorted({r["structure"] for r in ranking})
        lrs = sorted({r["lr"] for r in ranking})
        data = {(r["structure"], r["lr"]): (r["val_r2"], r["best_epoch"]) for r in ranking}
        plot_path = os.path.join(args.out_dir, f"param_{label_safe}_heatmap.png")
        _plot_heatmap(plot_path, f"val_r2 grid - {param_label}", structures, lrs, data)

        print(f"Wrote {rank_csv} and {plot_path}")

    print(f"Summary written to {summary_csv}")


if __name__ == "__main__":
    main()
