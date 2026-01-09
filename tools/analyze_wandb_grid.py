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


def _extract_per_param_val_r2(summary: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, val in summary.items():
        if key.startswith("val/r2/"):
            label = key.split("/", 2)[2]
            try:
                out[label] = float(val)
            except Exception:
                continue
    if out:
        return out
    for key, val in summary.items():
        if key.startswith("val_r2_"):
            label = key[len("val_r2_") :]
            try:
                out[label] = float(val)
            except Exception:
                continue
    return out


def _history_param_keys(history_path: str) -> List[str]:
    keys = set()
    try:
        with open(history_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                for key in row.keys():
                    if key.startswith("val/r2/") or key.startswith("val_r2_"):
                        keys.add(key)
    except Exception:
        return []
    return sorted(keys)


def _best_per_param_from_history(history_path: str) -> Dict[str, float]:
    best: Dict[str, float] = {}
    keys = _history_param_keys(history_path)
    for key in keys:
        series = _parse_epoch_series(history_path, metric_key=key)
        best_epoch, best_val, _last_epoch, _last_val = _best_from_series(series)
        if best_epoch is None or best_val is None:
            continue
        if key.startswith("val/r2/"):
            label = key.split("/", 2)[2]
        else:
            label = key[len("val_r2_") :]
        best[label] = float(best_val)
    return best


def _collect_best_per_param(
    wandb_dir: str,
    project_names: List[str],
) -> Dict[str, Dict[str, float]]:
    best: Dict[str, Dict[str, float]] = {}
    history_used = 0
    summary_used = 0
    for entry in sorted(os.listdir(wandb_dir)):
        if not entry.startswith("run-"):
            continue
        run_dir = os.path.join(wandb_dir, entry)
        summary_path = os.path.join(run_dir, "files", "wandb-summary.json")
        meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")
        history_candidates = [
            os.path.join(run_dir, "files", "wandb-history.jsonl"),
            os.path.join(run_dir, "files", "wandb-history.json"),
            os.path.join(run_dir, "files", "history.jsonl"),
        ]

        summary = _load_json(summary_path)
        meta = _load_json(meta_path)
        if summary is None or meta is None:
            continue

        parsed_args = _parse_args_list(meta.get("args", []))
        project = parsed_args.get("wandb_project")
        if project not in project_names:
            continue

        per_param = {}
        for history_path in history_candidates:
            if os.path.exists(history_path):
                per_param = _best_per_param_from_history(history_path)
                if per_param:
                    history_used += 1
                    break
        if not per_param:
            per_param = _extract_per_param_val_r2(summary)
            if per_param:
                summary_used += 1
        if not per_param:
            continue
        proj_best = best.setdefault(project, {})
        for label, val in per_param.items():
            prev = proj_best.get(label)
            if prev is None or val > prev:
                proj_best[label] = float(val)
    if history_used or summary_used:
        print(f"Best-per-param source: history={history_used}, summary_fallback={summary_used}")
    return best


def _ordered_param_labels(series_list: List[Dict[str, float]]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for series in series_list:
        for label in series.keys():
            if label not in seen:
                seen.add(label)
                ordered.append(label)
    return ordered


def _plot_per_param_bars(
    out_path: str,
    title: str,
    param_labels: List[str],
    series: List[Tuple[str, List[float]]],
    colors: Optional[Dict[str, str]] = None,
) -> None:
    n_models = max(1, len(series))
    x = np.arange(len(param_labels))
    width = 0.8 / n_models
    fig_w = max(8.0, 0.45 * len(param_labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    for i, (label, values) in enumerate(series):
        offset = -0.4 + (i + 0.5) * width
        color = colors.get(label) if colors else None
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, param_labels, rotation=55, ha="right")
    ax.set_ylabel("val_r2")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    vals = [v for _label, values in series for v in values if not math.isnan(v)]
    if vals:
        vmin = min(vals)
        vmax = max(vals)
        pad = 0.05 * (vmax - vmin) if vmax != vmin else max(0.1, abs(vmax) * 0.1)
        ax.set_ylim(vmin - pad, vmax + pad)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _best_per_param_plots(wandb_dir: str, out_dir: str) -> None:
    lh_projects = [
        ("optuna_test_smf", "smf"),
        ("optuna_test_deepset", "deepset"),
        ("optuna_test_slotsetpool", "slotsetpool"),
    ]
    sb28_projects = [
        ("optuna_test_smf_sb28", "smf_sb28"),
        ("optuna_test_slotsetpool_sb28", "slotsetpool_sb28"),
    ]
    model_colors = {
        "smf": "#2ecc71",
        "deepset": "#3498db",
        "slotsetpool": "#9b59b6",
        "smf_sb28": "#2ecc71",
        "slotsetpool_sb28": "#9b59b6",
    }
    targets = [p for p, _ in lh_projects + sb28_projects]
    best = _collect_best_per_param(wandb_dir, targets)

    os.makedirs(out_dir, exist_ok=True)

    def _build_series(items: List[Tuple[str, str]]) -> Tuple[List[str], List[Tuple[str, List[float]]]]:
        per_param_list = []
        series: List[Tuple[str, List[float]]] = []
        for project, model_label in items:
            per_param = best.get(project)
            if not per_param:
                print(f"No per-parameter metrics for {project}")
                continue
            per_param_list.append(per_param)
            series.append((model_label, []))
        if not per_param_list:
            return [], []
        param_labels = _ordered_param_labels(per_param_list)
        for idx, (model_label, _values) in enumerate(series):
            per_param = best.get(items[idx][0], {})
            values = [per_param.get(p, float("nan")) for p in param_labels]
            series[idx] = (model_label, values)
        return param_labels, series

    lh_params, lh_series = _build_series(lh_projects)
    if lh_params and lh_series:
        out_path = os.path.join(out_dir, "best_per_param_LH.png")
        _plot_per_param_bars(
            out_path,
            "Best per-parameter val_r2 by model (LH)",
            lh_params,
            lh_series,
            colors=model_colors,
        )
        print(f"Wrote {out_path}")

    sb_params, sb_series = _build_series(sb28_projects)
    if sb_params and sb_series:
        out_path = os.path.join(out_dir, "best_per_param_SB28.png")
        _plot_per_param_bars(
            out_path,
            "Best per-parameter val_r2 by model (SB28)",
            sb_params,
            sb_series,
            colors=model_colors,
        )
        print(f"Wrote {out_path}")


def _parse_epoch_series(path: str, metric_key: Optional[str] = None) -> List[Tuple[int, float]]:
    if not os.path.exists(path):
        return []
    series: List[Tuple[int, float]] = []
    if metric_key is not None and (path.endswith(".jsonl") or path.endswith(".json")):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if metric_key not in row:
                        continue
                    try:
                        val = float(row[metric_key])
                    except Exception:
                        continue
                    step = row.get("epoch")
                    if step is None:
                        step = row.get("global_step")
                    if step is None:
                        step = row.get("_step")
                    if step is None:
                        step = idx
                    try:
                        step_i = int(step)
                    except Exception:
                        step_i = idx
                    series.append((step_i, val))
        except Exception:
            return []
        return series

    epoch_re = re.compile(r"Epoch\s+(\d+).*?val_r2=([\-0-9.eE]+)")
    gen_re = re.compile(r"\[Gen\s+(\d+)\].*?val_r2=([\-0-9.eE]+)")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
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


def _plot_best_vs_last(
    out_path: str,
    title: str,
    param_labels: List[str],
    best_vals: List[float],
    last_vals: List[float],
) -> None:
    x = np.arange(len(param_labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6.5, 0.5 * len(param_labels)), 4.5))
    ax.bar(x - width / 2, best_vals, width, label="best_val_r2")
    ax.bar(x + width / 2, last_vals, width, label="last_val_r2")
    ax.set_xticks(x, param_labels, rotation=45, ha="right")
    ax.set_ylabel("val_r2")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-dir", type=str, default="wandb", help="Local wandb directory")
    parser.add_argument("--out-dir", type=str, default="run/grid_analysis", help="Output directory for reports/plots")
    parser.add_argument("--program", type=str, default="train.py", help="Filter runs by program basename")
    parser.add_argument("--project", type=str, default=None, help="Filter by single wandb project name (from args)")
    parser.add_argument("--projects", type=str, default=None,
                        help="Comma-separated list of wandb project names to include")
    parser.add_argument("--best-per-param", action="store_true",
                        help="Plot per-parameter val_r2 bars for best runs in LH/SB28 optuna_test projects")
    parser.add_argument("--improve-epochs", type=int, default=3, help="Epoch window to mark 'still_improving'")
    parser.add_argument("--improve-eps", type=float, default=0.002, help="Tolerance from best val_r2 for improving")
    parser.add_argument("--overfit-drop", type=float, default=0.01, help="Drop from best val_r2 to flag overfit")
    args = parser.parse_args()

    if args.best_per_param:
        _best_per_param_plots(args.wandb_dir, args.out_dir)
        return

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
        project = parsed_args.get("wandb_project")
        if args.projects:
            allowed = {p.strip() for p in args.projects.split(",") if p.strip()}
            if project not in allowed:
                continue
        elif args.project and project != args.project:
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
            "param_label", "structure", "lr", "val_r2_best", "best_epoch",
            "last_epoch", "val_r2_last", "target_epochs", "status",
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
                "rank", "structure", "lr", "val_r2_best", "best_epoch",
                "last_epoch", "val_r2_last", "target_epochs", "status", "run_id",
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

    # Best vs last per parameter (including "all" if present)
    param_labels = sorted(grouped.keys(), key=lambda k: (k != "all", k))
    best_vals = []
    last_vals = []
    for label in param_labels:
        entries = grouped[label]
        best_run = max(entries, key=lambda r: r["val_r2"])
        best_vals.append(float(best_run["val_r2"]))
        last_vals.append(float(best_run["last_val_r2"]) if best_run["last_val_r2"] is not None else float("nan"))

    if param_labels:
        plot_path = os.path.join(args.out_dir, "best_vs_last_per_param.png")
        _plot_best_vs_last(plot_path, "Best vs last val_r2 per parameter", param_labels, best_vals, last_vals)
        print(f"Wrote {plot_path}")

    print(f"Summary written to {summary_csv}")


if __name__ == "__main__":
    main()
