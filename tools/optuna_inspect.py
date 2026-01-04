"""Standalone Optuna inspection and dashboard helper.

Usage examples:
  # Print best trial from a sqlite storage
  python tools/optuna_inspect.py --storage sqlite:///data/optuna/optuna_deepset.db --study deepset_optuna --show-best

  # Export trials to CSV and HTML plot
  python tools/optuna_inspect.py --storage sqlite:///data/optuna/optuna_deepset.db --study deepset_optuna \
      --out-dir data/optuna/inspect_out --trials-csv --plot-history --plot-param-importances

SSH tunnel to use optuna-dashboard on a remote cluster:
  1) On your local machine run:
       ssh -L 8080:localhost:8080 user@cluster
  2) On the cluster (SSH session) run:
       optuna-dashboard sqlite:////full/path/to/data/optuna/optuna_deepset.db --study-name deepset_optuna --host 0.0.0.0 --port 8080
  3) Open http://localhost:8080 on your laptop.

This script provides programmatic exports and quick mem-log parsing utilities.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import pandas as pd
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate


def load_study(storage: str, study_name: str):
    return optuna.load_study(study_name=study_name, storage=storage)


def save_trials_df(study, out_dir: Path):
    df = study.trials_dataframe()
    csv_path = out_dir / f"{study.study_name}_trials.csv"
    json_path = out_dir / f"{study.study_name}_trials.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")
    print(f"Wrote {csv_path} and {json_path}")


def save_plots(study, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = plot_optimization_history(study)
    hist_path = out_dir / f"{study.study_name}_opt_hist.html"
    hist.write_html(str(hist_path))
    print(f"Wrote {hist_path}")

    imp = plot_param_importances(study)
    imp_path = out_dir / f"{study.study_name}_param_importances.html"
    imp.write_html(str(imp_path))
    print(f"Wrote {imp_path}")

    try:
        pc = plot_parallel_coordinate(study)
        pc_path = out_dir / f"{study.study_name}_parallel_coordinate.html"
        pc.write_html(str(pc_path))
        print(f"Wrote {pc_path}")
    except Exception:
        pass


def print_best(study):
    best = study.best_trial
    print("Best trial:")
    print("  value:", best.value)
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


def parse_mem_log(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    rows = []
    for L in lines:
        # naive parse: timestamp [mem] key=val ...
        try:
            ts, rest = L.split(" [mem] ", 1)
        except ValueError:
            continue
        d = {"ts": ts}
        for tok in rest.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                # try cast numeric
                try:
                    d[k] = float(v)
                except Exception:
                    d[k] = v
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--storage", required=True, help="Optuna storage URL, e.g. sqlite:///data/optuna/optuna_deepset.db")
    p.add_argument("--study", required=True, help="Study name")
    p.add_argument("--out-dir", default="data/optuna/inspect_out")
    p.add_argument("--show-best", action="store_true")
    p.add_argument("--trials-csv", action="store_true")
    p.add_argument("--plot-history", action="store_true")
    p.add_argument("--plot-param-importances", action="store_true")
    p.add_argument("--mem-log", help="Path to mem log (e.g. data/optuna/mem_deepset.log)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    study = load_study(args.storage, args.study)

    if args.show_best:
        print_best(study)

    if args.trials_csv:
        save_trials_df(study, out_dir)

    if args.plot_history or args.plot_param_importances:
        save_plots(study, out_dir)

    if args.mem_log:
        df = parse_mem_log(Path(args.mem_log))
        if df.empty:
            print("No rows parsed from mem log")
        else:
            print(df.sort_values("cpu_rss_mb", ascending=False).head(10))


if __name__ == "__main__":
    main()
