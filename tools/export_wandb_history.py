#!/usr/bin/env python3
"""
Export local W&B run history from .wandb files to JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from typing import Any, Dict, Iterable, Optional, Set

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def _parse_args_list(args_list: Iterable[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    i = 0
    args_list = list(args_list)
    while i < len(args_list):
        raw = args_list[i]
        if raw.startswith("--") and "=" in raw:
            flag, val = raw.split("=", 1)
            args_list = args_list[:i] + [flag, val] + args_list[i + 1 :]
            raw = flag
        if raw == "--wandb-project" and i + 1 < len(args_list):
            parsed["wandb_project"] = args_list[i + 1]
            i += 2
            continue
        i += 1
    return parsed


def _history_key(item: wandb_internal_pb2.HistoryItem) -> Optional[str]:
    key = item.key or ""
    if item.nested_key:
        if key:
            key = key + "/" + "/".join(item.nested_key)
        else:
            key = "/".join(item.nested_key)
    key = key.lstrip("/")
    return key or None


def _parse_value_json(raw: str) -> Any:
    try:
        return json.loads(raw, parse_constant=lambda _k: float("nan"))
    except Exception:
        return raw


def _iter_history_rows(wandb_path: str) -> Iterable[Dict[str, Any]]:
    ds = DataStore()
    ds.open_for_scan(wandb_path)
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if not rec.HasField("history"):
            continue
        row: Dict[str, Any] = {}
        for item in rec.history.item:
            key = _history_key(item)
            if key is None:
                continue
            row[key] = _parse_value_json(item.value_json)
        if rec.history.step and "_step" not in row:
            row["_step"] = rec.history.step
        if row:
            yield row


def _find_wandb_file(run_dir: str) -> Optional[str]:
    for name in os.listdir(run_dir):
        if name.startswith("run-") and name.endswith(".wandb"):
            return os.path.join(run_dir, name)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-dir", type=str, default="run/wandb")
    parser.add_argument("--out-name", type=str, default="wandb-history.jsonl")
    parser.add_argument("--projects", type=str, default=None,
                        help="Comma-separated list of wandb project names to include")
    parser.add_argument("--metric-prefix", type=str, default="val/r2/,val_r2_",
                        help="Comma-separated prefixes to keep (default: val/r2/,val_r2_)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_filter = None
    if args.projects:
        project_filter = {p.strip() for p in args.projects.split(",") if p.strip()}
    prefixes = [p.strip() for p in args.metric_prefix.split(",") if p.strip()]
    keep_keys = {"epoch", "global_step", "_step", "trial"}

    run_dirs: Optional[Set[str]] = None
    if project_filter:
        pattern = "|".join(re.escape(p) for p in sorted(project_filter))
        cmd = [
            "rg",
            "-l",
            pattern,
            os.path.join(args.wandb_dir, "run-*/files/config.yaml"),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode in (0, 1):
                run_dirs = set()
                for line in res.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    run_dirs.add(os.path.dirname(os.path.dirname(line)))
        except Exception:
            run_dirs = None

    exported = 0
    skipped = 0
    entries = sorted(run_dirs) if run_dirs is not None else [
        os.path.join(args.wandb_dir, d)
        for d in os.listdir(args.wandb_dir)
        if d.startswith("run-")
    ]
    for run_dir in entries:
        meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")
        if project_filter and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                parsed = _parse_args_list(meta.get("args", []))
                project = parsed.get("wandb_project")
                if project not in project_filter:
                    continue
            except Exception:
                continue

        out_path = os.path.join(run_dir, "files", args.out_name)
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        wandb_path = _find_wandb_file(run_dir)
        if wandb_path is None:
            continue

        os.makedirs(os.path.join(run_dir, "files"), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in _iter_history_rows(wandb_path):
                filtered = {}
                for k, v in row.items():
                    if k in keep_keys or any(k.startswith(p) for p in prefixes):
                        filtered[k] = v
                if not any(k.startswith(p) for p in prefixes for k in filtered.keys()):
                    continue
                f.write(json.dumps(filtered) + "\n")
        exported += 1

    print(f"Exported histories: {exported}, skipped (existing): {skipped}")


if __name__ == "__main__":
    main()
