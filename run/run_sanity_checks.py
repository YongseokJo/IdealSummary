#!/usr/bin/env python3
"""Sanity checks for mass cuts and masking (raw + SMF).

Produces:
- SMF plots with/without masking across multiple cut scenarios.
- Raw data counts/quantiles before/after cuts/masking.
"""
import argparse
import json
import os
import sys
from typing import List, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def stellar_mass_function(masses, bins: int, mass_range: Tuple[float, float], box_size: float):
    m = np.asarray(masses, dtype=np.float64)
    m = np.where(m <= 0, 1e6, m)  # avoid log10(0)
    logm = np.log10(m)

    hist, edges = np.histogram(logm, bins=bins, range=mass_range)
    dM = edges[1] - edges[0]
    centers = 0.5 * (edges[1:] + edges[:-1])
    phi = hist / dM / (box_size ** 3)
    return phi, centers


def smf_centers(bins: int, mass_range: Tuple[float, float]):
    edges = np.linspace(mass_range[0], mass_range[1], bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


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
        raise ValueError("Masking expects 1D masses per simulation")
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


def parse_cut_set(spec: str):
    spec = (spec or "").strip()
    if spec.lower() in ("none", "", "no", "null"):
        return "no_cuts", []
    cuts = []
    for token in spec.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = [p.strip() for p in token.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Bad cut spec '{token}'. Use Field:min:max")
        field, lo_s, hi_s = parts
        lo = None if lo_s.lower() in ("none", "") else float(lo_s)
        hi = None if hi_s.lower() in ("none", "") else float(hi_s)
        cuts.append((field, lo, hi))
    label = spec.replace(":", "[").replace(";", "] ")
    return label, cuts


def load_field_for_sim(f, snap: int, field: str, sim_idx: int) -> np.ndarray:
    g = f[f"snap_{snap:03d}"]
    ds = g
    for part in field.split("/"):
        if part not in ds:
            raise KeyError(f"Dataset '{field}' not found under snap_{snap:03d}")
        ds = ds[part]
    arr = np.array(ds[sim_idx], dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def load_fields_for_sim(f, snap: int, fields: List[str], sim_idx: int) -> np.ndarray:
    arrays = []
    for field in fields:
        arr = load_field_for_sim(f, snap, field, sim_idx)
        if arr.ndim != 1:
            raise ValueError("Only 1D vlen arrays are supported for multi-field analysis")
        arrays.append(arr)
    if len(arrays) == 1:
        return arrays[0]
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) != 1:
        raise ValueError("Data fields have mismatched lengths for the same simulation")
    return np.stack(arrays, axis=1)


def resolve_feature_index(fields: List[str], name: str, idx: int, fallback: str):
    if name is not None and idx is not None:
        raise ValueError("Specify only one of mask-feature-name or mask-feature-idx")
    if idx is not None:
        return int(idx)
    if name is not None:
        if name not in fields:
            raise ValueError(f"Unknown feature name '{name}'. Available: {fields}")
        return fields.index(name)
    if fallback in fields:
        return fields.index(fallback)
    return 0


def apply_cuts(X: np.ndarray, cuts, field_names: List[str]) -> np.ndarray:
    if not cuts:
        return X
    if X.ndim == 1:
        if len(cuts) > 1:
            fields = {c[0] for c in cuts}
            if len(fields) != 1:
                if len(field_names) == 1:
                    print(
                        "Warning: 1D data with multiple cut fields. "
                        f"Treating all cuts as applying to '{field_names[0]}'."
                    )
                else:
                    raise ValueError(
                        "Multiple cuts on 1D data must target the same field. "
                        "Use --raw-fields with multiple entries to enable multi-field cuts."
                    )
            lo = None
            hi = None
            for _f, lo_i, hi_i in cuts:
                if lo_i is not None:
                    lo = lo_i if lo is None else max(lo, lo_i)
                if hi_i is not None:
                    hi = hi_i if hi is None else min(hi, hi_i)
        else:
            _field, lo, hi = cuts[0]
        mask = np.ones(X.shape[0], dtype=bool)
        if lo is not None:
            mask &= X >= lo
        if hi is not None:
            mask &= X <= hi
        return X[mask]
    mask = np.ones(X.shape[0], dtype=bool)
    for field, lo, hi in cuts:
        if field not in field_names:
            raise ValueError(f"Cut field '{field}' not in fields: {field_names}")
        idx = field_names.index(field)
        vals = X[:, idx]
        if lo is not None:
            mask &= vals >= lo
        if hi is not None:
            mask &= vals <= hi
    return X[mask]


def summarize_array(values: List[int]):
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5-path", type=str, default="../data/camels_LH.hdf5")
    p.add_argument("--snap", type=int, default=90)
    p.add_argument("--n-sims", type=int, default=20)
    p.add_argument("--sim-start", type=int, default=0)
    p.add_argument("--sim-ids", type=int, nargs="+", default=None, help="Explicit simulation indices")
    p.add_argument("--raw-fields", nargs="+", default=["SubhaloStellarMass"], help="Fields for raw data diagnostics")
    p.add_argument("--smf-fields", nargs="+", default=["SubhaloStellarMass"], help="Fields for SMF (cuts can use these)")
    p.add_argument("--smf-field", type=str, default="SubhaloStellarMass", help="Field used to compute SMF")
    p.add_argument("--smf-bins", type=int, default=13)
    p.add_argument("--smf-mass-range", type=float, nargs=2, default=[7.0, 11.0])
    p.add_argument("--smf-box-size", type=float, default=25 / 0.6711)
    p.add_argument("--cut-set", action="append", default=None,
                   help="Cut spec: Field:min:max;Field2:min:max (repeatable). Use 'none' for no cuts.")
    p.add_argument("--mask-prob", type=float, default=0.0)
    p.add_argument("--mask-bias-low", type=float, default=8.0)
    p.add_argument("--mask-bias-high", type=float, default=11.0)
    p.add_argument("--mask-bias-strength", type=float, default=0.0)
    p.add_argument("--mask-allow-empty", action="store_true")
    p.add_argument("--mask-feature-name", type=str, default=None)
    p.add_argument("--mask-feature-idx", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--plot-std", action="store_true", help="Add +/- std shading on SMF plots")
    p.add_argument("--out-dir", type=str, default="sanity_outputs")
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.mask_prob < 0 or args.mask_prob > 1:
        raise ValueError("--mask-prob must be between 0 and 1")
    if args.mask_bias_strength < 0:
        raise ValueError("--mask-bias-strength must be >= 0")

    os.makedirs(args.out_dir, exist_ok=True)

    cut_specs = args.cut_set or ["none"]
    cut_sets = [parse_cut_set(spec) for spec in cut_specs]

    with h5py.File(args.h5_path, "r") as f:
        nsims = int(f["snap_%03d" % args.snap]["SubhaloStellarMass"].shape[0])
        if args.sim_ids is not None:
            sim_ids = args.sim_ids
        else:
            sim_ids = list(range(args.sim_start, min(args.sim_start + args.n_sims, nsims)))

        # Raw diagnostics
        raw_fields = list(args.raw_fields)
        raw_mask_idx = resolve_feature_index(
            raw_fields,
            args.mask_feature_name,
            args.mask_feature_idx,
            fallback=args.smf_field,
        )
        raw_summary = {}
        for label, cuts in cut_sets:
            counts_before = []
            counts_after_cuts = []
            counts_after_mask = []
            q_before = []
            q_after = []
            for sim in sim_ids:
                X = load_fields_for_sim(f, args.snap, raw_fields, sim)
                n0 = X.shape[0] if X.ndim > 0 else 0
                X_cut = apply_cuts(X, cuts, raw_fields)
                n1 = X_cut.shape[0] if X_cut.ndim > 0 else 0
                if X_cut.ndim == 1:
                    masses = X_cut
                else:
                    masses = X_cut[:, raw_mask_idx]
                if args.mask_prob > 0:
                    masses_masked = apply_subhalo_mask_np(
                        masses,
                        prob=args.mask_prob,
                        bias_low=args.mask_bias_low,
                        bias_high=args.mask_bias_high,
                        bias_strength=args.mask_bias_strength,
                        keep_one=not args.mask_allow_empty,
                    )
                else:
                    masses_masked = masses
                n2 = masses_masked.shape[0]

                if masses.size > 0:
                    q_before.append(np.quantile(np.log10(np.clip(masses, 1e-12, None)), [0.1, 0.5, 0.9]).tolist())
                if masses_masked.size > 0:
                    q_after.append(np.quantile(np.log10(np.clip(masses_masked, 1e-12, None)), [0.1, 0.5, 0.9]).tolist())

                counts_before.append(int(n0))
                counts_after_cuts.append(int(n1))
                counts_after_mask.append(int(n2))

            raw_summary[label] = {
                "n_sims": len(sim_ids),
                "counts_before": summarize_array(counts_before),
                "counts_after_cuts": summarize_array(counts_after_cuts),
                "counts_after_mask": summarize_array(counts_after_mask),
                "log10_mass_quantiles_before": q_before,
                "log10_mass_quantiles_after": q_after,
            }

        # SMF plots
        smf_fields = list(args.smf_fields)
        if args.smf_field not in smf_fields:
            raise ValueError("--smf-field must be included in --smf-fields")
        smf_idx = smf_fields.index(args.smf_field)
        centers = smf_centers(args.smf_bins, tuple(args.smf_mass_range))
        smf_stats = {"centers": centers.tolist(), "cuts": {}}

        for use_mask in (False, True):
            plt.figure(figsize=(6, 4))
            for label, cuts in cut_sets:
                phi_all = []
                for sim in sim_ids:
                    X = load_fields_for_sim(f, args.snap, smf_fields, sim)
                    X_cut = apply_cuts(X, cuts, smf_fields)
                    if X_cut.ndim == 1:
                        masses = X_cut
                    else:
                        masses = X_cut[:, smf_idx]
                    if use_mask and args.mask_prob > 0:
                        masses = apply_subhalo_mask_np(
                            masses,
                            prob=args.mask_prob,
                            bias_low=args.mask_bias_low,
                            bias_high=args.mask_bias_high,
                            bias_strength=args.mask_bias_strength,
                            keep_one=not args.mask_allow_empty,
                        )
                    phi, _ = stellar_mass_function(
                        masses,
                        bins=args.smf_bins,
                        mass_range=tuple(args.smf_mass_range),
                        box_size=args.smf_box_size,
                    )
                    phi_all.append(phi)
                phi_all = np.asarray(phi_all, dtype=np.float64)
                mean_phi = phi_all.mean(axis=0) if phi_all.size else np.zeros_like(centers)
                std_phi = phi_all.std(axis=0) if phi_all.size else np.zeros_like(centers)

                plt.plot(centers, mean_phi, marker="o", label=label)
                if args.plot_std and phi_all.size:
                    plt.fill_between(centers, mean_phi - std_phi, mean_phi + std_phi, alpha=0.2)

                smf_stats["cuts"].setdefault(label, {})["masked" if use_mask else "unmasked"] = {
                    "mean_phi": mean_phi.tolist(),
                    "std_phi": std_phi.tolist(),
                }

            plt.xlabel("log10(M)")
            plt.ylabel("phi")
            plt.title("SMF " + ("masked" if use_mask else "unmasked"))
            plt.legend(frameon=False, fontsize=8)
            plt.tight_layout()
            out_path = os.path.join(args.out_dir, f"smf_{'masked' if use_mask else 'unmasked'}.png")
            plt.savefig(out_path)
            plt.close()

    # Write summary JSON
    summary = {
        "raw": raw_summary,
        "smf": smf_stats,
        "sim_ids": sim_ids,
        "cut_sets": [c[0] for c in cut_sets],
    }
    out_json = os.path.join(args.out_dir, "sanity_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved outputs to {args.out_dir}")
    print(f"SMF plots: smf_unmasked.png, smf_masked.png")
    print(f"Raw summary JSON: {out_json}")


if __name__ == "__main__":
    main()
