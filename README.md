# IdealSummary

Research code for set-based regression on CAMELS-style data using PyTorch.

This repo contains:
- A simple DeepSets baseline ([src/deepset.py](src/deepset.py), [src/train.py](src/train.py))
- A summary-statistics model and training utilities ([src/stats_sym.py](src/stats_sym.py), [src/train_stats_sym.py](src/train_stats_sym.py))
- Scripts for baselines and analysis in [run/](run/)

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data

Most experiments expect an HDF5 dataset (e.g. CAMELS LH) available locally.

- Default path used by scripts: `data/camels_LH.hdf5`
- The dataset is intentionally ignored by git (see `.gitignore`).

## Key scripts

### 1) Synthetic DeepSets sanity run

```bash
python src/train.py --epochs 20
```

### 2) Summary-statistics baselines (ridge + MLP head)

Compute classical summary-stat features and fit:
- closed-form ridge regression
- an MLP “head” on top of the features

Ridge/MLP baseline:

```bash
python run/baseline_linear_stats.py --h5-path data/camels_LH.hdf5 --max-samples 200
```

MLP baseline with 4-way normalization comparison:
- **Input normalization**: per-element normalization before summary stats (e.g. `log_std`)
- **Feature normalization**: standardization after summary stats are computed

This runs the 4 combinations: input_norm {none, log_std} × feat_norm {none, standardize}.

```bash
python run/baseline_mlp_stats.py --h5-path data/camels_LH.hdf5 --max-samples 200 --input-norm log_std
```

### 3) Summary-statistics training

Main training entrypoint lives in [src/train_stats_sym.py](src/train_stats_sym.py).

Notes:
- Quantiles are optional (and off by default) because they can be expensive.
- Symbolic transforms and Top-K selection are optional; disabling them reduces the model to “raw features → summary stats (optionally weighted) → head”.

## Tests / sanity checks

Quick numeric/unit sanity checks:

```bash
python run/sanity_check.py
python run/tests/test_summary_stats.py
```
