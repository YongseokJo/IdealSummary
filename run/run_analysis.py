#!/usr/bin/env python3
"""CLI wrapper to load best Optuna trial (architecture) and run SlotSetPool analyses.

It reconstructs the model architecture from the best trial's params in the Optuna
SQLite DB, optionally loads a local checkpoint, samples a batch from the dataset,
and runs the analysis utilities in `tools/setpool_analysis.py`.
"""
import os
import sys
import argparse
import json

import torch

# ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
	sys.path.insert(0, SRC)

import optuna
from tools.setpool_analysis import slots_ablation, component_ablation, slot_gradients, slot_correlation_with_logm, plot_bar
from src import optuna_search


def reconstruct_model_spec_from_params(params: dict):
	# detect slotsetpool-style params
	if any(k.startswith("slot_") for k in params.keys()) or any(k in params for k in ("slot_K", "slot_H")):
		# map expected names from optuna_search
		spec = {"model_type": "slotsetpool", "use_train_model": False}
		# direct params
		if "slot_K" in params:
			spec["slot_K"] = int(params["slot_K"])
		if "slot_H" in params:
			spec["slot_H"] = int(params["slot_H"])
		# phi/head hidden may be named slot_phi_h0/1 etc
		phi_h = []
		head_h = []
		for k, v in params.items():
			if k.startswith("slot_phi_h"):
				phi_h.append(int(v))
			if k.startswith("slot_head_h"):
				head_h.append(int(v))
		if phi_h:
			spec["slot_phi_hidden"] = phi_h
		if head_h:
			spec["slot_head_hidden"] = head_h
		if "slot_dropout" in params:
			spec["slot_dropout"] = float(params["slot_dropout"])
		return spec

	# fallback: return empty spec (will create default MLP)
	return {"model_type": "mlp", "use_train_model": False}


def load_best_trial(storage_path: str, study_name: str):
	storage_url = f"sqlite:///{os.path.abspath(storage_path)}"
	study = optuna.load_study(study_name=study_name, storage=storage_url)
	best = study.best_trial
	return best


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--optuna-db", type=str, default="data/optuna/optuna_optuna_slosetpool.db", help="Path to Optuna sqlite DB")
	p.add_argument("--study-name", type=str, required=True, help="Optuna study name")
	p.add_argument("--h5-path", type=str, default="../data/camels_LH.hdf5")
	p.add_argument("--snap", type=int, default=90)
	p.add_argument("--batch", type=int, default=16)
	p.add_argument("--train-frac", type=float, default=0.8)
	p.add_argument("--val-frac", type=float, default=0.2)
	p.add_argument("--test-frac", type=float, default=0.0)
	p.add_argument("--use-smf", action="store_true", default=False)
	p.add_argument("--data-field", type=str, default="SubhaloStellarMass")
	p.add_argument("--ckpt-path", type=str, default=None, help="Optional checkpoint to load (torch.save dict with 'model_state_dict')")
	p.add_argument("--out-dir", type=str, default="analysis_outputs")
	args = p.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)

	print(f"Loading study from {args.optuna_db} (study={args.study_name})")
	best = load_best_trial(args.optuna_db, args.study_name)
	print(f"Best trial id={best.number} value={best.value}")
	print("Params:")
	print(best.params)

	model_spec = reconstruct_model_spec_from_params(best.params)
	print("Reconstructed model spec:", model_spec)

	# build small dataset and loader to infer input dims
	ds, train_loader, val_loader, _test_loader = optuna_search.get_data_loaders(
		args.h5_path,
		args.snap,
		None,
		args.data_field,
		args.batch,
		args.train_frac,
		args.val_frac,
		args.test_frac,
		args.use_smf,
	)

	# sample a batch to infer dims
	sample = next(iter(train_loader))
	if len(sample) == 3:
		Xb, maskb, yb = sample
		use_mask = True
		input_dim = int(Xb.shape[2])
	else:
		Xb, yb = sample
		use_mask = False
		input_dim = int(Xb.shape[1])
	target_dim = 1 if yb.ndim == 1 else int(yb.shape[1])

	model = optuna_search._build_model_from_spec(model_spec, input_dim, target_dim, use_mask)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	if args.ckpt_path is not None and os.path.exists(args.ckpt_path):
		print(f"Loading checkpoint {args.ckpt_path}")
		ck = torch.load(args.ckpt_path, map_location=device)
		if "model_state_dict" in ck:
			model.load_state_dict(ck["model_state_dict"], strict=False)
		else:
			try:
				model.load_state_dict(ck, strict=False)
			except Exception as e:
				print(f"Failed to load checkpoint: {e}")
	else:
		print("No checkpoint provided — model weights are untrained/random")

	model.eval()

	# use a small validation batch for analysis
	batch = next(iter(val_loader))
	if len(batch) == 3:
		X, mask, y = batch
	else:
		X, y = batch
		mask = None

	X = X.to(device)
	if mask is not None:
		mask = mask.to(device)

	# run analyses
	print("Running slot ablation...")
	sa = slots_ablation(model, X, mask)
	print("slot deltas:", sa["slot_deltas"])
	plot_bar(sa["slot_deltas"], labels=[f"slot{i}" for i in range(model.K)], title="Slot ablation deltas", out_path=os.path.join(args.out_dir, "slot_deltas.png"))

	print("Running component ablation...")
	ca = component_ablation(model, X, mask)
	print(ca["effects"])
	plot_bar(list(ca["effects"].values()), labels=list(ca["effects"].keys()), title="Component ablation effects", out_path=os.path.join(args.out_dir, "component_effects.png"))

	print("Computing slot gradients...")
	sg = slot_gradients(model, X, mask)
	if sg is not None:
		print("agg:", sg["agg"])
	else:
		print("No gradients (check requires_grad and model on CPU/GPU)")

	print("Computing slot vs logm correlations...")
	corr = slot_correlation_with_logm(model, X, mask)
	print("correlations:", corr)

	# save a summary JSON
	summary = {
		"trial": int(best.number),
		"value": float(best.value),
		"params": best.params,
		"slot_deltas": sa["slot_deltas"].tolist() if hasattr(sa["slot_deltas"], "tolist") else sa["slot_deltas"],
		"component_effects": ca["effects"],
		"slot_corr": corr.tolist() if hasattr(corr, "tolist") else corr,
	}
	with open(os.path.join(args.out_dir, "analysis_summary.json"), "w") as f:
		json.dump(summary, f, indent=2)
	print(f"Saved analysis outputs to {args.out_dir}")


if __name__ == "__main__":
	main()
