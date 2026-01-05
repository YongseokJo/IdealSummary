#!/bin/bash
set -euo pipefail

# Interactive shell with stats_sym.slurm-style resources.
# Override any resource with env vars, e.g. MEM=64g CPUS=8 GPUS=2 TIME=04:00:00

# Allow overriding some options via environment variables
N_GENERATIONS=${N_GENERATIONS:-20}
N_WEIGHT_EPOCHS=${N_WEIGHT_EPOCHS:-1000}
N_TRANSFORMS=${N_TRANSFORMS:-4}
TOP_K=${TOP_K:-16}
BATCH_SIZE=${BATCH_SIZE:-32}
H5_PATH=${H5_PATH:-../data/camels_LH.hdf5}

# Build command
PY_CMD=(python -u ../src/train_stats_sym.py \
  --h5-path "${H5_PATH}" \
  --n-generations ${N_GENERATIONS} \
  --n-weight-epochs ${N_WEIGHT_EPOCHS} \
  --n-transforms ${N_TRANSFORMS} \
  --top-k ${TOP_K} \
  --batch-size ${BATCH_SIZE} \
	--include-moments \
  --normalize-input log_std \
  --normalize-output minmax \
	--use-mlp-head \
  --use-learnable-weights
  --n-weight-kernels 12 \
	--head-hidden-dims 128 512 512 128 \
	--lr 1e-3 \
	--profile \
	--profile-steps \
  --profile-epochs 100 \
  --profile-steps-epochs 100)

#  --scheduler plateau \
#  --use-learnable-weights
#  --n-weight-kernels 12 \
#  --include-cumulants \
#  --include-moments \
#  --wandb-run-name only_moment \
#  --wandb-run-name moment_12_weigths \
#  --amp --amp-dtype fp16 \
# --amp --amp-dtype fp16 \
  

echo "Running: ${PY_CMD[*]}"
"${PY_CMD[@]}"
