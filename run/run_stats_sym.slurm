#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6    # <- match to OMP_NUM_THREADS
###SBATCH --partition=gpuA40x4
#SBATCH --account=bfpt-delta-gpu
#SBATCH --constraint="scratch"
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --mail-type="BEGIN,END"
#SBATCH -e logs/stats_sym_%j.err
#SBATCH -o logs/stats_sym_%j.out
###SBATCH --array=0-2

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none

###SBATCH --job-name=IS_stats_sym
#SBATCH --job-name=stats_sym
#SBATCH --time=12:00:00

module reset
module load python
module load cuda/11.8.0
module load ffmpeg
module list

set -euo pipefail
set -x

echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# Align OpenMP threads with allocated CPUs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

# Activate python virtualenv if present
if [ -d "$HOME/pyenv/torch" ]; then
    source "$HOME/pyenv/torch/bin/activate"
else
    echo "Warning: virtualenv $HOME/pyenv/torch not found; using system python"
fi

python -V
which python

echo "job is starting on $(hostname)"

# Allow overriding some options via environment variables
N_GENERATIONS=${N_GENERATIONS:-20}
N_WEIGHT_EPOCHS=${N_WEIGHT_EPOCHS:-100}
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
	--include-cumulants \
  --normalize-input log \
  --normalize-output minmax \
  --wandb \
  --wandb-project stats_sym \
  --save-model \
  --save-plots)

#--use-learnable-weights
##	--use-mlp-head \
##  --normalize-input log_std \

echo "Running: ${PY_CMD[*]}"
"${PY_CMD[@]}"

echo "Finished at: $(date)"
