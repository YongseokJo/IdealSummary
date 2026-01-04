#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --account=bfpt-delta-gpu
#SBATCH --constraint="scratch"
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --mail-type="BEGIN,END"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none

#SBATCH --job-name=IS_dsr
#SBATCH --time=24:00:00      # production run

module reset
module load python
module load cuda/11.8.0
module list

set -euo pipefail
set -x

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

# Activate virtualenv
if [ -d "$HOME/pyenv/torch" ]; then
    source "$HOME/pyenv/torch/bin/activate"
else
    echo "Warning: virtualenv not found; using system python"
fi

python -V
which python

echo "Job starting on $(hostname)"

cd /u/gkerex/projects/IdealSummary

# Run Set-DSR training - PRODUCTION RUN
# Full evolutionary search with advanced features
python src/train_dsr.py \
    --operator-scope intermediate \
    --use-learnable-constants \
    --h5-path data/camels_LH.hdf5 \
    --snap 90 \
    --param-keys 0 1 2 3 4 5 \
    --n-summaries 12 \
    --max-depth 6 \
    --n-generations 300 \
    --mlp-epochs 2000 \
    --population-size 200 \
    --complexity-weight 0.005 \
    --normalize-input log_std \
    --normalize-output minmax \
    --log-interval 10 \
    --wandb \
    --wandb-project set-dsr \
    --wandb-run-name "dsr_simple_retrain_rl_search_inter" \
    --save-path run/data/models/set_dsr/ \
    --plot-path run/data/models/plots/set_dsr/
    --use-rl-search

echo "Job completed successfully"
