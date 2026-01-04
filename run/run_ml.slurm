#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6    # <- match to OMP_NUM_THREADS
###SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bfpt-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --constraint="scratch"
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --mail-type="BEGIN,END"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
###SBATCH --array=0-2 

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest


###SBATCH --job-name=IS_ssp1
#SBATCH --job-name=IS_smf
###SBATCH --job-name=IS_mlp
#SBATCH --time=48:00:00      # hh:mm:ss for the job


module reset
module load python
module load cuda/11.8.0
module load ffmpeg
module list

# Fail fast and show commands
set -euo pipefail
set -x

# NOTE on job names:
# - `#SBATCH --job-name=...` is static at submission time.
# - To override at submission: `sbatch --job-name=myname run/run_ml.slurm`.
# - Or set JOB_NAME and we will attempt `scontrol update` inside the job (changes the visible name in squeue).

# Align OpenMP threads with allocated CPUs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

# Activate python virtualenv (update path if needed)
if [ -d "$HOME/pyenv/torch" ]; then
	source "$HOME/pyenv/torch/bin/activate"
else
	echo "Warning: virtualenv $HOME/pyenv/torch not found; using system python"
fi

python -V
which python

echo "job is starting on $(hostname)"


# Build command so it's easy to tweak before submission
# Select model type via env var `MODEL_TYPE` (mlp|deepset|slotsetpool). Default: mlp
MODEL_TYPE=${MODEL_TYPE:-mlp}
echo "Selected MODEL_TYPE=${MODEL_TYPE}"

# Parameter selection:
# - Set USE_PARAM_KEYS=0 (or false/no) to not pass --param-keys (use all outputs).
# - Or set PARAM_KEYS_CSV="0,1,2" to explicitly choose indices.
# - Or set PARAM_GROUP in {0,1,2,3} to pick one of the 7-parameter blocks.
USE_PARAM_KEYS=${USE_PARAM_KEYS:-1}
PARAM_GROUP=${PARAM_GROUP:-3}

GROUP_LABEL=""
PARAM_KEYS=()

if [ -n "${PARAM_KEYS_CSV:-}" ]; then
	IFS=',' read -r -a PARAM_KEYS <<< "$PARAM_KEYS_CSV"
	GROUP_LABEL="custom"
else
	case "$PARAM_GROUP" in
		1)
			PARAM_KEYS=(0 1 2 3 4 5 6)
			GROUP_LABEL="1"
			;;
		2)
			PARAM_KEYS=(7 8 9 10 11 12 13)
			GROUP_LABEL="2"
			;;
		3)
			PARAM_KEYS=(14 15 16 17 18 19 20)
			GROUP_LABEL="3"
			;;
		4)
			PARAM_KEYS=(21 22 23 24 25 26 27)
			GROUP_LABEL="4"
			;;
		-1)
			# convenience: -1 means use all parameters (no --param-keys)
			USE_PARAM_KEYS=0
			GROUP_LABEL="all"
			;;
		*)
			echo "Invalid PARAM_GROUP=${PARAM_GROUP}; valid -1,0,1,2,3" >&2
			exit 1
			;;
	esac
fi

if [ "${USE_PARAM_KEYS}" = "0" ] || [ "${USE_PARAM_KEYS,,}" = "false" ] || [ "${USE_PARAM_KEYS,,}" = "no" ]; then
	echo "Not passing --param-keys (USE_PARAM_KEYS=${USE_PARAM_KEYS}); using all parameters"
fi

# W&B naming
WAND_PROJECT_BASE=${WAND_PROJECT_BASE:-SB28_7params}
WAND_PROJECT="${WAND_PROJECT_BASE}_${GROUP_LABEL}"
WAND_RUN_NAME="${MODEL_TYPE}"

# Job name (can be overridden by JOB_NAME)
JOB_NAME=${JOB_NAME:-IS_${MODEL_TYPE}_${GROUP_LABEL}}
echo "JOB_NAME=${JOB_NAME}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
	if command -v scontrol >/dev/null 2>&1; then
		# Best-effort update; does not affect output filenames
		scontrol update JobId="${SLURM_JOB_ID}" JobName="${JOB_NAME}" || true
	fi
fi

# Common args used for all model types
COMMON_ARGS=(--use-hdf5 --h5-path ../data/camels_SB28.hdf5 --snap 90 \
	--train-frac 0.8 --val-frac 0.1 --test-frac 0.1 \
	--normalize-input log --normalize-output minmax --save-model --epochs 5000 --lr=1e-3 \
	--wandb --wandb-project "${WAND_PROJECT}" --wandb-run-name "${WAND_RUN_NAME}" --save-model --wandb-save-model)

# Model-specific args and default param-keys
case "$MODEL_TYPE" in
	mlp)
		MODEL_ARGS=(--model-type mlp --use-smf)
		;;
	deepset)
		MODEL_ARGS=(--model-type deepset)
		# example parameter keys for deepset (adjust as needed)
		;;
	slotsetpool)
		#MODEL_ARGS=(--model-type slotsetpool --slot-K 32 --slot-H 128 --slot-dropout 0.0)
		MODEL_ARGS=(--model-type slotsetpool)
		;;
	*)
		echo "Unknown MODEL_TYPE=$MODEL_TYPE; valid: mlp,deepset,slotsetpool" >&2
		exit 1
		;;
esac

# Build PY_CMD array from pieces; expand PARAM_KEYS into arguments
if [ "${USE_PARAM_KEYS}" = "0" ] || [ "${USE_PARAM_KEYS,,}" = "false" ] || [ "${USE_PARAM_KEYS,,}" = "no" ]; then
	PY_CMD=(python ../src/train.py "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}")
else
	PY_CMD=(python ../src/train.py "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}" --param-keys)
	for k in "${PARAM_KEYS[@]}"; do
		PY_CMD+=("$k")
	done
fi

echo "Running: ${PY_CMD[*]}"
"${PY_CMD[@]}"




#--wandb --wandb-project SB28_7params_4 --wandb-run-name slotsetpool --save-model --wandb-save-model --model-type slotsetpool

#--param-keys 14 15 16 17 18 19 20 \
#--param-keys 7 8 9 10 11 12 13 \
#--param-keys 0 1 2 3 4 5 6 \
#python ../src/train.py --use-hdf5 --h5-path ../data/camels_SB28.hdf5 --snap 90 \
#    --normalize-input log --normalize-output minmax --save-model --epochs 5000 --lr=1e-3 \
#	  --wandb --wandb-project SB28 --wandb-run-name smf2  --model-type mlp --use-smf \
#   	--save-model --wandb-save-model 


#python ../src/train.py --use-hdf5 --h5-path ../data/camels_LH.hdf5 --snap 90 --param-keys Omega_m sigma_8 A_SN1 A_SN2 A_AGN1 A_AGN2  \
#	   --normalize-input log --normalize-output minmax --save-model --epochs 100 --lr=1e-3 \
#	  --wandb --wandb-project deepset-reg2 --wandb-run-name smf-test --save-model --wandb-save-model --use-smf # --multi-phi

#python ../src/train.py --use-hdf5 --h5-path ../data/camels_LH.hdf5 --snap 90 --param-keys Omega_m sigma_8 A_SN1 A_SN2 A_AGN1 A_AGN2  \
#    --normalize-input log --normalize-output minmax --save-model --epochs 1000 --lr=1e-3 \
#	  --wandb --wandb-project deepset-reg3 --wandb-run-name deepset-test2 --save-model --wandb-save-model 

#python ../src/train.py --use-hdf5 --h5-path ../data/camels_LH.hdf5 --snap 90 --param-keys Omega_m sigma_8 A_SN1 A_SN2 A_AGN1 A_AGN2  \
#	   --normalize-input log --normalize-output minmax --save-model --epochs 100 --lr=1e-3 \
#		--wandb --wandb-project deepset-reg3 --wandb-run-name slotsetpool --save-model --wandb-save-model --model-type slotsetpool


#python ../src/train.py --use-hdf5 --h5-path ../data/camels_SB28.hdf5 --snap 90 \
#    --normalize-input log --normalize-output minmax --save-model --epochs 5000 --lr=1e-3 \
#	  --wandb --wandb-project SB28 --wandb-run-name slotsetpool2  --model-type slotsetpool \
#		--save-model --wandb-save-model 

