#!/bin/bash
#SBATCH --mem=32g
#BATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6    # <- match to OMP_NUM_THREADS
###SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bfpt-delta-gpu    # <- match to a "Project" returned by the "accounts" command
###SBATCH --account=benb-delta-gpu 


#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --mail-type="BEGIN,END"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest


###SBATCH --job-name=opt_deep
###SBATCH --job-name=opt_smf
#SBATCH --job-name=opt_ssp
###SBATCH --job-name=test
#SBATCH --array=0-3 


module reset 
module load python  
module laod cuda/11.8.0
module load ffmpeg
module list  

#export STORAGE_URL="sqlite:////u/gkerex/projects/IdealSummary/data/optuna/optuna_slotsetpool.db"
#export STORAGE_URL="sqlite:////u/gkerex/projects/IdealSummary/data/optuna/optuna_smf.db"
#export STORAGE_URL="sqlite:////u/gkerex/projects/IdealSummary/data/optuna/optuna_deepset.db"
#export STORAGE_URL="sqlite:////u/gkerex/projects/IdealSummary/data/optuna/optuna_test.db"

#export STORAGE_URL="sqlite:////u/gkerex/projects/IdealSummary/data/optuna/optuna_smf_sb28.db"
export STORAGE_URL="sqlite:////u/gkerex/projects/IdealSummary/data/optuna/optuna_slotsetpool_sb28.db"
#export STORAGE_URL="sqlite:////u/gkerex/projects/IdealSummary/data/optuna/optuna_deepset_sb28.db"

source $HOME/pyenv/torch/bin/activate
python -V
which python

echo "job is starting on `hostname`"

#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --train-size 200 --val-size 50 --trials 10 --epochs 3 --model-type deepset

#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --train-size 800 --val-size 200 --trials 100 --epochs 2000 --wandb --wandb-project optuna_test_deepset  --normalize-input log --normalize-output minmax  --batch=32 --max-batch=50 --storage "${STORAGE_URL}" --model-type deepset --mem-debug --mem-log-file ../data/optuna/mem_deepset.log

#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --train-size 800 --val-size 200 --trials 100 --epochs 2000 --wandb --wandb-project optuna_test_smf --normalize-input log --normalize-output minmax  --batch=32 --max-batch=50 --storage "${STORAGE_URL}" --use-smf --model-type mlp  --study-name smf_optuna --mem-debug --mem-log-file ../data/optuna/mem_smf.log

#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --train-size 800 --val-size 200 --trials 100 --epochs 2000 --wandb --wandb-project optuna_test_slotsetpool  --normalize-input log --normalize-output minmax  --batch=32 --max-batch=50 --storage "${STORAGE_URL}" --model-type slotsetpool  --study-name slotsetpool_optuna --mem-debug --mem-log-file ../data/optuna/mem_slotsetpool.log



#python ../src/optuna_search.py --h5-path ../data/camels_SB28.hdf5 --snap 90 --train-size 800 --val-size 200 --trials 100 --epochs 2000 --wandb --wandb-project optuna_test_deepset_sb28  --normalize-input log --normalize-output minmax  --batch=32 --max-batch=50 --storage "${STORAGE_URL}" --model-type deepset --mem-debug --mem-log-file ../data/optuna/mem_deepset.log

#python ../src/optuna_search.py --h5-path ../data/camels_SB28.hdf5 --snap 90 --train-size 800 --val-size 200 --trials 100 --epochs 2000 --wandb --wandb-project optuna_test_smf_sb28 --normalize-input log --normalize-output minmax  --batch=32 --max-batch=50 --storage "${STORAGE_URL}" --use-smf --model-type mlp  --study-name smf_optuna --mem-debug --mem-log-file ../data/optuna/mem_smf.log

python ../src/optuna_search.py --h5-path ../data/camels_SB28.hdf5 --snap 90 --train-size 1600 --val-size 400 --trials 200 --epochs 4000 --wandb --wandb-project optuna_test_slotsetpool_sb28  --normalize-input log --normalize-output minmax  --batch=32 --max-batch=50 --storage "${STORAGE_URL}" --model-type slotsetpool  --study-name slotsetpool_optuna --mem-debug --mem-log-file ../data/optuna/mem_slotsetpool.log





#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --trials 300 \
#	   --normalize-input log --normalize-output minmax --save-model --epochs 100 \
#		 --wandb --wandb-project slotsetpool-optuna --model-type slotsetpool --storage "${STORAGE_URL}" 
		 
#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --train-size 200 --val-size 50 --trials 10 --epochs 10 --wandb --wandb-project optuna_test --normalize-input log --normalize-output minmax  --batch=32 --max-batch=50 --storage "${STORAGE_URL}" --model-type slotsetpool  --study-name optuna_test --mem-debug --mem-log-file ../data/optuna/mem_test.log
