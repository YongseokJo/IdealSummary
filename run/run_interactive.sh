#!/bin/bash

module reset 
module load python  
module laod cuda/11.8.0
module load ffmpeg
module list  

source $HOME/pyenv/torch/bin/activate
python3 -V
which python3




#python ../src/train.py --use-hdf5 --h5-path ../data/camels_LH.hdf5 --snap 90 \
#	--param-keys Omega_m sigma_8 --normalize-input log_std --normalize-output minmax \
#	--wandb --wandb-project deepset-reg --save-model --wandb-save-model --epochs 100



#python ../src/diagnostics.py --h5-path ../data/camels_SB28.hdf5 --snap 90 --sample 500 --use-smf

#python ../src/train.py --use-hdf5 --use-smf --h5-path ../data/camels_LH.hdf5 --snap 90 --param-keys Omega_m A_SN1  \
#	   --normalize-input log --normalize-output minmax --save-model --epochs 100 \
#	  --wandb --wandb-project deepset-reg --wandb-run-name smf-mlp-test --save-model --wandb-save-model

#Omega_m sigma_8 A_SN1 A_SN2 A_AGN1 A_AGN2  \

#--param-keys 0 1 2 3 \
python ../src/train.py --use-hdf5 --h5-path ../data/camels_SB28.hdf5 --snap 90 --train-frac 0.8 --test-frac 0.1 --val-frac 0.1 \
	--normalize-input log --normalize-output minmax --save-model --epochs 5000 --lr=1e-3 \
	--param-keys 0 1 2 3 4 5 6 \
 	--wandb --wandb-project SB28_7params --wandb-run-name slotsetpool --save-model --wandb-save-model --model-type slotsetpool
#	 --wandb --wandb-project deepset-reg3 --wandb-run-name smf-test2 --save-model --wandb-save-model --use-smf 
 	#--wandb --wandb-project deepset-reg3 --wandb-run-name slotsetpool --save-model --wandb-save-model --model-type slotsetpool
		#--wandb --wandb-project deepset-reg3 --wandb-run-name deepset-test2 --save-model --wandb-save-model # --multi-phi 



#python ../src/optuna_search.py --h5-path ../data/camels_SB28.hdf5 --snap 90 --train-size 200 --val-size 50 --trials 10 --epochs 3 --model-type deepset
#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --train-size 200 --val-size 50 --trials 10 --epochs 3 --model-type deepset

#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --train-size 800 --val-size 200 --trials 100 --epochs 40 --model-type deepset --wandb --wandb-project optuna_test3  --normalize-input log --normalize-output minmax 

#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --train-size 800 --val-size 200 --trials 1000 --epochs 100 --model-type deepset --wandb --wandb-project optuna_test3  --normalize-input log --normalize-output minmax  --batch=32 --max-batch=50 #--storage "${STORAGE_URL}"


#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --train-size 800 --val-size 200 --trials 100 --epochs 2000 --normalize-input log --normalize-output minmax  --batch=32 --max-batch=50  --use-smf --model-type mlp --study-name smf_optuna --mem-debug

#python ../src/optuna_search.py --h5-path ../data/camels_LH.hdf5 --snap 90 --train-size 800 --val-size 200 --trials 100 --epochs 2000  --normalize-input log --normalize-output minmax  --batch=32 --max-batch=50 --model-type slotsetpool --study-name slotsetpool_optuna

