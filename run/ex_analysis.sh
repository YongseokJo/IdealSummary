#!/bin/bash

module reset 
module load python  
module laod cuda/11.8.0
module load ffmpeg

source $HOME/pyenv/torch/bin/activate
python3 -V
which python3


# this is for optuna best trial
python run_slot_profiles.py --optuna-db ../data/optuna/optuna_slotsetpool.db --study-name slotsetpool_optuna --h5-path ../data/camels_LH.hdf5 --snap 90 --max-batches 20 --out-dir ../data/analysis/slot_profiles_best
