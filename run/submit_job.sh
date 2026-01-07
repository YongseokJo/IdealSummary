#!/bin/bash

set -euo pipefail
set -x


#MODEL_TYPE=mlp PARAM_GROUP=3 sbatch run_ml.slurm

#MODEL_TYPE=mlp PARAM_GROUP=3 sbatch run_ml.slurm

#H5_PATH=../data/camels_SB28.hdf5 N_WEIGHT_EPOCHS=10000 sbatch run_stats_sym.slurm
N_WEIGHT_EPOCHS=1000 sbatch run_cuts.slurm
