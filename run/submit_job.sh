#!/bin/bash

set -euo pipefail
set -x


#MODEL_TYPE=mlp PARAM_GROUP=3 sbatch run_ml.slurm

#MODEL_TYPE=mlp PARAM_GROUP=3 sbatch run_ml.slurm

H5_PATH=../data/camels_SB28.hdf5 N_WEIGHT_EPOCHS=5000 sbatch run_stats_sym.slurm
