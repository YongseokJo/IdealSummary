#!/bin/bash

set -euo pipefail
set -x


MODEL_TYPE=mlp PARAM_GROUP=3 sbatch run_ml.slurm
