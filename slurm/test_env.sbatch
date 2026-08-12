#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --job-name=shift-env-test
#SBATCH --account=proj_1892
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

#SBATCH --output=/home/nashirikov/shift-flux-project/logs/%x-%j.out
#SBATCH --error=/home/nashirikov/shift-flux-project/logs/%x-%j.err

set -euo pipefail


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_DIR=/home/nashirikov/shift-flux-project

export HF_HOME=/home/nashirikov/huggingface
export HF_HUB_OFFLINE=1


# ---------------------------------------------------------
# Go to project
# ---------------------------------------------------------

cd "$PROJECT_DIR"


# ---------------------------------------------------------
# Load Python module used to create the venv
# ---------------------------------------------------------

module load python/miniconda


# ---------------------------------------------------------
# Activate virtual environment
# ---------------------------------------------------------

source "/home/nashirikov/.venv/bin/activate"


# ---------------------------------------------------------
# Basic environment info
# ---------------------------------------------------------

echo "============================================================"
echo "SLURM ENVIRONMENT TEST"
echo "============================================================"

echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-not-set}"
echo "Working dir: $(pwd)"

echo
echo "USER=$USER"
echo "HOME=$HOME"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"

echo
echo "Python executable:"
which python

echo
echo "Python version:"
python --version

echo
echo "Pip:"
python -m pip --version

echo
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-not-set}"


# ---------------------------------------------------------
# Python library test
# ---------------------------------------------------------

echo
echo "============================================================"
echo "TESTING PYTHON LIBRARIES"
echo "============================================================"

python - <<'PY'
import sys

print("Python:", sys.version)
print("Executable:", sys.executable)

import torch
print("torch:", torch.__version__)

import diffusers
print("diffusers:", diffusers.__version__)

import transformers
print("transformers:", transformers.__version__)

import huggingface_hub
print("huggingface_hub:", huggingface_hub.__version__)

import hydra
print("hydra: OK")

import omegaconf
print("omegaconf:", omegaconf.__version__)

import sklearn
print("scikit-learn:", sklearn.__version__)

import PIL
print("Pillow:", PIL.__version__)

import numpy
print("numpy:", numpy.__version__)

print()
print("CUDA available:", torch.cuda.is_available())

print()
print("ALL IMPORTS OK")
PY


# ---------------------------------------------------------
# Check Hugging Face cache
# without loading the ~large FLUX model into RAM
# ---------------------------------------------------------

echo
echo "============================================================"
echo "TESTING FLUX CACHE"
echo "============================================================"

python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "black-forest-labs/FLUX.1-schnell"
revision = "741f7c3ce8b383c54771c7003378a50191e9efe9"

print("Looking for:")
print(" repo:", repo_id)
print(" revision:", revision)

path = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_files_only=True,
)

print()
print("FLUX snapshot found locally:")
print(path)

print()
print("HF CACHE OK")
PY


echo
echo "============================================================"
echo "ALL TESTS PASSED"
echo "============================================================"
