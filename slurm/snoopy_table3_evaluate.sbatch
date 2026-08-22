#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"

#SBATCH --job-name=shift-snoopy-table3-eval
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

# Set this according to your cluster's permitted
# maximum walltime.
#SBATCH --time=24:00:00

#SBATCH --output=/home/nashirikov/flux-shift/logs/%x-%A_%a.out
#SBATCH --error=/home/nashirikov/flux-shift/logs/%x-%A_%a.err


set -euo pipefail

PROJECT_DIR=/home/nashirikov/flux-shift

export HF_HOME=/home/nashirikov/huggingface
export HF_HUB_OFFLINE=1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export SNOOPY_NUM_WORKERS=4
export SNOOPY_NUM_TEMPLATES=4


cd "$PROJECT_DIR"

module load python/miniconda

source "/home/nashirikov/.venv/bin/activate"

time python evaluate_snoopy_benchmark.py --skip-fid
