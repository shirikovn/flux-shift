#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --job-name=shift-snoopy-smoke
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

#SBATCH --gres=gpu:1
#SBATCH --constraint="[type_e|type_f]"

#SBATCH --output=/home/nashirikov/shift-flux-project/logs/%x-%j.out
#SBATCH --error=/home/nashirikov/shift-flux-project/logs/%x-%j.err

set -euo pipefail


PROJECT_DIR=/home/nashirikov/shift-flux-project

export HF_HOME=/home/nashirikov/huggingface
export HF_HUB_OFFLINE=1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false


cd "$PROJECT_DIR"

module load python/miniconda

source "/home/nashirikov/.venv/bin/activate"


echo "============================================================"
echo "SNOOPY GPU SMOKE TEST"
echo "============================================================"

echo "Date: $(date)"
echo "Host: $(hostname)"

nvidia-smi


# task_id=3:
# Snoopy, seed 42
#
# Only 4/80 templates.

python snoopy_benchmark.py \
    benchmark.task_id=3 \
    benchmark.num_templates=4


echo
echo "============================================================"
echo "SMOKE TEST PASSED"
echo "============================================================"
