#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"

#SBATCH --job-name=shift-flux-v100
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

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
echo "FLUX V100 SMOKE TEST"
echo "============================================================"

echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-not-set}"

echo

nvidia-smi


mkdir -p outputs


python test_flux_v100.py


echo
echo "============================================================"
echo "TEST FINISHED"
echo "============================================================"