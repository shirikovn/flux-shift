#!/bin/bash

#SBATCH --job-name=shift-snoopy-table3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

#SBATCH --array=0-53%12

#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# ---------------------------------------------------------
# Activate your environment here.
#
# source .venv/bin/activate
#
# or:
#
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate flux-shift
# ---------------------------------------------------------

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

python snoopy_benchmark.py