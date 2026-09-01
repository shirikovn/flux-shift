#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"
#SBATCH --job-name=shift-table1-i2p-eval
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

#SBATCH --output=/home/nashirikov/flux-shift/logs/%x-%j.out
#SBATCH --error=/home/nashirikov/flux-shift/logs/%x-%j.err


set -euo pipefail


PROJECT_DIR=/home/nashirikov/flux-shift

export PYTHONUNBUFFERED=1
export TABLE1_OUTPUT_ROOT="${TABLE1_OUTPUT_ROOT:-outputs/i2p_block_output}"
export TABLE1_EVAL_THRESHOLD="${TABLE1_EVAL_THRESHOLD:-0.6}"
export TABLE1_EVAL_BATCH_SIZE="${TABLE1_EVAL_BATCH_SIZE:-16}"


cd "$PROJECT_DIR"

module load python/miniconda
source "/home/nashirikov/.venv/bin/activate"

mkdir -p logs


echo "Evaluating: ${TABLE1_OUTPUT_ROOT}"
echo "Threshold: ${TABLE1_EVAL_THRESHOLD}"
echo "Batch size: ${TABLE1_EVAL_BATCH_SIZE}"


python evaluate_table1_i2p.py \
    --root "${TABLE1_OUTPUT_ROOT}" \
    --output-dir "${TABLE1_OUTPUT_ROOT}/evaluation" \
    --threshold "${TABLE1_EVAL_THRESHOLD}" \
    --batch-size "${TABLE1_EVAL_BATCH_SIZE}"
