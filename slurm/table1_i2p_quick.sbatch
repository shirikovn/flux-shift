#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"
#SBATCH --job-name=shift-table1-i2p
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

#SBATCH --time=24:00:00

#SBATCH --array=0-3

#SBATCH --output=/home/nashirikov/flux-shift/logs/%x-%A_%a.out
#SBATCH --error=/home/nashirikov/flux-shift/logs/%x-%A_%a.err


set -euo pipefail


PROJECT_DIR=/home/nashirikov/flux-shift

export HF_HOME=/home/nashirikov/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false


# ---------------------------------------------------------
# Experiment controls
#
# You can override these when submitting:
#
# TABLE1_SAMPLE_SIZE=1024 \
# TABLE1_STRENGTHS=250,500 \
# sbatch slurm/table1_i2p_quick.sbatch
# ---------------------------------------------------------

export TABLE1_NUM_WORKERS=4

export TABLE1_SAMPLE_SIZE="${
    TABLE1_SAMPLE_SIZE:-512
}"

export TABLE1_SAMPLE_SEED="${
    TABLE1_SAMPLE_SEED:-2026
}"

export TABLE1_STRENGTHS="${
    TABLE1_STRENGTHS:-500
}"

export TABLE1_I2P_CSV="${
    TABLE1_I2P_CSV:-data/i2p.csv
}"

export TABLE1_OUTPUT_ROOT="${
    TABLE1_OUTPUT_ROOT:-outputs/table1_i2p_quick
}"


cd "$PROJECT_DIR"

module load python/miniconda
source "/home/nashirikov/.venv/bin/activate"

mkdir -p logs


echo "=========================================="
echo "SHIFT TABLE 1 / I2P WORKER"
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Worker ID: ${SLURM_ARRAY_TASK_ID}"
echo
echo "Workers: ${TABLE1_NUM_WORKERS}"
echo "Sample size: ${TABLE1_SAMPLE_SIZE}"
echo "Sample seed: ${TABLE1_SAMPLE_SEED}"
echo "Strengths: ${TABLE1_STRENGTHS}"
echo "Dataset: ${TABLE1_I2P_CSV}"
echo "Output: ${TABLE1_OUTPUT_ROOT}"
echo


nvidia-smi


echo
echo "=========================================="
echo "CHECK DATASET"
echo "=========================================="

test -f "${TABLE1_I2P_CSV}"

ROW_COUNT=$(
    tail -n +2 "${TABLE1_I2P_CSV}" \
    | wc -l
)

echo "I2P rows: ${ROW_COUNT}"


echo
echo "=========================================="
echo "CHECK SHIFT ARTIFACTS"
echo "=========================================="

VECTOR_COUNT=$(
    find \
        artifacts/nudity/dit/vectors \
        -type f \
        -name 'step_00_vector.pt' \
        | wc -l
)

CLASSIFIER_COUNT=$(
    find \
        artifacts/nudity/svm_training/classifiers \
        -type f \
        -name 'step_*_classifier.joblib' \
        | wc -l
)


echo "Vectors: ${VECTOR_COUNT}"
echo "Classifiers: ${CLASSIFIER_COUNT}"


if [[ "${VECTOR_COUNT}" -ne 19 ]]; then
    echo "ERROR: expected 19 step-0 steering vectors."
    exit 1
fi


if [[ "${CLASSIFIER_COUNT}" -ne 76 ]]; then
    echo "ERROR: expected 76 SVM classifiers."
    exit 1
fi


test -f \
    artifacts/nudity/pooled/pooled/pooled_vector.pt

test -f \
    artifacts/nudity/pooled/pooled/target_embedding.pt


echo "Artifacts OK"


echo
echo "=========================================="
echo "STARTING WORKER"
echo "=========================================="

START_TIME=$(date +%s)

time python table1_i2p.py

END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))


echo
echo "=========================================="
echo "WORKER FINISHED"
echo "=========================================="
echo "Worker ID: ${SLURM_ARRAY_TASK_ID}"

printf "Elapsed: %02d:%02d:%02d\n" \
    $((ELAPSED / 3600)) \
    $(((ELAPSED % 3600) / 60)) \
    $((ELAPSED % 60))