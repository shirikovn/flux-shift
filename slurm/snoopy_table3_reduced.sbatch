#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"

#SBATCH --job-name=shift-snoopy-table3
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

# Set this according to your cluster's permitted
# maximum walltime.
#SBATCH --time=24:00:00

# Exactly four persistent GPU workers.
#SBATCH --array=0-3

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


mkdir -p logs


echo "SNOOPY TABLE 3 WORKER"

echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Worker ID:  ${SLURM_ARRAY_TASK_ID}"
echo "Workers:    ${SNOOPY_NUM_WORKERS}"

echo


nvidia-smi


echo
echo "CHECKING ARTIFACTS"


VECTOR_COUNT=$(
    find \
        artifacts/snoopy/dit/vectors \
        -type f \
        -name 'step_00_vector.pt' \
        | wc -l
)


CLASSIFIER_COUNT=$(
    find \
        artifacts/snoopy/svm_training/classifiers \
        -type f \
        -name 'step_*_classifier.joblib' \
        | wc -l
)


echo "Vectors:     ${VECTOR_COUNT}"
echo "Classifiers: ${CLASSIFIER_COUNT}"


[[ "${VECTOR_COUNT}" -eq 19 ]]
[[ "${CLASSIFIER_COUNT}" -eq 76 ]]


test -f \
    artifacts/snoopy/pooled/pooled/pooled_vector.pt


test -f \
    artifacts/snoopy/pooled/pooled/target_embedding.pt


echo "Artifacts OK"


echo
echo "STARTING WORKER"


START_TIME=$(date +%s)


time python snoopy_benchmark.py


END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))


echo
echo "WORKER FINISHED"

echo "Worker ID: ${SLURM_ARRAY_TASK_ID}"

printf "Elapsed: %02d:%02d:%02d\n" \
    $((ELAPSED / 3600)) \
    $(((ELAPSED % 3600) / 60)) \
    $((ELAPSED % 60))
