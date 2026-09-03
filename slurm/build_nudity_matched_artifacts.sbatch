#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"
#SBATCH --job-name=shift-nudity-matched
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

#SBATCH --time=08:00:00

#SBATCH --output=/home/nashirikov/flux-shift/logs/%x-%j.out
#SBATCH --error=/home/nashirikov/flux-shift/logs/%x-%j.err


set -euo pipefail


PROJECT_DIR=/home/nashirikov/flux-shift

export HF_HOME=/home/nashirikov/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SHIFT_ARTIFACTS_ROOT="${SHIFT_ARTIFACTS_ROOT:-artifacts/nudity_matched_block_output}"
export SHIFT_MODEL_DTYPE="${SHIFT_MODEL_DTYPE:-float16}"
export SHIFT_MEMORY_STRATEGY="${SHIFT_MEMORY_STRATEGY:-model_cpu_offload}"
export SHIFT_SEEDS_PER_PAIR="${SHIFT_SEEDS_PER_PAIR:-2}"
export SHIFT_COLLECTION_RESOLUTION="${SHIFT_COLLECTION_RESOLUTION:-512}"


cd "$PROJECT_DIR"

module load python/miniconda
source "/home/nashirikov/.venv/bin/activate"

mkdir -p logs


echo "=========================================="
echo "SHIFT MATCHED NUDITY ARTIFACTS"
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Root: ${SHIFT_ARTIFACTS_ROOT}"
echo "Model dtype: ${SHIFT_MODEL_DTYPE}"
echo "Memory strategy: ${SHIFT_MEMORY_STRATEGY}"
echo "Matched seeds per source pair: ${SHIFT_SEEDS_PER_PAIR}"
echo "Collection resolution: ${SHIFT_COLLECTION_RESOLUTION}x${SHIFT_COLLECTION_RESOLUTION}"
echo

nvidia-smi


echo
echo "=========================================="
echo "1/3 COLLECT MATCHED DIT ACTIVATIONS"
echo "=========================================="

time python collect_activations.py \
    dataset=nudity_matched_135 \
    model=flux_schnell_v100 \
    generation=schnell_512 \
    generation.width="${SHIFT_COLLECTION_RESOLUTION}" \
    generation.height="${SHIFT_COLLECTION_RESOLUTION}" \
    collection.seeds_per_pair="${SHIFT_SEEDS_PER_PAIR}" \
    hydra.run.dir="${SHIFT_ARTIFACTS_ROOT}/dit"


echo
echo "=========================================="
echo "2/3 TRAIN GROUP-VALIDATED SVM CLASSIFIERS"
echo "=========================================="

time python train_svm.py \
    trainer.dataset_dir="${SHIFT_ARTIFACTS_ROOT}/dit/svm_dataset" \
    trainer.split_by_pair=true \
    trainer.refit_full_after_validation=true \
    hydra.run.dir="${SHIFT_ARTIFACTS_ROOT}/svm_training"


echo
echo "=========================================="
echo "3/3 COLLECT MATCHED POOLED VECTOR"
echo "=========================================="

time python collect_pooled_vector.py \
    dataset=nudity_matched_135 \
    model=flux_schnell_v100 \
    generation=schnell_512 \
    generation.width="${SHIFT_COLLECTION_RESOLUTION}" \
    generation.height="${SHIFT_COLLECTION_RESOLUTION}" \
    target_prompt=nudity \
    hydra.run.dir="${SHIFT_ARTIFACTS_ROOT}/pooled"


echo
echo "=========================================="
echo "CHECK ARTIFACTS"
echo "=========================================="

STANDARD_COUNT=$(
    find "${SHIFT_ARTIFACTS_ROOT}/dit/vectors" \
        -type f -name 'step_00_vector.pt' | wc -l
)

CONSISTENT_COUNT=$(
    find "${SHIFT_ARTIFACTS_ROOT}/dit/vectors" \
        -type f -name 'step_00_consistent_vector.pt' | wc -l
)

CLASSIFIER_COUNT=$(
    find "${SHIFT_ARTIFACTS_ROOT}/svm_training/classifiers" \
        -type f -name 'step_00_classifier.joblib' | wc -l
)

echo "Standard token-wise vectors: ${STANDARD_COUNT}"
echo "Consistency-weighted token-wise vectors: ${CONSISTENT_COUNT}"
echo "SVM classifiers: ${CLASSIFIER_COUNT}"

[[ "${STANDARD_COUNT}" -eq 19 ]]
[[ "${CONSISTENT_COUNT}" -eq 19 ]]
[[ "${CLASSIFIER_COUNT}" -eq 19 ]]

test -f "${SHIFT_ARTIFACTS_ROOT}/pooled/pooled/pooled_vector.pt"
test -f "${SHIFT_ARTIFACTS_ROOT}/pooled/pooled/target_embedding.pt"

python inspect_shift_artifacts.py --root "${SHIFT_ARTIFACTS_ROOT}"

echo "Matched artifact validation: OK"
