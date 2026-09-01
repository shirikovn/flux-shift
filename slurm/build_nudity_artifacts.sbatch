#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"
#SBATCH --job-name=shift-nudity-artifacts
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
export SHIFT_ARTIFACTS_ROOT="${SHIFT_ARTIFACTS_ROOT:-artifacts/nudity_block_output}"


cd "$PROJECT_DIR"

module load python/miniconda
source "/home/nashirikov/.venv/bin/activate"

mkdir -p logs


echo "=========================================="
echo "SHIFT NUDITY ARTIFACT CONSTRUCTION"
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo

nvidia-smi

echo
echo "Python:"
python --version

echo
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

memory = (
    torch.cuda.get_device_properties(0)
    .total_memory
    / 1024**3
)

print("VRAM:", round(memory, 2), "GiB")
PY


echo
echo "=========================================="
echo "1/3 COLLECT DIT ACTIVATIONS"
echo "=========================================="

time python collect_activations.py \
    dataset=nudity_135 \
    model=flux_schnell_v100 \
    generation=schnell_512 \
    hydra.run.dir="${SHIFT_ARTIFACTS_ROOT}/dit"


echo
echo "=========================================="
echo "2/3 TRAIN SVM CLASSIFIERS"
echo "=========================================="

time python train_svm.py \
    trainer.dataset_dir="${SHIFT_ARTIFACTS_ROOT}/dit/svm_dataset" \
    hydra.run.dir="${SHIFT_ARTIFACTS_ROOT}/svm_training"


echo
echo "=========================================="
echo "3/3 COLLECT POOLED VECTOR"
echo "=========================================="

time python collect_pooled_vector.py \
    dataset=nudity_135 \
    model=flux_schnell_v100 \
    generation=schnell_512 \
    target_prompt=nudity \
    hydra.run.dir="${SHIFT_ARTIFACTS_ROOT}/pooled"


echo
echo "=========================================="
echo "CHECK ARTIFACTS"
echo "=========================================="

VECTOR_COUNT=$(
    find \
        "${SHIFT_ARTIFACTS_ROOT}/dit/vectors" \
        -type f \
        -name 'step_00_vector.pt' \
        | wc -l
)

CLASSIFIER_COUNT=$(
    find \
        "${SHIFT_ARTIFACTS_ROOT}/svm_training/classifiers" \
        -type f \
        -name 'step_*_classifier.joblib' \
        | wc -l
)


echo "Step-0 steering vectors: ${VECTOR_COUNT}"
echo "SVM classifiers: ${CLASSIFIER_COUNT}"


if [[ "${VECTOR_COUNT}" -ne 19 ]]; then
    echo "ERROR: expected 19 step-0 steering vectors."
    exit 1
fi


if [[ "${CLASSIFIER_COUNT}" -ne 19 ]]; then
    echo "ERROR: expected 19 shared step-0 classifiers."
    exit 1
fi


test -f \
    "${SHIFT_ARTIFACTS_ROOT}/pooled/pooled/pooled_vector.pt"

test -f \
    "${SHIFT_ARTIFACTS_ROOT}/pooled/pooled/target_embedding.pt"

python inspect_shift_artifacts.py \
    --root "${SHIFT_ARTIFACTS_ROOT}"


echo
echo "Artifact files:"
find "${SHIFT_ARTIFACTS_ROOT}" \
    -maxdepth 5 \
    -type f \
    | sort


echo
echo "=========================================="
echo "DONE"
echo "=========================================="
