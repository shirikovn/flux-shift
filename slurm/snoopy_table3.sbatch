#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"

#SBATCH --job-name=shift-snoopy-table3
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

#SBATCH --time=04:00:00

# 6 concepts * 9 seeds = 54 tasks.
#
# Start conservatively with at most 4 GPUs
# running simultaneously.
#SBATCH --array=0-53%4

#SBATCH --output=/home/nashirikov/shift-flux-project/logs/%x-%A_%a.out
#SBATCH --error=/home/nashirikov/shift-flux-project/logs/%x-%A_%a.err


set -euo pipefail


# =========================================================
# Paths / environment
# =========================================================

PROJECT_DIR=/home/nashirikov/shift-flux-project


export HF_HOME=/home/nashirikov/huggingface
export HF_HUB_OFFLINE=1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false


cd "$PROJECT_DIR"


module load python/miniconda

source "/home/nashirikov/.venv/bin/activate"


mkdir -p \
    /home/nashirikov/shift-flux-project/logs


# =========================================================
# Job information
# =========================================================

echo "============================================================"
echo "SNOOPY TABLE 3"
echo "============================================================"

echo "Date:             $(date)"
echo "Host:             $(hostname)"
echo "Job ID:           ${SLURM_JOB_ID:-not-set}"
echo "Array job ID:     ${SLURM_ARRAY_JOB_ID:-not-set}"
echo "Array task ID:    ${SLURM_ARRAY_TASK_ID:-not-set}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Working dir:      $(pwd)"
echo "HF_HOME:          ${HF_HOME}"
echo "HF_HUB_OFFLINE:   ${HF_HUB_OFFLINE}"

echo


# =========================================================
# Decode task ID for readable logs
# =========================================================

SEEDS=(
    10
    20
    30
    42
    50
    60
    70
    80
    90
)

CONCEPTS=(
    snoopy
    mickey
    spongebob
    pikachu
    dog
    legislator
)


TASK_ID="${SLURM_ARRAY_TASK_ID}"

NUM_SEEDS="${#SEEDS[@]}"

CONCEPT_INDEX=$(( TASK_ID / NUM_SEEDS ))
SEED_INDEX=$(( TASK_ID % NUM_SEEDS ))

CONCEPT="${CONCEPTS[$CONCEPT_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"


echo "Benchmark task:"
echo "  task:    ${TASK_ID}"
echo "  concept: ${CONCEPT}"
echo "  seed:    ${SEED}"

echo


# =========================================================
# GPU check
# =========================================================

nvidia-smi

echo


python - <<'PY'
import torch

assert torch.cuda.is_available(), (
    "CUDA is unavailable"
)

props = torch.cuda.get_device_properties(0)

print("PyTorch:", torch.__version__)
print("GPU:", props.name)
print(
    "VRAM:",
    round(
        props.total_memory / 1024**3,
        2,
    ),
    "GiB",
)
print(
    "Compute capability:",
    torch.cuda.get_device_capability(0),
)
print(
    "Native BF16:",
    torch.cuda.is_bf16_supported(
        including_emulation=False
    ),
)
PY


# =========================================================
# Artifact preflight
# =========================================================

echo
echo "============================================================"
echo "CHECKING SNOOPY ARTIFACTS"
echo "============================================================"


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


echo "Shared step-0 token vectors: ${VECTOR_COUNT}"
echo "Per-step SVM classifiers:    ${CLASSIFIER_COUNT}"


if [[ "${VECTOR_COUNT}" -ne 19 ]]; then
    echo "ERROR: expected 19 step-0 vectors"
    exit 1
fi


if [[ "${CLASSIFIER_COUNT}" -ne 76 ]]; then
    echo "ERROR: expected 76 classifiers"
    exit 1
fi


test -f \
    artifacts/snoopy/pooled/pooled/pooled_vector.pt


test -f \
    artifacts/snoopy/pooled/pooled/target_embedding.pt


echo "Pooled artifacts: OK"


# =========================================================
# Optional model-cache sanity check
# =========================================================

python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=(
        "black-forest-labs/"
        "FLUX.1-schnell"
    ),
    revision=(
        "741f7c3ce8b383c54771c7003378a50191e9efe9"
    ),
    local_files_only=True,
)

print("Offline FLUX snapshot:")
print(path)
PY


# =========================================================
# GPU monitor
# =========================================================

GPU_MONITOR_FILE="/home/nashirikov/shift-flux-project/logs/gpu-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"


nvidia-smi \
    --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
    --format=csv \
    -l 5 \
    > "${GPU_MONITOR_FILE}" &


NVIDIA_SMI_PID=$!


cleanup_monitor()
{
    kill "${NVIDIA_SMI_PID}" \
        2>/dev/null || true

    wait "${NVIDIA_SMI_PID}" \
        2>/dev/null || true
}


trap cleanup_monitor EXIT


# =========================================================
# Run
# =========================================================

echo
echo "============================================================"
echo "STARTING GENERATION"
echo "============================================================"

time python snoopy_benchmark.py

# =========================================================
# Stop monitoring cleanly
# =========================================================

cleanup_monitor

trap - EXIT


# =========================================================
# Validate task outputs
# =========================================================

echo
echo "============================================================"
echo "VALIDATING OUTPUT"
echo "============================================================"


NUM_TEMPLATES="${SNOOPY_NUM_TEMPLATES:-80}"

EXPECTED_RUNS=$(( NUM_TEMPLATES * 2 ))

TASK_OUTPUT="outputs/snoopy_table3/task_${SLURM_ARRAY_TASK_ID}/benchmark"

IMAGE_COUNT=0
RECORD_COUNT=0


if [[ -d "${TASK_OUTPUT}/images" ]]; then

    IMAGE_COUNT=$(
        find \
            "${TASK_OUTPUT}/images" \
            -type f \
            -name '*.png' \
            | wc -l
    )

fi


if [[ -d "${TASK_OUTPUT}/records" ]]; then

    RECORD_COUNT=$(
        find \
            "${TASK_OUTPUT}/records" \
            -type f \
            -name '*.yaml' \
            | wc -l
    )

fi


echo "Expected runs: ${EXPECTED_RUNS}"
echo "Images:        ${IMAGE_COUNT}"
echo "Records:       ${RECORD_COUNT}"


if [[ "${IMAGE_COUNT}" -ne "${EXPECTED_RUNS}" ]]; then
    echo "ERROR: unexpected number of images"
    exit 1
fi


if [[ "${RECORD_COUNT}" -ne "${EXPECTED_RUNS}" ]]; then
    echo "ERROR: unexpected number of records"
    exit 1
fi


echo
echo "============================================================"
echo "TASK COMPLETED SUCCESSFULLY"
echo "============================================================"

echo "Task:    ${TASK_ID}"
echo "Concept: ${CONCEPT}"
echo "Seed:    ${SEED}"
echo "Images:  ${IMAGE_COUNT}"
echo "GPU log: ${GPU_MONITOR_FILE}"