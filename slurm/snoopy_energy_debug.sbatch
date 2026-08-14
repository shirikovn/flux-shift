#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"

#SBATCH --job-name=shift-snoopy-debug
#SBATCH --account=proj_1892

# ---------------------------------------------------------
# Exactly ONE node, ONE process, ONE GPU.
# No SLURM array.
# ---------------------------------------------------------

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

#SBATCH --time=02:00:00

#SBATCH --output=/home/nashirikov/flux-shift/logs/%x-%j.out
#SBATCH --error=/home/nashirikov/flux-shift/logs/%x-%j.err


set -euo pipefail


PROJECT_DIR=/home/nashirikov/flux-shift

export HF_HOME=/home/nashirikov/huggingface
export HF_HUB_OFFLINE=1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false


cd "${PROJECT_DIR}"


module load python/miniconda
source "/home/nashirikov/.venv/bin/activate"


mkdir -p logs


echo "============================================================"
echo "SNOOPY ENERGY-RESTORATION DEBUG"
echo "============================================================"

echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "GPUs/job:   1"
echo "CUDA vis:   ${CUDA_VISIBLE_DEVICES:-not-set}"

echo


nvidia-smi


echo
echo "============================================================"
echo "CHECKING ARTIFACTS"
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
echo "============================================================"
echo "STARTING SMALL DEBUG RUN"
echo "============================================================"

#
# IMPORTANT:
#
# The normal benchmark contains:
#
#     6 concepts x 9 seeds = 54 logical tasks.
#
# Setting:
#
#     worker_id  = 0
#     num_workers = 54
#
# means THIS SINGLE PROCESS receives only logical task 0:
#
#     Snoopy, seed 10
#
# num_workers here is only the benchmark's LOGICAL SHARD COUNT.
# It does NOT request 54 GPUs/processes.
#
# We generate only two prompt templates.
#
# For the first test:
#
#   - Snoopy only
#   - seed 10
#   - 2 templates
#   - blocks 0-4
#   - all 4 diffusion steps
#   - gamma 100
#   - classifier ON
#   - pooled steering effectively OFF
#   - Energy Restoration ON from YAML
#


START_TIME=$(date +%s)


time python snoopy_benchmark.py \
    benchmark.worker_id=0 \
    benchmark.num_workers=54 \
    benchmark.num_templates=2 \
    benchmark.output_root=outputs/snoopy_energy_debug_all_blocks \
    'shift.blocks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]' \
    'shift.steps=[0,1,2,3]' \
    shift.dit_gamma=100 \
    shift.pooled_gamma=0 \
    intervention.controller.validate_runtime=true \
    experiment.resume.mode=overwrite


END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))


echo
echo "============================================================"
echo "DEBUG RUN FINISHED"
echo "============================================================"

printf "Elapsed: %02d:%02d:%02d\n" \
    $((ELAPSED / 3600)) \
    $(((ELAPSED % 3600) / 60)) \
    $((ELAPSED % 60))


echo
echo "Inspect:"
echo "  outputs/snoopy_energy_debug/task_0/benchmark/"
echo
echo "Look especially at:"
echo "  p_cls"
echo "  eta_cls"
echo "  effective_strength"
echo "  relative_scale"
echo "  raw_norm_ratio"
echo "  final_norm_ratio"