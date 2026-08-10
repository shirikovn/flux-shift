#!/bin/bash

#SBATCH --job-name=shift-snoopy-artifacts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# ---------------------------------------------------------
# Activate your environment here.
# Example:
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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi


echo
echo "========================================"
echo "1/3 Collecting Snoopy DiT activations"
echo "========================================"

python collect_activations.py \
    dataset=snoopy_20 \
    model=flux_schnell_cluster \
    generation=schnell_1024 \
    hydra.run.dir=artifacts/snoopy/dit


echo
echo "========================================"
echo "2/3 Training block/step SVM classifiers"
echo "========================================"

python train_svm.py \
    trainer.dataset_dir=artifacts/snoopy/dit/svm_dataset \
    hydra.run.dir=artifacts/snoopy/svm_training


echo
echo "========================================"
echo "3/3 Collecting pooled CLIP vector"
echo "========================================"

python collect_pooled_vector.py \
    dataset=snoopy_20 \
    model=flux_schnell_cluster \
    generation=schnell_1024 \
    target_prompt=Snoopy \
    hydra.run.dir=artifacts/snoopy/pooled


echo
echo "========================================"
echo "Artifacts completed"
echo "========================================"

find artifacts/snoopy \
    -maxdepth 4 \
    -type f \
    | sort