#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"

#SBATCH --job-name=shift-snoopy-artifacts
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

#SBATCH --output=/home/nashirikov/flux-shift/logs/%x-%j.out
#SBATCH --error=/home/nashirikov/flux-shift/logs/%x-%j.err

set -euo pipefail


PROJECT_DIR=/home/nashirikov/flux-shift

export HF_HOME=/home/nashirikov/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false


cd "$PROJECT_DIR"

module load python/miniconda
source "/home/nashirikov/.venv/bin/activate"


echo "============================================================"
echo "SNOOPY ARTIFACT CONSTRUCTION"
echo "============================================================"

echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-not-set}"

nvidia-smi


python - <<'PY'
import torch

print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2,
    ),
    "GiB",
)
PY


echo
echo "============================================================"
echo "1/3 COLLECT DIT ACTIVATIONS"
echo "============================================================"

time python collect_activations.py \
    dataset=snoopy_20 \
    model=flux_schnell_v100 \
    generation=schnell_1024 \
    hydra.run.dir=artifacts/snoopy/dit


echo
echo "============================================================"
echo "2/3 TRAIN SVM CLASSIFIERS"
echo "============================================================"

time python train_svm.py \
    trainer.dataset_dir=artifacts/snoopy/dit/svm_dataset \
    hydra.run.dir=artifacts/snoopy/svm_training


echo
echo "============================================================"
echo "3/3 COLLECT POOLED VECTOR"
echo "============================================================"

time python collect_pooled_vector.py \
    dataset=snoopy_20 \
    model=flux_schnell_v100 \
    generation=schnell_1024 \
    target_prompt=Snoopy \
    hydra.run.dir=artifacts/snoopy/pooled


echo
echo "============================================================"
echo "ARTIFACTS"
echo "============================================================"

find artifacts/snoopy \
    -maxdepth 5 \
    -type f \
    | sort


echo
echo "============================================================"
echo "DONE"
echo "============================================================"