#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"

#SBATCH --job-name=shift-v100-test
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

#SBATCH --output=/home/nashirikov/shift-flux-project/logs/%x-%j.out
#SBATCH --error=/home/nashirikov/shift-flux-project/logs/%x-%j.err

set -euo pipefail


PROJECT_DIR=/home/nashirikov/shift-flux-project

export HF_HOME=/home/nashirikov/huggingface
export HF_HUB_OFFLINE=1


cd "$PROJECT_DIR"

module load python/miniconda
source "/home/nashirikov/.venv/bin/activate"


echo "============================================================"
echo "V100 GPU TEST"
echo "============================================================"

echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-not-set}"

echo
nvidia-smi

echo

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available()

device = torch.device("cuda:0")

props = torch.cuda.get_device_properties(device)

print("GPU:", props.name)
print("Compute capability:", f"{props.major}.{props.minor}")
print("VRAM:", round(props.total_memory / 1024**3, 2), "GiB")

print("BF16 supported:", torch.cuda.is_bf16_supported())

print()
print("Testing FP16...")

x = torch.randn(
    4096,
    4096,
    device=device,
    dtype=torch.float16,
)

y = x @ x

torch.cuda.synchronize()

print("FP16 matmul OK")
print("result dtype:", y.dtype)

del x, y

print()
print(
    "Allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 3),
    "GiB",
)

print(
    "Reserved:",
    round(torch.cuda.memory_reserved() / 1024**3, 3),
    "GiB",
)

print()
print("V100 TEST PASSED")
PY