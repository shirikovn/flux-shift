#!/bin/bash

#SBATCH --partition=rocky
#SBATCH --constraint="[type_a|type_b|type_c]"
#SBATCH --job-name=shift-manual20-fp32
#SBATCH --account=proj_1892

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

#SBATCH --time=08:00:00
#SBATCH --array=0-3

#SBATCH --output=/home/nashirikov/flux-shift/logs/%x-%A_%a.out
#SBATCH --error=/home/nashirikov/flux-shift/logs/%x-%A_%a.err


# Fixed 20-prompt FP32 stress test. The shared worker script
# handles artifact validation, generation, and resumption.
export TABLE1_NUM_WORKERS=4
export TABLE1_SAMPLE_SIZE=0
export TABLE1_I2P_CSV=data/i2p_stress_manual_20.csv
export TABLE1_OUTPUT_ROOT=outputs/i2p_manual_20_fp32
export TABLE1_STRENGTHS=10,20,45,100,250
export TABLE1_ETA_MAX=1
export SHIFT_MODEL_DTYPE=float32
export SHIFT_MEMORY_STRATEGY=sequential_cpu_offload


source /home/nashirikov/flux-shift/slurm/table1_i2p_quick.sbatch
