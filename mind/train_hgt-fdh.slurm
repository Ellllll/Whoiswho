#!/bin/bash
#SBATCH --job-name=mind_fdh_sup_hgt_train
#SBATCH --gres=gpu:6000ada:1
#SBATCH --nodelist=gpu-6000ada-1
#SBATCH --time=24:00:00
#SBATCH --output=../log/hgt/hgt_train_%j.out
#SBATCH --error=../log/hgt/hgt_train_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -eo pipefail

mkdir -p ../log/hgt

export HGT_BATCH_SIZE=${HGT_BATCH_SIZE:-1024}
export HGT_NUM_NEIGHBORS=${HGT_NUM_NEIGHBORS:-10,5}
export HGT_NUM_WORKERS=${HGT_NUM_WORKERS:-4}
export HGT_EPOCHS=${HGT_EPOCHS:-20}
export HGT_OUTPUT_DIR=${HGT_OUTPUT_DIR:-hgt/output/fdh_hgt}
export HGT_SPLIT_DATE=${HGT_SPLIT_DATE:-2018-09-01}

(
while true
do
    gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    cpu_mem=$(free -m | awk 'NR==2{printf "%s/%s MB", $3,$2}')
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU MEM: $gpu_mem | CPU MEM: $cpu_mem"
    sleep 1
done
) > ../log/hgt/resource_monitor_hgt_train_${SLURM_JOB_ID}.log &

MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

module unload cuDNN
module unload CUDA
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Miniforge3
source ~/.bashrc
conda activate mind

cd /home/yang1078/proj/WhoIsWho/mind

mkdir -p "$HGT_OUTPUT_DIR"

echo "Starting supervised HGT training with batch_size=$HGT_BATCH_SIZE num_neighbors=$HGT_NUM_NEIGHBORS num_workers=$HGT_NUM_WORKERS epochs=$HGT_EPOCHS split_date=$HGT_SPLIT_DATE output_dir=$HGT_OUTPUT_DIR"

bash script/train_hgt_fdh.sh