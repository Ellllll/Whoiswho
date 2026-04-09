#!/bin/bash
#SBATCH --job-name=mind_stage3
#SBATCH --gres=gpu:6000ada:1
#SBATCH --nodelist=gpu-6000ada-1
#SBATCH --time=24:00:00
#SBATCH --output=../log/mind/stage3_%j.out
#SBATCH --error=../log/mind/stage3_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ==============================================
# 后台每秒记录 GPU 显存 + CPU 内存
# ==============================================
(
while true
do
    # 1. GPU 显存（保留你原来的逻辑）
    gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    # 2. CPU 内存（单位：MB，used / total）
    cpu_mem=$(free -m | awk 'NR==2{printf "%s/%s MB", $3,$2}')
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU MEM: $gpu_mem | CPU MEM: $cpu_mem"
    sleep 1
done
) > ../log/mind/resource_monitor.log &

# 保存监控进程 PID
MONITOR_PID=$!

module unload cuDNN
module unload CUDA
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Miniforge3
source ~/.bashrc
conda activate mind

# 在这里注入你的 WandB API Key
export WANDB_API_KEY="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"
export WANDB_MODE=online

export WANDB_PROJECT="mind-stage3-training"  # 自定义项目名
export WANDB_NAME="llama3-8b-stage3" 

mkdir -p output/llama3

cd /home/yang1078/proj/WhoIsWho/mind
echo "=== before run ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
bash script/stage3.sh
echo "=== after run ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# 停止 GPU 监控
kill $MONITOR_PID