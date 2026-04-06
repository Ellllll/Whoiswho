#!/bin/bash
#SBATCH --job-name=mind_eval
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=gpu-a6000-1
#SBATCH --time=12:00:00
#SBATCH --output=../log/mind/eval_%j.out
#SBATCH --error=../log/mind/eval_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16  # 关键！给1个进程分配16核CPU（8核/进程是最低标准）

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
) > ../log/mind/eval_resource_monitor.log &

# 保存监控进程 PID
MONITOR_PID=$!

module unload cuDNN
module unload CUDA
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Miniforge3
conda init
conda activate mind

# 在这里注入你的 WandB API Key
export WANDB_API_KEY="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"
export WANDB_MODE=online

export WANDB_PROJECT="mind-eval"  # 自定义项目名
export WANDB_NAME="llama3-8b-eval" 

mkdir -p output/llama3
cd /home/yang1078/proj/WhoIsWho/mind
bash script/eval.sh

# 停止 GPU 监控
kill $MONITOR_PID