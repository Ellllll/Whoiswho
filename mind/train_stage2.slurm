#!/bin/bash
#SBATCH --job-name=mind_stage2
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-a6000-1
#SBATCH --constraint=gpu
#SBATCH --time=12:00:00
#SBATCH --output=../log/mind/stage2_%j.out
#SBATCH --error=../log/mind/stage2_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16  # 关键！给1个进程分配16核CPU（8核/进程是最低标准）

module unload cuDNN
module unload CUDA
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Miniforge3
conda activate mind

# 在这里注入你的 WandB API Key
if [[ -z "${WANDB_API_KEY:-}" ]]; then
	echo "WANDB_API_KEY is not set" >&2
	exit 1
fi
export WANDB_MODE=online

export WANDB_PROJECT="mind-stage2-training"  # 自定义项目名
export WANDB_NAME="llama3-8b-stage2" 

mkdir -p output/llama3

cd /home/yang1078/proj/WhoIsWho/mind
bash script/stage2.sh
