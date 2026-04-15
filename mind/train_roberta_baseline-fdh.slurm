#!/bin/bash
#SBATCH --job-name=roberta_fdh_baseline
#SBATCH --gres=gpu:6000ada:1
#SBATCH --nodelist=gpu-6000ada-1
#SBATCH --time=24:00:00
#SBATCH --output=../log/hgt/roberta_baseline_%j.out
#SBATCH --error=../log/hgt/roberta_baseline_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -eo pipefail

mkdir -p ../log/hgt

module unload cuDNN
module unload CUDA
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Miniforge3
source ~/.bashrc
conda activate mind

if [[ -z "${WANDB_API_KEY:-}" ]]; then
	echo "WANDB_API_KEY is not set" >&2
	exit 1
fi

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-mind-roberta-baseline}
export WANDB_NAME=${WANDB_NAME:-roberta-fdh-baseline}

cd /home/yang1078/proj/WhoIsWho/mind

bash script/roberta_baseline_fdh.sh