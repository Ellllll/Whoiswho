#!/bin/bash
#SBATCH --job-name=mind_fdh_sup_hgt_export
#SBATCH --gres=gpu:6000ada:1
#SBATCH --nodelist=gpu-6000ada-1
#SBATCH --time=12:00:00
#SBATCH --output=../log/hgt/hgt_export_%j.out
#SBATCH --error=../log/hgt/hgt_export_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module unload cuDNN
module unload CUDA
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Miniforge3
source ~/.bashrc
conda activate mind

mkdir -p ../log/hgt

export HGT_OUTPUT_DIR=${HGT_OUTPUT_DIR:-hgt/output/fdh_hgt}
export HGT_CHECKPOINT=${HGT_CHECKPOINT:-$HGT_OUTPUT_DIR/checkpoint_epoch_001.pt}
export HGT_EXPORT_PATH=${HGT_EXPORT_PATH:-/projects/whoiswho/model/graph/graphembedding_fdh_hgt2.pkl}
export HGT_SPLIT_DATE=${HGT_SPLIT_DATE:-2018-09-01}

cd /home/yang1078/proj/WhoIsWho/mind

echo "Starting supervised HGT export with checkpoint=$HGT_CHECKPOINT split_date=$HGT_SPLIT_DATE output_path=$HGT_EXPORT_PATH"

bash script/export_hgt_fdh.sh