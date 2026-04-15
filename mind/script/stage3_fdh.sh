set -exo pipefail

LOG_DIR=/home/yang1078/proj/WhoIsWho/log/mind
JOB_TAG=${SLURM_JOB_ID:-manual}
STAGE3_LOG_PATH="$LOG_DIR/fdh_stage3_${JOB_TAG}.log"

mkdir -p "$LOG_DIR"

if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
fi
wandb online
wandb enabled

NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS pipeline_for_multimodal.py \
    configs/llama3/stage3_fdh.json \
    2>&1 | tee "$STAGE3_LOG_PATH"