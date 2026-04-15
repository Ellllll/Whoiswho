set -ex

if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
fi
wandb online
wandb enabled

NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS pipeline_for_multimodal.py \
    configs/llama3/stage1_fdh.json 2>&1 | tee output/llama3/fdh_stage1.log