set -ex

# pip install -r requirements.txt
wandb login $WANDB_API_KEY
wandb online   
wandb enabled

NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS  pipeline_for_multimodal.py \
    configs/llama3/stage2.json 2>&1 | tee output/llama3/stage2.log

