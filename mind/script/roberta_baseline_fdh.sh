set -ex

mkdir -p /home/yang1078/proj/WhoIsWho/log/hgt

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-mind-roberta-baseline}

if [ -n "$WANDB_API_KEY" ] && command -v wandb >/dev/null 2>&1; then
    wandb login "$WANDB_API_KEY"
fi

python -u roberta_baseline.py \
    configs/roberta/baseline_fdh.json \
    > /home/yang1078/proj/WhoIsWho/log/hgt/roberta_baseline_fdh.log 2>&1