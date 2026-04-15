set -ex

mkdir -p /home/yang1078/proj/WhoIsWho/log/hgt

WANDB_PROJECT_ARG=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_PROJECT_ARG=(--wandb-project "$WANDB_PROJECT")
fi

WANDB_NAME_ARG=()
if [[ -n "${WANDB_NAME:-}" ]]; then
    WANDB_NAME_ARG=(--wandb-name "$WANDB_NAME")
fi

WANDB_MODE_ARG=()
if [[ -n "${WANDB_MODE:-}" ]]; then
    WANDB_MODE_ARG=(--wandb-mode "$WANDB_MODE")
fi

python -u hgt/eval_hgt_classifier_baseline.py \
    --embedding-path /projects/whoiswho/model/graph/graphembedding_fdh_hgt.pkl \
    --train-author /projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/author_data.json \
    --eval-author /projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/eval_author.json \
    --ground-truth /projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/eval_data.json \
    --output-dir output/hgt_baseline/fdh_classifier_train_only \
    --hidden-dim 256 \
    --epochs 5 \
    --batch-size 2048 \
    --negative-class-weight 5.0 \
    --log-interval 50 \
    "${WANDB_PROJECT_ARG[@]}" \
    "${WANDB_NAME_ARG[@]}" \
    "${WANDB_MODE_ARG[@]}" \
    > /home/yang1078/proj/WhoIsWho/log/hgt/eval_hgt_classifier_baseline_fdh.log 2>&1
