set -ex

mkdir -p /home/yang1078/proj/WhoIsWho/log/hgt

python hgt/train_supervised_hgt.py \
    --graph-csv "${HGT_GRAPH_CSV:-/projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/pub_data_graph.csv}" \
    --train-author "${HGT_TRAIN_AUTHOR:-/projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/author_data.json}" \
    --eval-author "${HGT_EVAL_AUTHOR:-/projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/eval_author.json}" \
    --ground-truth "${HGT_GROUND_TRUTH:-/projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/eval_data.json}" \
    --split-date "${HGT_SPLIT_DATE:-2018-09-01}" \
    --output-dir "${HGT_OUTPUT_DIR:-hgt/output/fdh_hgt}" \
    --hidden-dim 769 \
    --epochs "${HGT_EPOCHS:-20}" \
    --batch-size "${HGT_BATCH_SIZE:-1024}" \
    --num-neighbors "${HGT_NUM_NEIGHBORS:-10,5}" \
    --num-workers "${HGT_NUM_WORKERS:-4}" \
    --log-interval "${HGT_LOG_INTERVAL:-20}" \
    > "/home/yang1078/proj/WhoIsWho/log/hgt/train_fdh_${SLURM_JOB_ID:-manual}.log" 2>&1