set -ex

mkdir -p /home/yang1078/proj/WhoIsWho/log/hgt

python hgt/export_embeddings.py \
    --graph-csv "${HGT_GRAPH_CSV:-/projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/pub_data_graph.csv}" \
    --checkpoint "${HGT_CHECKPOINT:-hgt/output/fdh_hgt/best_model.pt}" \
    --train-author "${HGT_TRAIN_AUTHOR:-/projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/author_data.json}" \
    --eval-author "${HGT_EVAL_AUTHOR:-/projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/eval_author.json}" \
    --split-date "${HGT_SPLIT_DATE:-2018-09-01}" \
    --output-path "${HGT_EXPORT_PATH:-/projects/whoiswho/model/graph/graphembedding_fdh_hgt2.pkl}" \
    --hidden-dim 769 \
    > /home/yang1078/proj/WhoIsWho/log/hgt/export_fdh.log 2>&1