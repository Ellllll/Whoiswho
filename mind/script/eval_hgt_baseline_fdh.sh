set -ex

mkdir -p /home/yang1078/proj/WhoIsWho/log/hgt

python -u hgt/eval_hgt_baseline.py \
    --embedding-path /projects/whoiswho/model/graph/graphembedding_fdh_hgt.pkl \
    --eval-author /projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/eval_author.json \
    --ground-truth /projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/eval_data.json \
    --output-dir output/hgt_baseline/fdh \
    --score-type cosine \
    > /home/yang1078/proj/WhoIsWho/log/hgt/eval_hgt_baseline_fdh.log 2>&1