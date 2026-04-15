# HGT For FDH

This directory contains a minimal HGT pipeline for the FDH dataset using three node types:

- `customer`
- `transaction`
- `terminal`

The design matches the current stage3 graph interface in `mind/utils.py`:

- `graph_emb[customer_id][transaction_id]` stores one transaction-level graph embedding
- `graph_emb[customer_id]["graph"]` stores one customer-level graph embedding

The exported embedding size defaults to `769`, so stage3 can concatenate:

- transaction graph embedding: `769`
- customer graph embedding: `769`

into the current `1538`-dimensional graph input expected by `graph_proj`.

## What This Pipeline Does

1. Builds one global hetero graph directly from `pub_data_graph.csv`.
2. Uses HGT to encode `customer`, `transaction`, and `terminal` nodes.
3. Trains the HGT encoder with supervised `customer-transaction` pair classification using `normal_data` and `outliers` from `author_data.json`.
4. Encodes the training split once to export all train-side customer and transaction embeddings.
5. Exports test-side transaction embeddings in chronological order using a history-prefix local graph, so earlier test transactions do not see later test transactions.
6. Merges train and test embeddings into one stage3-compatible `pkl`.

## Files

- `dataset.py`: global graph building and feature normalization
- `model.py`: HGT encoder and link decoder
- `train_hgt.py`: original sampled self-supervised link prediction training
- `train_supervised_hgt.py`: supervised HGT training on labeled customer-transaction pairs
- `export_embeddings.py`: export embeddings to stage3 format

## Environment

This code expects:

- `torch`
- `torch-geometric`

The current `mind` requirements do not declare `torch-geometric`, so you may need to install it in the active environment first.

## Example

Train supervised HGT:

```bash
cd /home/yang1078/proj/WhoIsWho/mind
python hgt/train_supervised_hgt.py \
  --graph-csv /projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/pub_data_graph.csv \
  --train-author /projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/author_data.json \
  --split-date 2018-09-01 \
  --output-dir hgt/output/fdh_hgt \
  --hidden-dim 769 \
  --batch-size 4096 \
  --epochs 20
```

Export embeddings for both train and eval customers:

```bash
cd /home/yang1078/proj/WhoIsWho/mind
python hgt/export_embeddings.py \
  --graph-csv /projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/pub_data_graph.csv \
  --checkpoint hgt/output/fdh_hgt/best_model.pt \
  --train-author /projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/author_data.json \
  --eval-author /projects/whoiswho/data/finance/fdh/mind_format_trainfgt2_eval3to1_norm300_split/eval_author.json \
  --split-date 2018-09-01 \
  --output-path /projects/whoiswho/model/graph/graphembedding_fdh_hgt.pkl \
  --hidden-dim 769
```

Or use the helper scripts:

```bash
cd /home/yang1078/proj/WhoIsWho/mind
bash script/train_hgt_fdh.sh
bash script/export_hgt_fdh.sh
bash script/eval_hgt_baseline_fdh.sh
```

The helper scripts always write logs to:

- `/home/yang1078/proj/WhoIsWho/log/hgt/train_fdh.log`
- `/home/yang1078/proj/WhoIsWho/log/hgt/export_fdh.log`
- `/home/yang1078/proj/WhoIsWho/log/hgt/stage3_fdh.log`

## FDH Stage3 Integration

`configs/llama3/stage3_fdh.json` now expects:

```text
/projects/whoiswho/model/graph/graphembedding_fdh_hgt.pkl
```

So the intended order is:

1. Train supervised HGT on the training split
2. Export merged train/test HGT embeddings with time-aware test inference
3. Run FDH stage3

## Pure HGT Baseline

If you want a pure graph-only baseline without the MIND/Llama stack, score each candidate transaction directly with the exported HGT embeddings:

```bash
cd /home/yang1078/proj/WhoIsWho/mind
bash script/eval_hgt_baseline_fdh.sh
```

This writes:

- `output/hgt_baseline/fdh/predict_res.json`
- `output/hgt_baseline/fdh/metrics.json`
- `output/hgt_baseline/fdh/predict_preview.json`

The default baseline score is cosine similarity between the customer graph embedding and the candidate transaction embedding.

## Temporal Handling

- Training uses the already time-split training partition as one graph; it does not do an additional rolling split inside the training period.
- Export uses the training graph for train embeddings.
- Test transaction embeddings are exported in chronological order and each test transaction only sees historical information plus the current transaction.

## Notes

- The scripts are designed to be launched from the `mind` directory.
- `pub_data_graph.csv` is now the only graph source file.
- `customer` comes from `customer_key`, `transaction` comes from `transaction_id`, and `terminal` comes from `terminal_id`.
- The current transaction features are derived from `tx_amount`, `tx_time_seconds`, `tx_time_days`, and `tx_datetime`.
- The pipeline is transductive over the CSV contents: train and export both cover all customers present in the CSV.
- The default training path is now supervised: the exported embeddings come from an HGT encoder trained to separate `normal_data` from `outliers` at the `customer-transaction` pair level.
- Input features are z-score normalized per node type before HGT training.