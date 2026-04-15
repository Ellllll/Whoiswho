#!/usr/bin/env python

import argparse
import glob
import json
import pickle
import random
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pandas.core.indexes.numeric" and name in {"Int64Index", "UInt64Index", "Float64Index"}:
            return pd.Index
        return super().find_class(module, name)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Fraud Detection Handbook PKL files into MIND JSON inputs.")
    parser.add_argument(
        "--input-dir",
        default="data/fdh/simulated-data-raw/data",
        help="Directory containing daily PKL files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/fdh/mind_format",
        help="Directory to write author_data.json, eval_data.json and pub_data.json.",
    )
    parser.add_argument(
        "--split-date",
        default="2018-09-01",
        help="Transactions before this date go to train author_data; on/after go to eval_data.",
    )
    parser.add_argument(
        "--min-tx-per-customer",
        type=int,
        default=20,
        help="Keep only customers with at least this many transactions overall.",
    )
    parser.add_argument(
        "--min-fraud-per-customer",
        type=int,
        default=1,
        help="Keep only customers with at least this many fraud transactions overall.",
    )
    parser.add_argument(
        "--max-customers",
        type=int,
        default=500,
        help="Maximum number of customers to export. Lower this if pub_data.json becomes too large.",
    )
    parser.add_argument(
        "--max-normal-per-customer",
        type=int,
        default=300,
        help="Cap the number of non-fraud transactions exported per customer across train+eval.",
    )
    parser.add_argument(
        "--max-fraud-per-customer",
        type=int,
        default=80,
        help="Cap the number of fraud transactions exported per customer across train+eval.",
    )
    parser.add_argument(
        "--packing-size",
        type=int,
        default=10,
        help="Used to filter train authors that would produce zero batches in INDPacking.",
    )
    parser.add_argument(
        "--train-min-fraud-count",
        type=int,
        default=3,
        help="Minimum fraud transaction count required for a customer to remain in train. 3 means keep fraud > 2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for customer selection when truncation is required.",
    )
    parser.add_argument(
        "--eval-positive-negative-customer-ratio",
        type=float,
        default=3.0,
        help="Target positive:negative customer ratio in eval after keeping only train-seen customers. Example 3.0 means about 3:1.",
    )
    return parser.parse_args()


def iter_frames(input_dir):
    for path in sorted(glob.glob(str(Path(input_dir) / "*.pkl"))):
        with open(path, "rb") as file_obj:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                yield path, CompatUnpickler(file_obj).load()


def build_customer_stats(input_dir):
    stats = defaultdict(lambda: {"tx": 0, "fraud": 0})
    for _, frame in iter_frames(input_dir):
        grouped = frame.groupby("CUSTOMER_ID").agg(tx=("TRANSACTION_ID", "size"), fraud=("TX_FRAUD", "sum"))
        for customer_id, row in grouped.iterrows():
            stats[str(customer_id)]["tx"] += int(row["tx"])
            stats[str(customer_id)]["fraud"] += int(row["fraud"])
    return stats


def select_customers(stats, args):
    selected = [
        customer_id
        for customer_id, customer_stats in stats.items()
        if customer_stats["tx"] >= args.min_tx_per_customer and customer_stats["fraud"] >= args.min_fraud_per_customer
    ]
    selected.sort(key=lambda customer_id: (-stats[customer_id]["fraud"], -stats[customer_id]["tx"], customer_id))
    if args.max_customers and len(selected) > args.max_customers:
        random.seed(args.seed)
        head = selected[: args.max_customers // 2]
        tail_pool = selected[args.max_customers // 2 :]
        tail = random.sample(tail_pool, args.max_customers - len(head))
        selected = sorted(head + tail, key=lambda customer_id: (-stats[customer_id]["fraud"], -stats[customer_id]["tx"], customer_id))
    return set(selected)


def make_author_key(customer_id):
    return f"customer_{customer_id}"


def make_tx_key(transaction_id):
    return f"tx_{transaction_id}"


def build_title(row):
    timestamp = pd.to_datetime(row["TX_DATETIME"])
    return (
        f"customer transaction amount {float(row['TX_AMOUNT']):.2f} at terminal {row['TERMINAL_ID']} "
        f"on {timestamp.strftime('%Y-%m-%d %H:%M:%S')} weekday {timestamp.day_name().lower()} "
        f"hour {timestamp.hour}"
    )


def build_pub_record(row):
    timestamp = pd.to_datetime(row["TX_DATETIME"])
    customer_id = str(row["CUSTOMER_ID"])
    terminal_id = str(row["TERMINAL_ID"])
    amount = float(row["TX_AMOUNT"])
    return {
        "id": make_tx_key(row["TRANSACTION_ID"]),
        "title": build_title(row),
        "authors": [
            {
                "name": make_author_key(customer_id),
                "org": f"terminal_{terminal_id}",
            }
        ],
        "abstract": (
            f"transaction id {row['TRANSACTION_ID']}; customer {customer_id}; terminal {terminal_id}; "
            f"amount {amount:.2f}; "
            f"tx_time_seconds {int(row['TX_TIME_SECONDS'])}; tx_time_days {int(row['TX_TIME_DAYS'])}."
        ),
        "keywords": [
            f"terminal_{terminal_id}",
            f"day_{int(row['TX_TIME_DAYS'])}",
        ],
        "venue": f"terminal_{terminal_id}",
        "year": int(timestamp.year),
    }


def append_transaction(target, author_id, tx_id, is_fraud):
    bucket = "outliers" if is_fraud else "normal_data"
    target[author_id][bucket].append(tx_id)


def prune_train_authors(author_data, packing_size, train_min_fraud_count):
    kept = {}
    for author_id, payload in author_data.items():
        total = len(payload["normal_data"]) + len(payload["outliers"])
        if (
            payload["normal_data"]
            and len(payload["outliers"]) >= train_min_fraud_count
            and total >= packing_size
        ):
            kept[author_id] = payload
    return kept


def prune_eval_authors(eval_data):
    kept = {}
    for author_id, payload in eval_data.items():
        total = len(payload["normal_data"]) + len(payload["outliers"])
        if total > 0:
            kept[author_id] = payload
    return kept


def filter_eval_to_train_authors(eval_data, train_author_ids):
    return {author_id: payload for author_id, payload in eval_data.items() if author_id in train_author_ids}


def downsample_eval_negative_authors(eval_data, positive_negative_ratio, seed):
    if positive_negative_ratio <= 0:
        return eval_data

    positive_authors = []
    negative_authors = []
    for author_id, payload in eval_data.items():
        if len(payload["outliers"]) > 0:
            positive_authors.append(author_id)
        else:
            negative_authors.append(author_id)

    if not positive_authors or not negative_authors:
        return eval_data

    target_negative_count = min(len(negative_authors), int(len(positive_authors) / positive_negative_ratio))
    if target_negative_count >= len(negative_authors):
        return eval_data

    if target_negative_count <= 0:
        kept_negative_authors = set()
    else:
        random.seed(seed)
        kept_negative_authors = set(random.sample(negative_authors, target_negative_count))

    kept_positive_authors = set(positive_authors)
    kept_authors = kept_positive_authors | kept_negative_authors
    return {author_id: payload for author_id, payload in eval_data.items() if author_id in kept_authors}


def summarize_split(split_data):
    tx_count = 0
    fraud_count = 0
    positive_customers = 0
    negative_customers = 0

    for payload in split_data.values():
        normal_count = len(payload["normal_data"])
        outlier_count = len(payload["outliers"])
        tx_count += normal_count + outlier_count
        fraud_count += outlier_count
        if outlier_count > 0:
            positive_customers += 1
        else:
            negative_customers += 1

    return {
        "customers": len(split_data),
        "positive_customers": positive_customers,
        "negative_customers": negative_customers,
        "transactions": tx_count,
        "fraud_transactions": fraud_count,
        "fraud_rate": (fraud_count / tx_count) if tx_count else 0.0,
    }


def collect_referenced_ids(*datasets):
    referenced = set()
    for dataset in datasets:
        for payload in dataset.values():
            referenced.update(payload.get("normal_data", []))
            referenced.update(payload.get("outliers", []))
            referenced.update(payload.get("papers", []))
    return referenced


def main():
    args = parse_args()
    split_date = datetime.strptime(args.split_date, "%Y-%m-%d")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = build_customer_stats(args.input_dir)
    selected_customers = select_customers(stats, args)

    author_data = {}
    eval_data = {}
    train_normal_counts = defaultdict(int)
    eval_normal_counts = defaultdict(int)
    fraud_counts = defaultdict(int)
    pub_data = {}

    for customer_id in selected_customers:
        author_id = make_author_key(customer_id)
        author_data[author_id] = {"name": author_id, "normal_data": [], "outliers": []}
        eval_data[author_id] = {"name": author_id, "normal_data": [], "outliers": []}

    for _, frame in iter_frames(args.input_dir):
        frame = frame[frame["CUSTOMER_ID"].astype(str).isin(selected_customers)]
        if frame.empty:
            continue
        frame = frame.sort_values(["CUSTOMER_ID", "TX_DATETIME", "TRANSACTION_ID"])

        for _, row in frame.iterrows():
            customer_id = str(row["CUSTOMER_ID"])
            author_id = make_author_key(customer_id)
            is_fraud = int(row["TX_FRAUD"]) == 1
            is_eval = pd.to_datetime(row["TX_DATETIME"]).to_pydatetime() >= split_date

            if is_fraud:
                if args.max_fraud_per_customer and fraud_counts[author_id] >= args.max_fraud_per_customer:
                    continue
                fraud_counts[author_id] += 1
            else:
                if is_eval:
                    if args.max_normal_per_customer and eval_normal_counts[author_id] >= args.max_normal_per_customer:
                        continue
                    eval_normal_counts[author_id] += 1
                else:
                    if args.max_normal_per_customer and train_normal_counts[author_id] >= args.max_normal_per_customer:
                        continue
                    train_normal_counts[author_id] += 1

            tx_id = make_tx_key(row["TRANSACTION_ID"])
            pub_data[tx_id] = build_pub_record(row)
            target = eval_data if is_eval else author_data
            append_transaction(target, author_id, tx_id, is_fraud)

    author_data = prune_train_authors(author_data, args.packing_size, args.train_min_fraud_count)
    eval_data = prune_eval_authors(eval_data)
    train_author_ids = set(author_data.keys())
    eval_before_overlap = len(eval_data)
    eval_data = filter_eval_to_train_authors(eval_data, train_author_ids)
    eval_after_overlap = len(eval_data)
    eval_before_ratio = len(eval_data)
    eval_data = downsample_eval_negative_authors(
        eval_data,
        positive_negative_ratio=args.eval_positive_negative_customer_ratio,
        seed=args.seed,
    )
    eval_after_ratio = len(eval_data)
    referenced_ids = collect_referenced_ids(author_data, eval_data)
    pub_data = {tx_id: payload for tx_id, payload in pub_data.items() if tx_id in referenced_ids}

    train_summary = summarize_split(author_data)
    eval_summary = summarize_split(eval_data)

    summary = {
        "selected_customers": len(selected_customers),
        "train_authors": len(author_data),
        "eval_authors": len(eval_data),
        "pub_records": len(pub_data),
        "split_date": args.split_date,
        "max_customers": args.max_customers,
        "max_normal_per_customer": args.max_normal_per_customer,
        "max_fraud_per_customer": args.max_fraud_per_customer,
        "packing_size": args.packing_size,
        "train_min_fraud_count": args.train_min_fraud_count,
        "eval_positive_negative_customer_ratio": args.eval_positive_negative_customer_ratio,
        "eval_authors_before_overlap_filter": eval_before_overlap,
        "eval_authors_after_overlap_filter": eval_after_overlap,
        "eval_authors_before_ratio_filter": eval_before_ratio,
        "eval_authors_after_ratio_filter": eval_after_ratio,
        "train_summary": train_summary,
        "eval_summary": eval_summary,
    }

    with open(output_dir / "author_data.json", "w", encoding="utf-8") as file_obj:
        json.dump(author_data, file_obj, ensure_ascii=False)
    with open(output_dir / "eval_data.json", "w", encoding="utf-8") as file_obj:
        json.dump(eval_data, file_obj, ensure_ascii=False)
    with open(output_dir / "pub_data.json", "w", encoding="utf-8") as file_obj:
        json.dump(pub_data, file_obj, ensure_ascii=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()