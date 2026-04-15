import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hgt.dataset import FDHGlobalGraphData, filter_transactions, load_parsed_transactions
from hgt.model import HGTEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Export FDH HGT embeddings to stage3 graph format")
    parser.add_argument("--graph-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--train-author", required=True)
    parser.add_argument("--eval-author", required=True)
    parser.add_argument("--split-date", default="")
    parser.add_argument("--hidden-dim", type=int, default=769)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def parse_split_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def build_model(checkpoint, sample_graph, device):
    model = HGTEncoder(
        sample_graph.metadata(),
        in_dims=checkpoint["in_dims"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        heads=checkpoint["heads"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def encode_graph(model, graph_bundle, device):
    with torch.no_grad():
        graph = graph_bundle.data.to(device)
        embeddings = model(graph.x_dict, graph.edge_index_dict)
        return {
            "customer": embeddings["customer"].cpu(),
            "transaction": embeddings["transaction"].cpu(),
        }


def collect_train_transaction_ids(train_author):
    transaction_ids = set()
    for payload in train_author.values():
        transaction_ids.update(payload.get("normal_data", []))
        transaction_ids.update(payload.get("outliers", []))
    return transaction_ids


def collect_eval_transaction_ids(eval_author):
    transaction_ids = set()
    for payload in eval_author.values():
        transaction_ids.update(payload.get("papers", []))
    return transaction_ids


def build_local_prefix_transactions(current_transaction, history_by_customer, history_by_terminal):
    local_transactions = []
    seen_ids = set()

    for item in history_by_customer.get(current_transaction["customer_id"], []):
        if item["transaction_id"] in seen_ids:
            continue
        seen_ids.add(item["transaction_id"])
        local_transactions.append(item)
    for item in history_by_terminal.get(current_transaction["terminal_id"], []):
        if item["transaction_id"] in seen_ids:
            continue
        seen_ids.add(item["transaction_id"])
        local_transactions.append(item)

    if current_transaction["transaction_id"] not in seen_ids:
        local_transactions.append(current_transaction)
    return local_transactions


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    split_date_raw = args.split_date or checkpoint.get("split_date")
    if not split_date_raw:
        raise ValueError("split_date must be provided either as an argument or stored in the checkpoint.")
    split_datetime = parse_split_date(split_date_raw)

    train_author = load_json(args.train_author)
    eval_author = load_json(args.eval_author)
    parsed_transactions = load_parsed_transactions(args.graph_csv)
    train_transactions = filter_transactions(parsed_transactions, split_datetime=split_datetime, include_before_split=True)
    test_transactions = filter_transactions(parsed_transactions, split_datetime=split_datetime, include_before_split=False)

    train_graph_bundle = FDHGlobalGraphData(parsed_transactions=train_transactions)
    graph = train_graph_bundle.data
    if graph["customer"].x.shape[0] == 0:
        raise ValueError("No training graph could be built from the provided FDH data.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, train_graph_bundle.data, device)

    graph_embedding = {}
    train_transaction_ids = collect_train_transaction_ids(train_author)
    eval_transaction_ids = collect_eval_transaction_ids(eval_author)

    train_embeddings = encode_graph(model, train_graph_bundle, device)
    train_customer_index = {customer_id: idx for idx, customer_id in enumerate(train_graph_bundle.customer_ids)}
    train_transaction_index = {transaction_id: idx for idx, transaction_id in enumerate(train_graph_bundle.transaction_ids)}

    for customer_id in set(list(train_author.keys()) + list(eval_author.keys())):
        customer_idx = train_customer_index.get(customer_id)
        if customer_idx is not None:
            graph_embedding.setdefault(customer_id, {})["graph"] = train_embeddings["customer"][customer_idx].tolist()

    for transaction_id in train_transaction_ids:
        transaction_idx = train_transaction_index.get(transaction_id)
        if transaction_idx is None:
            continue
        customer_id = train_graph_bundle.transaction_customer_ids[transaction_idx]
        graph_embedding.setdefault(customer_id, {})[transaction_id] = train_embeddings["transaction"][transaction_idx].tolist()

    test_transactions_by_id = {item["transaction_id"]: item for item in test_transactions}
    history_by_customer = {}
    history_by_terminal = {}
    for item in train_transactions:
        history_by_customer.setdefault(item["customer_id"], []).append(item)
        history_by_terminal.setdefault(item["terminal_id"], []).append(item)

    ordered_eval_transactions = [
        test_transactions_by_id[transaction_id]
        for transaction_id in eval_transaction_ids
        if transaction_id in test_transactions_by_id
    ]
    ordered_eval_transactions.sort(key=lambda item: item["sort_key"])

    for item in ordered_eval_transactions:
        local_transactions = build_local_prefix_transactions(item, history_by_customer, history_by_terminal)
        local_graph_bundle = FDHGlobalGraphData(parsed_transactions=local_transactions)
        local_embeddings = encode_graph(model, local_graph_bundle, device)

        local_customer_index = {customer_id: idx for idx, customer_id in enumerate(local_graph_bundle.customer_ids)}
        local_transaction_index = {transaction_id: idx for idx, transaction_id in enumerate(local_graph_bundle.transaction_ids)}
        customer_id = item["customer_id"]
        transaction_id = item["transaction_id"]

        if "graph" not in graph_embedding.get(customer_id, {}):
            customer_idx = local_customer_index.get(customer_id)
            if customer_idx is not None:
                graph_embedding.setdefault(customer_id, {})["graph"] = local_embeddings["customer"][customer_idx].tolist()

        transaction_idx = local_transaction_index.get(transaction_id)
        if transaction_idx is not None:
            graph_embedding.setdefault(customer_id, {})[transaction_id] = local_embeddings["transaction"][transaction_idx].tolist()

        history_by_customer.setdefault(customer_id, []).append(item)
        history_by_terminal.setdefault(item["terminal_id"], []).append(item)

    missing_customers = [customer_id for customer_id, payload in eval_author.items() if "graph" not in graph_embedding.get(customer_id, {})]
    if missing_customers:
        print(f"warning: missing customer graph embeddings for {len(missing_customers)} eval customers", flush=True)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "wb") as file_obj:
        pickle.dump(graph_embedding, file_obj)


if __name__ == "__main__":
    main()