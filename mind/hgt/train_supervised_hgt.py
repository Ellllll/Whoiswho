import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import LinkNeighborLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hgt.dataset import FDHGlobalGraphData, filter_transactions, load_parsed_transactions
from hgt.model import HGTEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train supervised HGT on FDH customer-transaction pairs")
    parser.add_argument("--graph-csv", required=True)
    parser.add_argument("--train-author", required=True)
    parser.add_argument("--eval-author", default="")
    parser.add_argument("--ground-truth", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-date", required=True)
    parser.add_argument("--hidden-dim", type=int, default=769)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-neighbors", type=str, default="20,10")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--negative-class-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_num_neighbors(raw_value: str) -> List[int]:
    return [int(value) for value in raw_value.split(",") if value.strip()]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def parse_split_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def dot_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(dim=-1, keepdim=True)


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left, p=2, dim=-1)
    right = F.normalize(right, p=2, dim=-1)
    return (left * right).sum(dim=-1, keepdim=True)


def negative_l2_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return -torch.norm(left - right, p=2, dim=-1, keepdim=True)


def build_pair_features(customer_embeddings: torch.Tensor, transaction_embeddings: torch.Tensor) -> torch.Tensor:
    abs_diff = torch.abs(customer_embeddings - transaction_embeddings)
    prod = customer_embeddings * transaction_embeddings
    similarity_features = torch.cat(
        [
            cosine_similarity(customer_embeddings, transaction_embeddings),
            dot_similarity(customer_embeddings, transaction_embeddings),
            negative_l2_distance(customer_embeddings, transaction_embeddings),
        ],
        dim=-1,
    )
    return torch.cat(
        [customer_embeddings, transaction_embeddings, abs_diff, prod, similarity_features],
        dim=-1,
    )


class PairClassifier(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        feature_dim = hidden_dim * 4 + 3
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, customer_embeddings: torch.Tensor, transaction_embeddings: torch.Tensor) -> torch.Tensor:
        pair_features = build_pair_features(customer_embeddings, transaction_embeddings)
        return self.network(pair_features).squeeze(-1)


def build_labeled_pairs(author_payloads: Dict[str, Dict], customer_index: Dict[str, int], transaction_index: Dict[str, int]):
    customer_indices = []
    transaction_indices = []
    labels = []
    skipped_pairs = 0
    for customer_id, payload in author_payloads.items():
        customer_idx = customer_index.get(customer_id)
        if customer_idx is None:
            continue
        for transaction_id in payload.get("normal_data", []):
            transaction_idx = transaction_index.get(transaction_id)
            if transaction_idx is None:
                skipped_pairs += 1
                continue
            customer_indices.append(customer_idx)
            transaction_indices.append(transaction_idx)
            labels.append(1.0)
        for transaction_id in payload.get("outliers", []):
            transaction_idx = transaction_index.get(transaction_id)
            if transaction_idx is None:
                skipped_pairs += 1
                continue
            customer_indices.append(customer_idx)
            transaction_indices.append(transaction_idx)
            labels.append(0.0)

    if not labels:
        raise ValueError("No labeled customer-transaction pairs were built from the provided author data.")

    edge_label_index = torch.tensor([customer_indices, transaction_indices], dtype=torch.long)
    edge_label = torch.tensor(labels, dtype=torch.float32)
    return edge_label_index, edge_label, skipped_pairs


def build_loader(data, edge_label_index, edge_label, batch_size, num_neighbors, num_workers, shuffle):
    return LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        edge_label_index=(("customer", "owns", "transaction"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def roc_auc_score(labels: List[int], preds: List[float]) -> float:
    positives = [(pred, idx) for idx, (label, pred) in enumerate(zip(labels, preds)) if label == 1]
    negatives = [(pred, idx) for idx, (label, pred) in enumerate(zip(labels, preds)) if label == 0]
    if not positives or not negatives:
        return 0.0

    ranked = sorted(enumerate(preds), key=lambda item: item[1])
    rank_sum = 0.0
    rank = 1
    while rank <= len(ranked):
        start = rank - 1
        end = start
        while end + 1 < len(ranked) and ranked[end + 1][1] == ranked[start][1]:
            end += 1
        average_rank = (rank + end + 1) / 2.0
        for idx in range(start, end + 1):
            sample_index = ranked[idx][0]
            if labels[sample_index] == 1:
                rank_sum += average_rank
        rank = end + 2

    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def average_precision_score(labels: List[int], preds: List[float]) -> float:
    ranked = sorted(zip(preds, labels), key=lambda item: item[0], reverse=True)
    positive_total = sum(labels)
    if positive_total == 0:
        return 0.0

    true_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positive_total


def cal_auc_map(pred: Dict[str, Dict[str, float]], ground_truth: Dict[str, Dict[str, List[str]]]):
    total_weight = 0
    total_auc = 0.0
    total_ap = 0.0
    valid_sample_count = 0
    total_sample_count = 0
    missing_author_count = 0
    missing_paper_count = 0

    for author_id, labels in ground_truth.items():
        if author_id not in pred:
            missing_author_count += 1
            continue

        normal_data = labels["normal_data"]
        outliers = labels["outliers"]
        total_sample_count += len(normal_data) + len(outliers)
        cur_labels = []
        cur_preds = []
        cur_weight = len(outliers)

        for transaction_id in normal_data:
            if transaction_id in pred[author_id]:
                cur_labels.append(1)
                cur_preds.append(pred[author_id][transaction_id])
                valid_sample_count += 1
            else:
                missing_paper_count += 1
        for transaction_id in outliers:
            if transaction_id in pred[author_id]:
                cur_labels.append(0)
                cur_preds.append(pred[author_id][transaction_id])
                valid_sample_count += 1
            else:
                missing_paper_count += 1

        if len(cur_preds) < 2 or len(set(cur_labels)) < 2:
            continue

        total_auc += cur_weight * roc_auc_score(cur_labels, cur_preds)
        total_ap += cur_weight * average_precision_score(cur_labels, cur_preds)
        total_weight += cur_weight

    metrics = {
        "total_sample_count": total_sample_count,
        "valid_sample_count": valid_sample_count,
        "missing_author_count": missing_author_count,
        "missing_paper_count": missing_paper_count,
        "sample_valid_ratio": (valid_sample_count / total_sample_count) if total_sample_count else 0.0,
        "AUC": (total_auc / total_weight) if total_weight else 0.0,
        "MAP": (total_ap / total_weight) if total_weight else 0.0,
    }
    return metrics


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


def run_time_safe_eval(encoder, classifier, parsed_transactions, train_transactions, split_datetime, ground_truth, device):
    test_transactions = filter_transactions(parsed_transactions, split_datetime=split_datetime, include_before_split=False)
    test_transactions_by_id = {item["transaction_id"]: item for item in test_transactions}
    eval_transaction_ids = set()
    for payload in ground_truth.values():
        eval_transaction_ids.update(payload.get("normal_data", []))
        eval_transaction_ids.update(payload.get("outliers", []))

    ordered_eval_transactions = [
        test_transactions_by_id[transaction_id]
        for transaction_id in eval_transaction_ids
        if transaction_id in test_transactions_by_id
    ]
    ordered_eval_transactions.sort(key=lambda item: item["sort_key"])

    history_by_customer = {}
    history_by_terminal = {}
    for item in train_transactions:
        history_by_customer.setdefault(item["customer_id"], []).append(item)
        history_by_terminal.setdefault(item["terminal_id"], []).append(item)

    pred = {}
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        for item in ordered_eval_transactions:
            local_transactions = build_local_prefix_transactions(item, history_by_customer, history_by_terminal)
            local_graph_bundle = FDHGlobalGraphData(parsed_transactions=local_transactions)
            local_graph = local_graph_bundle.data.to(device)
            embeddings = encoder(local_graph.x_dict, local_graph.edge_index_dict)

            local_customer_index = {
                customer_id: idx for idx, customer_id in enumerate(local_graph_bundle.customer_ids)
            }
            local_transaction_index = {
                transaction_id: idx for idx, transaction_id in enumerate(local_graph_bundle.transaction_ids)
            }

            customer_id = item["customer_id"]
            transaction_id = item["transaction_id"]
            customer_idx = local_customer_index.get(customer_id)
            transaction_idx = local_transaction_index.get(transaction_id)
            if customer_idx is not None and transaction_idx is not None:
                customer_embedding = embeddings["customer"][customer_idx].unsqueeze(0)
                transaction_embedding = embeddings["transaction"][transaction_idx].unsqueeze(0)
                logit = classifier(customer_embedding, transaction_embedding)
                score = torch.sigmoid(logit).item()
                pred.setdefault(customer_id, {})[transaction_id] = score

            history_by_customer.setdefault(customer_id, []).append(item)
            history_by_terminal.setdefault(item["terminal_id"], []).append(item)

    return cal_auc_map(pred, ground_truth)


def build_checkpoint_payload(args, encoder, classifier, graph, in_dims, epoch, best_metric_name, best_metric_value):
    return {
        "model_state_dict": encoder.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "metadata": graph.metadata(),
        "in_dims": in_dims,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "training_objective": "supervised_pair_classification",
        "split_date": args.split_date,
        "epoch": epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    split_datetime = parse_split_date(args.split_date)
    parsed_transactions = load_parsed_transactions(args.graph_csv)
    train_transactions = filter_transactions(parsed_transactions, split_datetime=split_datetime, include_before_split=True)
    graph_bundle = FDHGlobalGraphData(parsed_transactions=train_transactions)
    graph = graph_bundle.data
    if graph["customer"].x.shape[0] == 0 or graph["transaction"].x.shape[0] == 0:
        raise ValueError("No global graph could be built from the provided FDH data.")

    customer_index = {customer_id: idx for idx, customer_id in enumerate(graph_bundle.customer_ids)}
    transaction_index = {transaction_id: idx for idx, transaction_id in enumerate(graph_bundle.transaction_ids)}
    train_author = load_json(args.train_author)
    train_edge_label_index, train_edge_label, skipped_pairs = build_labeled_pairs(
        train_author,
        customer_index,
        transaction_index,
    )

    num_neighbors = parse_num_neighbors(args.num_neighbors)
    train_loader = build_loader(
        graph,
        train_edge_label_index,
        train_edge_label,
        batch_size=args.batch_size,
        num_neighbors=num_neighbors,
        num_workers=args.num_workers,
        shuffle=True,
    )

    in_dims = {node_type: graph[node_type].x.shape[-1] for node_type in graph.node_types}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = HGTEncoder(
        graph.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    classifier = PairClassifier(args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    positive_weight = torch.tensor(args.positive_class_weight, device=device)
    negative_weight = torch.tensor(args.negative_class_weight, device=device)

    positive_count = int(train_edge_label.sum().item())
    negative_count = int(train_edge_label.shape[0] - positive_count)
    eval_ground_truth = load_json(args.ground_truth) if args.ground_truth else None
    print(
        json.dumps(
            {
                "train_pairs": int(train_edge_label.shape[0]),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "skipped_pairs": skipped_pairs,
                "split_date": args.split_date,
                "train_graph_transactions": len(train_transactions),
                "device": str(device),
            },
            indent=2,
        ),
        flush=True,
    )

    best_metric = None
    best_metric_name = "eval_AUC" if eval_ground_truth else "train_loss"
    history = []

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()
        total_loss = 0.0
        total_batches = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            embeddings = encoder(batch.x_dict, batch.edge_index_dict)
            edge_label_index = batch[("customer", "owns", "transaction")].edge_label_index
            labels = batch[("customer", "owns", "transaction")].edge_label.float()
            customer_embeddings = embeddings["customer"][edge_label_index[0]]
            transaction_embeddings = embeddings["transaction"][edge_label_index[1]]
            logits = classifier(customer_embeddings, transaction_embeddings)
            sample_weights = torch.where(labels > 0.5, positive_weight, negative_weight)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=sample_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), args.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                print(
                    f"epoch={epoch} batch={batch_idx} loss={total_loss / total_batches:.6f}",
                    flush=True,
                )

        epoch_loss = total_loss / max(total_batches, 1)
        history_entry = {
            "epoch": epoch,
            "train_loss": epoch_loss,
        }
        print(f"epoch={epoch} train_loss={epoch_loss:.6f}", flush=True)

        if eval_ground_truth:
            eval_metrics = run_time_safe_eval(
                encoder=encoder,
                classifier=classifier,
                parsed_transactions=parsed_transactions,
                train_transactions=train_transactions,
                split_datetime=split_datetime,
                ground_truth=eval_ground_truth,
                device=device,
            )
            history_entry["eval"] = eval_metrics
            print(
                "epoch={epoch} eval_AUC={auc:.6f} eval_MAP={map_score:.6f} eval_valid_ratio={ratio:.6f}".format(
                    epoch=epoch,
                    auc=eval_metrics["AUC"],
                    map_score=eval_metrics["MAP"],
                    ratio=eval_metrics["sample_valid_ratio"],
                ),
                flush=True,
            )
            score_to_track = eval_metrics["AUC"]
        else:
            score_to_track = -epoch_loss

        history.append(history_entry)
        epoch_checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(
            build_checkpoint_payload(
                args=args,
                encoder=encoder,
                classifier=classifier,
                graph=graph,
                in_dims=in_dims,
                epoch=epoch,
                best_metric_name=best_metric_name,
                best_metric_value=score_to_track,
            ),
            epoch_checkpoint_path,
        )

        is_better = best_metric is None or score_to_track > best_metric
        if is_better:
            best_metric = score_to_track
            torch.save(
                build_checkpoint_payload(
                    args=args,
                    encoder=encoder,
                    classifier=classifier,
                    graph=graph,
                    in_dims=in_dims,
                    epoch=epoch,
                    best_metric_name=best_metric_name,
                    best_metric_value=score_to_track,
                ),
                os.path.join(args.output_dir, "best_model.pt"),
            )

    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2)


if __name__ == "__main__":
    main()