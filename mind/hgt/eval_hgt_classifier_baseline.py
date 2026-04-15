import argparse
import json
import math
import os
import pickle
import random
import time
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a lightweight classifier on HGT embeddings")
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--train-author", required=True)
    parser.add_argument("--eval-author", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--negative-class-weight", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-mode", default="")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def dot_similarity(left: List[float], right: List[float]) -> float:
    return float(sum(left_value * right_value for left_value, right_value in zip(left, right)))


def l2_norm(vector: List[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(left: List[float], right: List[float]) -> float:
    denom = l2_norm(left) * l2_norm(right)
    if denom <= 1e-12:
        return 0.0
    return dot_similarity(left, right) / denom


def negative_l2_distance(left: List[float], right: List[float]) -> float:
    return -math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


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
    missing_authors = []
    missing_papers = []
    valid_sample_count = 0
    total_sample_count = 0

    total_weight = 0
    total_auc = 0.0
    total_ap = 0.0
    for author_id, labels in ground_truth.items():
        if author_id not in pred:
            missing_authors.append(author_id)
            continue

        normal_data = labels["normal_data"]
        outliers = labels["outliers"]
        total_sample_count += len(normal_data) + len(outliers)

        cur_labels = []
        cur_preds = []
        cur_weight = len(outliers)

        for pub_id in normal_data:
            if pub_id in pred[author_id]:
                cur_labels.append(1)
                cur_preds.append(pred[author_id][pub_id])
                valid_sample_count += 1
            else:
                missing_papers.append((author_id, pub_id))

        for pub_id in outliers:
            if pub_id in pred[author_id]:
                cur_labels.append(0)
                cur_preds.append(pred[author_id][pub_id])
                valid_sample_count += 1
            else:
                missing_papers.append((author_id, pub_id))

        if len(cur_preds) < 2 or len(set(cur_labels)) < 2:
            continue

        cur_auc = roc_auc_score(cur_labels, cur_preds)
        cur_map = average_precision_score(cur_labels, cur_preds)
        total_ap += cur_weight * cur_map
        total_auc += cur_weight * cur_auc
        total_weight += cur_weight

    metrics = {
        "total_sample_count": total_sample_count,
        "valid_sample_count": valid_sample_count,
        "missing_author_count": len(missing_authors),
        "missing_paper_count": len(missing_papers),
        "sample_valid_ratio": (valid_sample_count / total_sample_count) if total_sample_count else 0.0,
    }
    if total_weight == 0:
        metrics["AUC"] = 0.0
        metrics["MAP"] = 0.0
    else:
        metrics["AUC"] = total_auc / total_weight
        metrics["MAP"] = total_ap / total_weight
    return metrics


def build_preview(predictions, ground_truth, limit=20):
    preview = []
    for author_id, pub_scores in predictions.items():
        normal_pubs = set(ground_truth.get(author_id, {}).get("normal_data", []))
        outlier_pubs = set(ground_truth.get(author_id, {}).get("outliers", []))
        ranked_items = sorted(pub_scores.items(), key=lambda item: item[1], reverse=True)
        for pub_id, score in ranked_items:
            label = None
            if pub_id in normal_pubs:
                label = 1
            elif pub_id in outlier_pubs:
                label = 0
            preview.append({
                "author": author_id,
                "pub": pub_id,
                "score": score,
                "label": label,
            })
            if len(preview) >= limit:
                return preview
    return preview


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb(args, train_samples: int, eval_samples: int, input_dim: int):
    if not args.wandb_project:
        return None

    import wandb

    init_kwargs = {
        "project": args.wandb_project,
        "config": {
            "embedding_path": args.embedding_path,
            "train_author": args.train_author,
            "eval_author": args.eval_author,
            "ground_truth": args.ground_truth,
            "output_dir": args.output_dir,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "positive_class_weight": args.positive_class_weight,
            "negative_class_weight": args.negative_class_weight,
            "log_interval": args.log_interval,
            "seed": args.seed,
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "input_dim": input_dim,
        },
    }
    if args.wandb_name:
        init_kwargs["name"] = args.wandb_name
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def build_feature(author_embedding: List[float], paper_embedding: List[float]) -> torch.Tensor:
    author_tensor = torch.tensor(author_embedding, dtype=torch.float32)
    paper_tensor = torch.tensor(paper_embedding, dtype=torch.float32)
    abs_diff = torch.abs(author_tensor - paper_tensor)
    prod = author_tensor * paper_tensor
    similarity_features = torch.tensor(
        [
            cosine_similarity(author_embedding, paper_embedding),
            dot_similarity(author_embedding, paper_embedding),
            negative_l2_distance(author_embedding, paper_embedding),
        ],
        dtype=torch.float32,
    )
    return torch.cat([author_tensor, paper_tensor, abs_diff, prod, similarity_features], dim=0)


class HGTBinaryDataset(Dataset):
    def __init__(self, graph_embedding, author_payloads, include_labels=True):
        self.graph_embedding = graph_embedding
        self.include_labels = include_labels
        self.samples: List[Tuple[str, str, int]] = []
        for author_id, payload in author_payloads.items():
            if author_id not in self.graph_embedding or "graph" not in self.graph_embedding[author_id]:
                continue
            if include_labels:
                for pub_id in payload.get("normal_data", []):
                    if pub_id in self.graph_embedding[author_id]:
                        self.samples.append((author_id, pub_id, 1))
                for pub_id in payload.get("outliers", []):
                    if pub_id in self.graph_embedding[author_id]:
                        self.samples.append((author_id, pub_id, 0))
            else:
                for pub_id in payload.get("papers", []):
                    if pub_id in self.graph_embedding[author_id]:
                        self.samples.append((author_id, pub_id, -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        author_id, pub_id, label = self.samples[index]
        author_embedding = self.graph_embedding[author_id]["graph"]
        paper_embedding = self.graph_embedding[author_id][pub_id]
        feature = build_feature(author_embedding, paper_embedding)
        return {
            "features": feature,
            "label": torch.tensor(label, dtype=torch.float32),
            "author": author_id,
            "pub": pub_id,
        }


def collate_batch(batch):
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "author": [item["author"] for item in batch],
        "pub": [item["pub"] for item in batch],
    }


class HGTBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.embedding_path, "rb") as file_obj:
        graph_embedding = pickle.load(file_obj)

    train_author = load_json(args.train_author)
    eval_author = load_json(args.eval_author)
    ground_truth = load_json(args.ground_truth)

    train_dataset = HGTBinaryDataset(graph_embedding, train_author, include_labels=True)
    eval_dataset = HGTBinaryDataset(graph_embedding, eval_author, include_labels=False)
    if len(train_dataset) == 0:
        raise ValueError("No valid labeled training samples were built from the provided HGT embeddings and train author data.")
    if len(eval_dataset) == 0:
        raise ValueError("No valid evaluation samples were built from the provided HGT embeddings and eval author data.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    input_dim = train_dataset[0]["features"].shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HGTBinaryClassifier(input_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    wandb_run = init_wandb(args, len(train_dataset), len(eval_dataset), input_dim)

    labels = [sample[2] for sample in train_dataset.samples]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    positive_weight = torch.tensor(args.positive_class_weight, device=device)
    negative_weight = torch.tensor(args.negative_class_weight, device=device)

    print(
        json.dumps(
            {
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "device": str(device),
            },
            indent=2,
        ),
        flush=True,
    )
    if wandb_run is not None:
        wandb_run.summary["train_samples"] = len(train_dataset)
        wandb_run.summary["eval_samples"] = len(eval_dataset)
        wandb_run.summary["positive_count"] = positive_count
        wandb_run.summary["negative_count"] = negative_count
        wandb_run.summary["device"] = str(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        epoch_start_time = time.time()
        for batch_idx, batch in enumerate(train_loader, start=1):
            features = batch["features"].to(device)
            labels_tensor = batch["label"].to(device)
            logits = model(features)

            sample_weights = torch.where(labels_tensor > 0.5, positive_weight, negative_weight)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_tensor, weight=sample_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                average_loss = total_loss / total_batches
                print(
                    f"epoch={epoch} batch={batch_idx} loss={average_loss:.6f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/batch_loss": average_loss,
                            "train/epoch": epoch,
                            "train/batch": batch_idx,
                            "train/global_step": (epoch - 1) * len(train_loader) + batch_idx,
                        }
                    )

        epoch_train_loss = total_loss / max(total_batches, 1)
        epoch_runtime = time.time() - epoch_start_time
        print(f"epoch={epoch} train_loss={epoch_train_loss:.6f}", flush=True)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_loss": epoch_train_loss,
                    "train/epoch": epoch,
                    "train/epoch_runtime_sec": epoch_runtime,
                }
            )

    model.eval()
    predictions: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for batch in eval_loader:
            features = batch["features"].to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().tolist()
            for author_id, pub_id, score in zip(batch["author"], batch["pub"], probs):
                predictions.setdefault(author_id, {})[pub_id] = float(score)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "predict_res.json"), "w", encoding="utf-8") as file_obj:
        json.dump(predictions, file_obj)

    metrics = cal_auc_map(predictions, ground_truth)
    metrics.update(
        {
            "classifier": "mlp_binary_classifier",
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "positive_count": positive_count,
            "negative_count": negative_count,
        }
    )
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)
    if wandb_run is not None:
        wandb_run.log(
            {
                "eval/AUC": metrics["AUC"],
                "eval/MAP": metrics["MAP"],
                "eval/valid_sample_ratio": metrics["sample_valid_ratio"],
                "eval/missing_author_count": metrics["missing_author_count"],
                "eval/missing_paper_count": metrics["missing_paper_count"],
            }
        )
        wandb_run.summary["AUC"] = metrics["AUC"]
        wandb_run.summary["MAP"] = metrics["MAP"]

    preview = build_preview(predictions, ground_truth)
    with open(os.path.join(args.output_dir, "predict_preview.json"), "w", encoding="utf-8") as file_obj:
        json.dump(preview, file_obj, ensure_ascii=False, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        os.path.join(args.output_dir, "classifier.pt"),
    )

    print(json.dumps(metrics, indent=2), flush=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()