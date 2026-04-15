import argparse
import json
import math
import os
import pickle
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pure HGT baseline on FDH customer-transaction ranking")
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--eval-author", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-type", choices=["cosine", "dot", "neg_l2"], default="cosine")
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


def compute_score(author_embedding: List[float], paper_embedding: List[float], score_type: str) -> float:
    if score_type == "cosine":
        return cosine_similarity(author_embedding, paper_embedding)
    if score_type == "dot":
        return dot_similarity(author_embedding, paper_embedding)
    return negative_l2_distance(author_embedding, paper_embedding)


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


def main():
    args = parse_args()

    with open(args.embedding_path, "rb") as file_obj:
        graph_embedding = pickle.load(file_obj)

    eval_author = load_json(args.eval_author)
    ground_truth = load_json(args.ground_truth)

    predictions = {}
    for author_id, payload in eval_author.items():
        if author_id not in graph_embedding or "graph" not in graph_embedding[author_id]:
            continue

        author_embedding = graph_embedding[author_id]["graph"]
        candidate_pubs = payload.get("papers", [])
        author_predictions = {}
        for pub_id in candidate_pubs:
            paper_embedding = graph_embedding.get(author_id, {}).get(pub_id)
            if paper_embedding is None:
                continue
            author_predictions[pub_id] = compute_score(author_embedding, paper_embedding, args.score_type)

        if author_predictions:
            predictions[author_id] = author_predictions

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "predict_res.json"), "w", encoding="utf-8") as file_obj:
        json.dump(predictions, file_obj)

    metrics = cal_auc_map(predictions, ground_truth)
    metrics["score_type"] = args.score_type
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)

    preview = build_preview(predictions, ground_truth)
    with open(os.path.join(args.output_dir, "predict_preview.json"), "w", encoding="utf-8") as file_obj:
        json.dump(preview, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()