import argparse
import json
import os
import random
import sys

import torch
from torch_geometric.loader import LinkNeighborLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hgt.dataset import FDHGlobalGraphData
from hgt.model import HGTEncoder, edge_label_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train HGT on FDH customer-transaction-terminal graphs")
    parser.add_argument("--graph-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=769)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-neighbors", type=str, default="20,10")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--customer-transaction-weight", type=float, default=1.0)
    parser.add_argument("--transaction-terminal-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_num_neighbors(raw_value):
    return [int(value) for value in raw_value.split(",") if value.strip()]


def build_link_loader(data, edge_type, batch_size, num_neighbors, num_workers, shuffle=True):
    edge_index = data[edge_type].edge_index
    edge_label = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        edge_label_index=(edge_type, edge_index),
        edge_label=edge_label,
        neg_sampling_ratio=1.0,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    graph_bundle = FDHGlobalGraphData(args.graph_csv)
    graph = graph_bundle.data
    if graph["customer"].x.shape[0] == 0 or graph["transaction"].x.shape[0] == 0:
        raise ValueError("No global graph could be built from the provided FDH data.")
    sample_graph = graph
    in_dims = {node_type: sample_graph[node_type].x.shape[-1] for node_type in sample_graph.node_types}
    num_neighbors = parse_num_neighbors(args.num_neighbors)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HGTEncoder(
        sample_graph.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    relation_loaders = []
    if args.customer_transaction_weight > 0:
        relation_loaders.append(
            (
                "customer_transaction",
                args.customer_transaction_weight,
                ("customer", "owns", "transaction"),
                "customer",
                "transaction",
                build_link_loader(
                    graph,
                    ("customer", "owns", "transaction"),
                    batch_size=args.batch_size,
                    num_neighbors=num_neighbors,
                    num_workers=args.num_workers,
                ),
            )
        )
    if args.transaction_terminal_weight > 0:
        relation_loaders.append(
            (
                "transaction_terminal",
                args.transaction_terminal_weight,
                ("transaction", "at", "terminal"),
                "transaction",
                "terminal",
                build_link_loader(
                    graph,
                    ("transaction", "at", "terminal"),
                    batch_size=args.batch_size,
                    num_neighbors=num_neighbors,
                    num_workers=args.num_workers,
                ),
            )
        )
    if not relation_loaders:
        raise ValueError("At least one training relation must have a positive weight.")

    best_loss = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        skipped_steps = 0
        relation_metrics = {}
        relation_batch_counts = {}
        for relation_name, relation_weight, edge_type, source_type, target_type, loader in relation_loaders:
            relation_total = 0.0
            relation_batches = 0
            for batch_idx, batch in enumerate(loader, start=1):
                batch = batch.to(device)
                embeddings = model(batch.x_dict, batch.edge_index_dict)
                raw_loss = edge_label_loss(
                    embeddings,
                    batch[edge_type].edge_label_index,
                    batch[edge_type].edge_label,
                    source_type,
                    target_type,
                )
                if not torch.isfinite(raw_loss):
                    skipped_steps += 1
                    continue
                loss = relation_weight * raw_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1
                relation_total += raw_loss.item()
                relation_batches += 1

                if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                    running_relation_loss = relation_total / relation_batches
                    running_total_loss = total_loss / total_batches
                    print(
                        f"epoch={epoch} relation={relation_name} batch={batch_idx} "
                        f"relation_loss={running_relation_loss:.6f} total_loss={running_total_loss:.6f} "
                        f"skipped_steps={skipped_steps}",
                        flush=True,
                    )

            relation_metrics[relation_name] = relation_total / max(relation_batches, 1)
            relation_batch_counts[relation_name] = relation_batches

        average_loss = total_loss / max(total_batches, 1)
        history_entry = {
            "epoch": epoch,
            "loss": average_loss,
            "skipped_steps": skipped_steps,
        }
        history_entry.update(relation_metrics)
        history.append(history_entry)

        relation_summary = " ".join(
            f"{relation_name}={relation_loss:.6f}({relation_batch_counts[relation_name]} batches)"
            for relation_name, relation_loss in relation_metrics.items()
        )
        print(
            f"epoch={epoch} loss={average_loss:.6f} {relation_summary} skipped_steps={skipped_steps}",
            flush=True,
        )

        if best_loss is None or average_loss < best_loss:
            best_loss = average_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": sample_graph.metadata(),
                    "in_dims": in_dims,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "heads": args.heads,
                    "dropout": args.dropout,
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )

    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2)


if __name__ == "__main__":
    main()