#!/usr/bin/env python

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
import random

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a customer-transaction-terminal tripartite graph.")
    parser.add_argument(
        "--input-csv",
        default="data/fdh/mind_format_trainfgt2_eval3to1_norm300_split/pub_data_structured.csv",
        help="Structured CSV with labels and split metadata.",
    )
    parser.add_argument(
        "--output-image",
        default="output/fdh_customer_transaction_terminal_tripartite.png",
        help="Path to save the rendered PNG image.",
    )
    parser.add_argument(
        "--output-meta",
        default="output/fdh_customer_transaction_terminal_tripartite.json",
        help="Path to save metadata about the selected nodes.",
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        default=5,
        help="Number of highly overlapping customers to visualize.",
    )
    parser.add_argument(
        "--max-normal-transactions-per-customer",
        type=int,
        default=10,
        help="Maximum sampled normal transactions per selected customer.",
    )
    parser.add_argument(
        "--max-anomaly-transactions-per-customer",
        type=int,
        default=6,
        help="Maximum sampled anomalous transactions per selected customer.",
    )
    parser.add_argument(
        "--max-terminals",
        type=int,
        default=18,
        help="Maximum number of terminals to draw.",
    )
    return parser.parse_args()


def choose_customers(customer_terminals, num_customers):
    customers = list(customer_terminals.keys())
    pair_scores = []
    for customer_a, customer_b in combinations(customers, 2):
        overlap = customer_terminals[customer_a] & customer_terminals[customer_b]
        if overlap:
            pair_scores.append((len(overlap), customer_a, customer_b))

    pair_scores.sort(reverse=True)
    if not pair_scores:
        return customers[:num_customers]

    _, customer_a, customer_b = pair_scores[0]
    selected = [customer_a, customer_b]
    remaining = [customer for customer in customers if customer not in selected]

    while len(selected) < num_customers and remaining:
        selected_union = set()
        for customer in selected:
            selected_union |= customer_terminals[customer]

        scored = []
        for candidate in remaining:
            pair_overlap = sum(len(customer_terminals[candidate] & customer_terminals[c]) for c in selected)
            union_overlap = len(customer_terminals[candidate] & selected_union)
            scored.append((pair_overlap, union_overlap, -len(customer_terminals[candidate]), candidate))
        scored.sort(reverse=True)
        selected_customer = scored[0][3]
        selected.append(selected_customer)
        remaining.remove(selected_customer)

    return selected


def sample_transactions(
    df,
    selected_customers,
    max_normal_transactions_per_customer,
    max_anomaly_transactions_per_customer,
    max_terminals,
):
    subset = df[df["customer_key"].isin(selected_customers)].copy()
    terminal_customer_counts = subset.groupby("terminal_id")["customer_key"].nunique().to_dict()
    sampled_parts = []

    for customer in selected_customers:
        customer_rows = subset[subset["customer_key"] == customer].copy()
        anomaly_rows = customer_rows[customer_rows["tx_fraud"] == 1].copy()
        normal_rows = customer_rows[customer_rows["tx_fraud"] == 0].copy()

        if not anomaly_rows.empty:
            anomaly_rows["shared_score"] = anomaly_rows["terminal_id"].map(lambda terminal: terminal_customer_counts.get(terminal, 0))
            anomaly_rows = anomaly_rows.sort_values(
                ["shared_score", "tx_amount", "transaction_id"],
                ascending=[False, False, True],
            )
            anomaly_rows = anomaly_rows.head(max_anomaly_transactions_per_customer)

        if not normal_rows.empty:
            normal_rows["shared_score"] = normal_rows["terminal_id"].map(lambda terminal: terminal_customer_counts.get(terminal, 0))
            normal_rows = normal_rows.sort_values(
                ["shared_score", "tx_amount", "transaction_id"],
                ascending=[False, False, True],
            )
            normal_rows = normal_rows.head(max_normal_transactions_per_customer)

        sampled_parts.append(anomaly_rows)
        sampled_parts.append(normal_rows)

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.drop_duplicates(subset=["tx_key"]).copy()
    terminal_usage = sampled.groupby("terminal_id")["tx_key"].nunique().reset_index(name="tx_count")
    terminal_usage = terminal_usage.sort_values(["tx_count", "terminal_id"], ascending=[False, True])
    selected_terminals = terminal_usage.head(max_terminals)["terminal_id"].tolist()
    sampled = sampled[sampled["terminal_id"].isin(selected_terminals)].copy()
    return sampled, selected_terminals


def build_graph(sampled, selected_customers, selected_terminals):
    graph = nx.Graph()
    for customer in selected_customers:
        graph.add_node(customer, kind="customer")
    for terminal in selected_terminals:
        graph.add_node(f"terminal_{int(terminal)}", kind="terminal")

    for _, row in sampled.iterrows():
        tx_node = row["tx_key"]
        graph.add_node(tx_node, kind="transaction", is_anomaly=bool(row["tx_fraud"]))
        graph.add_edge(row["customer_key"], tx_node)
        graph.add_edge(tx_node, f"terminal_{int(row['terminal_id'])}")

    return graph


def sample_points_in_ellipse(nodes, center_x, center_y, radius_x, radius_y, min_distance, rng):
    positions = {}
    for node in nodes:
        for _ in range(2000):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = math.sqrt(rng.uniform(0.0, 1.0))
            x = center_x + radius_x * radius * math.cos(angle)
            y = center_y + radius_y * radius * math.sin(angle)
            if all(math.dist((x, y), existing_pos) >= min_distance for existing_pos in positions.values()):
                positions[node] = (x, y)
                break
        else:
            positions[node] = (center_x, center_y)
    return positions


def sample_points_on_ellipse_ring(nodes, center_x, center_y, radius_x, radius_y, jitter_ratio, rng):
    if not nodes:
        return {}

    base_angles = [2.0 * math.pi * index / len(nodes) for index in range(len(nodes))]
    rng.shuffle(base_angles)
    positions = {}
    for node, base_angle in zip(nodes, base_angles):
        angle = base_angle + rng.uniform(-math.pi / len(nodes), math.pi / len(nodes))
        scale = 1.0 + rng.uniform(-jitter_ratio, jitter_ratio)
        positions[node] = (
            center_x + radius_x * scale * math.cos(angle),
            center_y + radius_y * scale * math.sin(angle),
        )
    return positions


def draw_graph(graph, sampled, selected_customers, selected_terminals, output_image):
    rng = random.Random(42)
    terminal_transaction_distance_scale = 3.0
    customer_positions = sample_points_on_ellipse_ring(
        selected_customers,
        center_x=0.0,
        center_y=0.0,
        radius_x=7.6,
        radius_y=5.8,
        jitter_ratio=0.08,
        rng=rng,
    )
    terminal_nodes = [f"terminal_{int(terminal)}" for terminal in selected_terminals]
    terminal_positions = sample_points_in_ellipse(
        terminal_nodes,
        center_x=0.0,
        center_y=0.0,
        radius_x=2.8,
        radius_y=2.2,
        min_distance=0.85,
        rng=rng,
    )

    positions = customer_positions | terminal_positions
    grouped = sampled.sort_values(["terminal_id", "customer_key", "tx_fraud", "transaction_id"], ascending=[True, True, False, True])
    terminal_node_map = {int(terminal): f"terminal_{int(terminal)}" for terminal in selected_terminals}
    tx_rows_by_terminal = {}
    for _, row in grouped.iterrows():
        tx_rows_by_terminal.setdefault(int(row["terminal_id"]), []).append(row)

    for terminal_id, terminal_rows in tx_rows_by_terminal.items():
        terminal_node = terminal_node_map[terminal_id]
        terminal_pos = terminal_positions[terminal_node]
        total = len(terminal_rows)
        for index, row in enumerate(terminal_rows):
            tx_key = row["tx_key"]
            customer_pos = customer_positions[row["customer_key"]]
            base_angle = math.atan2(customer_pos[1] - terminal_pos[1], customer_pos[0] - terminal_pos[0])
            spread = 0.24 * ((index % 5) - 2)
            ring = index // 5
            radial_distance = terminal_transaction_distance_scale * (
                0.9 + 0.38 * ring + (0.2 if int(row["tx_fraud"]) == 1 else 0.0)
            )
            tangential_jitter = rng.uniform(-0.06, 0.06)
            angle = base_angle + spread + tangential_jitter
            crowd_shift = 0.18 * ((index + 0.5) / max(total, 1) - 0.5)
            positions[tx_key] = (
                terminal_pos[0] + radial_distance * math.cos(angle) + crowd_shift,
                terminal_pos[1] + radial_distance * math.sin(angle),
            )

    plt.figure(figsize=(18, 12))
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=selected_customers,
        node_color="#1f77b4",
        node_size=2200,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[node for node, data in graph.nodes(data=True) if data.get("kind") == "transaction" and data.get("is_anomaly")],
        node_color="#d62728",
        node_size=280,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[node for node, data in graph.nodes(data=True) if data.get("kind") == "transaction" and not data.get("is_anomaly")],
        node_color="#c7c7c7",
        node_size=180,
        alpha=0.9,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[f"terminal_{int(terminal)}" for terminal in selected_terminals],
        node_color="#ffcc66",
        node_size=900,
        alpha=0.95,
    )

    nx.draw_networkx_edges(graph, positions, width=1.0, edge_color="#95a5a6", alpha=0.45)

    nx.draw_networkx_labels(graph, positions, labels={customer: customer for customer in selected_customers}, font_size=10)
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={f"terminal_{int(terminal)}": f"terminal_{int(terminal)}" for terminal in selected_terminals},
        font_size=9,
    )
    anomaly_labels = {node: node for node, data in graph.nodes(data=True) if data.get("kind") == "transaction" and data.get("is_anomaly")}
    nx.draw_networkx_labels(graph, positions, labels=anomaly_labels, font_size=7)

    plt.title("FDH Customer-Transaction-Terminal Tripartite Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_image, dpi=200, bbox_inches="tight")
    plt.close()


def main(args):
    input_csv = Path(args.input_csv)
    output_image = Path(args.output_image)
    output_meta = Path(args.output_meta)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    customer_terminals = df.groupby("customer_key")["terminal_id"].apply(lambda values: set(map(int, values))).to_dict()
    selected_customers = choose_customers(customer_terminals, args.num_customers)
    sampled, selected_terminals = sample_transactions(
        df,
        selected_customers,
        args.max_normal_transactions_per_customer,
        args.max_anomaly_transactions_per_customer,
        args.max_terminals,
    )
    graph = build_graph(sampled, selected_customers, selected_terminals)
    draw_graph(graph, sampled, selected_customers, selected_terminals, output_image)

    customer_breakdown = []
    for customer in selected_customers:
        customer_rows = sampled[sampled["customer_key"] == customer]
        customer_breakdown.append(
            {
                "customer": customer,
                "sampled_transactions": int(len(customer_rows)),
                "sampled_anomalies": int(customer_rows["tx_fraud"].sum()),
                "sampled_normals": int((customer_rows["tx_fraud"] == 0).sum()),
            }
        )

    metadata = {
        "selected_customers": selected_customers,
        "selected_terminals": [int(terminal) for terminal in selected_terminals],
        "sampled_transaction_count": int(sampled["tx_key"].nunique()),
        "sampled_anomaly_count": int(sampled["tx_fraud"].sum()),
        "sampled_normal_count": int((sampled["tx_fraud"] == 0).sum()),
        "customer_breakdown": customer_breakdown,
        "edge_count": graph.number_of_edges(),
        "image_path": str(output_image),
    }
    with open(output_meta, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)