#!/usr/bin/env python

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a customer-terminal overlap graph from pub_data_graph.csv.")
    parser.add_argument(
        "--input-csv",
        default="data/fdh/mind_format_trainfgt2_eval3to1_norm300_split/pub_data_graph.csv",
        help="Graph-friendly CSV exported from the FDH-to-MIND pipeline.",
    )
    parser.add_argument(
        "--output-image",
        default="output/fdh_customer_terminal_overlap.png",
        help="Path to save the rendered PNG image.",
    )
    parser.add_argument(
        "--output-meta",
        default="output/fdh_customer_terminal_overlap.json",
        help="Path to save the selected customers and terminal summary.",
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        default=5,
        help="Number of customers to include in the overlap graph.",
    )
    parser.add_argument(
        "--max-terminals",
        type=int,
        default=25,
        help="Maximum number of shared terminal nodes to draw.",
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

        candidate_scores = []
        for candidate in remaining:
            pair_overlap = sum(len(customer_terminals[candidate] & customer_terminals[customer]) for customer in selected)
            union_overlap = len(customer_terminals[candidate] & selected_union)
            candidate_scores.append((pair_overlap, union_overlap, -len(customer_terminals[candidate]), candidate))

        candidate_scores.sort(reverse=True)
        selected_customer = candidate_scores[0][3]
        selected.append(selected_customer)
        remaining.remove(selected_customer)

    return selected


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_image = Path(args.output_image)
    output_meta = Path(args.output_meta)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    edge_weights = (
        df.groupby(["customer_key", "terminal_id"])
        .size()
        .reset_index(name="transaction_count")
    )

    customer_terminals = edge_weights.groupby("customer_key")["terminal_id"].apply(lambda values: set(map(int, values))).to_dict()
    selected_customers = choose_customers(customer_terminals, args.num_customers)

    sub_edges = edge_weights[edge_weights["customer_key"].isin(selected_customers)].copy()
    terminal_customer_counts = sub_edges.groupby("terminal_id")["customer_key"].nunique().reset_index(name="customer_count")
    shared_terminals = terminal_customer_counts[terminal_customer_counts["customer_count"] >= 2].copy()

    terminal_weights = sub_edges.groupby("terminal_id")["transaction_count"].sum().reset_index(name="total_transactions")
    shared_terminals = shared_terminals.merge(terminal_weights, on="terminal_id", how="left")
    shared_terminals = shared_terminals.sort_values(
        ["customer_count", "total_transactions", "terminal_id"],
        ascending=[False, False, True],
    )
    selected_terminals = shared_terminals.head(args.max_terminals)["terminal_id"].tolist()
    sub_edges = sub_edges[sub_edges["terminal_id"].isin(selected_terminals)].copy()

    graph = nx.Graph()
    for customer in selected_customers:
        graph.add_node(customer, kind="customer")
    for terminal in selected_terminals:
        graph.add_node(f"terminal_{terminal}", kind="terminal")

    for _, row in sub_edges.iterrows():
        graph.add_edge(
            row["customer_key"],
            f"terminal_{int(row['terminal_id'])}",
            weight=int(row["transaction_count"]),
        )

    customer_positions = {
        customer: (0.0, float(len(selected_customers) - index - 1))
        for index, customer in enumerate(selected_customers)
    }
    terminal_positions = {
        f"terminal_{terminal}": (3.5, float(len(selected_terminals) - index - 1))
        for index, terminal in enumerate(selected_terminals)
    }
    positions = customer_positions | terminal_positions

    plt.figure(figsize=(16, max(8, 0.35 * len(selected_terminals))))
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=selected_customers,
        node_color="#1f77b4",
        node_size=1800,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[f"terminal_{terminal}" for terminal in selected_terminals],
        node_color="#ffcc66",
        node_size=700,
        alpha=0.95,
    )

    edge_widths = [1.0 + 0.3 * graph.edges[edge]["weight"] for edge in graph.edges]
    nx.draw_networkx_edges(graph, positions, width=edge_widths, edge_color="#7f8c8d", alpha=0.7)
    nx.draw_networkx_labels(graph, positions, font_size=9)

    plt.title("FDH Customer-Terminal Overlap Subgraph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_image, dpi=200, bbox_inches="tight")
    plt.close()

    pairwise_overlap = []
    for customer_a, customer_b in combinations(selected_customers, 2):
        pairwise_overlap.append(
            {
                "customer_a": customer_a,
                "customer_b": customer_b,
                "shared_terminal_count": len(customer_terminals[customer_a] & customer_terminals[customer_b]),
            }
        )

    metadata = {
        "selected_customers": selected_customers,
        "selected_terminals": selected_terminals,
        "shared_terminal_count": len(selected_terminals),
        "pairwise_overlap": pairwise_overlap,
        "edge_count": graph.number_of_edges(),
        "image_path": str(output_image),
    }
    with open(output_meta, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()