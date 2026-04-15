import argparse
import json
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx


def shorten(text: str, limit: int = 22) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def choose_connected_papers(pub_data, num_papers=10, seed=42):
    rng = random.Random(seed)
    candidates = []
    for pid, paper in pub_data.items():
        authors = [a.get("name", "").strip() for a in paper.get("authors", []) if a.get("name", "").strip()]
        if 2 <= len(authors) <= 12:
            candidates.append((pid, authors, paper.get("title", pid)))

    rng.shuffle(candidates)
    author_counter = Counter()
    chosen = []

    for pid, authors, title in candidates:
        score = sum(1 for name in authors if author_counter[name] > 0)
        if not chosen or score > 0:
            chosen.append((pid, authors, title))
            author_counter.update(authors)
        if len(chosen) >= num_papers:
            break

    if len(chosen) < num_papers:
        for pid, authors, title in candidates:
            if (pid, authors, title) not in chosen:
                chosen.append((pid, authors, title))
            if len(chosen) >= num_papers:
                break

    return chosen[:num_papers]


def expand_with_related_papers(pub_data, chosen_papers, extra_papers=12):
    if extra_papers <= 0:
        return chosen_papers

    chosen_ids = {pid for pid, _, _ in chosen_papers}
    author_counter = Counter()
    for _, authors, _ in chosen_papers:
        author_counter.update(authors)

    # Prefer authors that already appear multiple times so the graph gains cycles/overlap.
    focus_authors = {name for name, count in author_counter.items() if count >= 2}
    if not focus_authors:
        focus_authors = set(author_counter.keys())

    candidates = []
    for pid, paper in pub_data.items():
        if pid in chosen_ids:
            continue
        authors = [a.get("name", "").strip() for a in paper.get("authors", []) if a.get("name", "").strip()]
        overlap = [name for name in authors if name in focus_authors]
        if overlap:
            score = sum(author_counter[name] for name in overlap)
            candidates.append((score, len(overlap), pid, authors, paper.get("title", pid)))

    # Highest overlap first, then more reused authors.
    candidates.sort(key=lambda x: (x[0], x[1], -len(x[3])), reverse=True)

    expanded = list(chosen_papers)
    for _, _, pid, authors, title in candidates[: extra_papers * 5]:
        if pid in chosen_ids:
            continue
        expanded.append((pid, authors, title))
        chosen_ids.add(pid)
        if len(expanded) >= len(chosen_papers) + extra_papers:
            break

    return expanded


def build_graph(chosen_papers):
    graph = nx.Graph()
    for pid, authors, title in chosen_papers:
        graph.add_node(pid, node_type="paper", label=shorten(title, 24))
        for author_name in authors:
            if author_name not in graph:
                graph.add_node(author_name, node_type="author", label=shorten(author_name, 20))
            graph.add_edge(author_name, pid)
    return graph


def component_packed_layout(graph):
    components = [graph.subgraph(nodes).copy() for nodes in nx.connected_components(graph)]
    components.sort(key=lambda g: g.number_of_nodes(), reverse=True)

    positions = {}
    x_cursor = 0.0
    gap = 4.2

    for idx, subg in enumerate(components):
        sub_pos = nx.spring_layout(subg, seed=42 + idx, k=2.0, iterations=500)
        xs = [xy[0] for xy in sub_pos.values()]
        ys = [xy[1] for xy in sub_pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y

        scale = 6.8 if subg.number_of_nodes() >= 10 else 5.2
        y_shift = -(min_y + max_y) / 2
        x_shift = x_cursor - min_x

        for node, (x, y) in sub_pos.items():
            positions[node] = ((x + x_shift) * scale, (y + y_shift) * scale)

        x_cursor += (width * scale) + gap

    return positions


def draw_graph(graph, output_path):
    pos = component_packed_layout(graph)

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.axis("off")

    author_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "author"]
    paper_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "paper"]

    nx.draw_networkx_edges(graph, pos, width=1.2, edge_color="#6b7280", alpha=0.48, ax=ax)
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=author_nodes,
        node_color="#111111",
        edgecolors="white",
        linewidths=0.9,
        node_size=520,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=paper_nodes,
        node_color="#d9dde3",
        edgecolors="#4b5563",
        linewidths=0.9,
        node_size=620,
        ax=ax,
    )

    paper_label_pos = {}
    for n in paper_nodes:
        x, y = pos[n]
        paper_label_pos[n] = (x, y + 0.065)
    nx.draw_networkx_labels(
        graph,
        paper_label_pos,
        labels={n: graph.nodes[n]["label"] for n in paper_nodes},
        font_size=7,
        font_color="black",
        ax=ax,
    )

    author_label_pos = {}
    for n in author_nodes:
        x, y = pos[n]
        author_label_pos[n] = (x, y - 0.07)
    nx.draw_networkx_labels(
        graph,
        author_label_pos,
        labels={n: graph.nodes[n]["label"] for n in author_nodes},
        font_size=7,
        font_color="#c1121f",
        ax=ax,
    )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#111111", markersize=11, label="author"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d9dde3", markeredgecolor="#4b5563", markersize=11, label="paper"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    ax.set_title("Author-Paper Topology from pid_to_info_all.json['authors']\n10 papers, no layered layout", fontsize=17, pad=18)
    fig.text(0.02, 0.02, "This graph uses the real coauthor lists from pid_to_info_all.json, not train_author.json ownership.", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize an author-paper topology directly from pid_to_info_all.json authors.")
    parser.add_argument("--pub-data", default="data/IND-WhoIsWho/pid_to_info_all.json")
    parser.add_argument("--num-papers", type=int, default=10)
    parser.add_argument("--extra-papers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/stage3_author_graphs/pid_authors_topology.png")
    args = parser.parse_args()

    with open(args.pub_data, "r", encoding="utf-8") as f:
        pub_data = json.load(f)

    chosen_papers = choose_connected_papers(pub_data, num_papers=args.num_papers, seed=args.seed)
    chosen_papers = expand_with_related_papers(pub_data, chosen_papers, extra_papers=args.extra_papers)
    graph = build_graph(chosen_papers)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    draw_graph(graph, args.output)
    print("chosen_papers:")
    for pid, authors, title in chosen_papers:
        print(pid, "|", len(authors), "authors |", title[:80])
    print(args.output)


if __name__ == "__main__":
    main()
