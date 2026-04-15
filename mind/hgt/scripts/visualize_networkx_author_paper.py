import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx


DEFAULT_AUTHOR_IDS = ["0ne0AhGR", "H4sLQRZO", "cnGw5I4m"]


def shorten(text: str, limit: int = 18) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_global_paper_to_authors(author_data):
    paper_to_authors = defaultdict(set)
    paper_to_roles = defaultdict(list)
    for author_id, info in author_data.items():
        for pid in info.get("normal_data", []):
            paper_to_authors[pid].add(author_id)
            paper_to_roles[pid].append((author_id, "normal"))
        for pid in info.get("outliers", []):
            paper_to_authors[pid].add(author_id)
            paper_to_roles[pid].append((author_id, "outlier"))
    return paper_to_authors, paper_to_roles


def sample_papers(info, max_papers, seed):
    import random

    normal = list(info.get("normal_data", []))
    outlier = list(info.get("outliers", []))
    rng = random.Random(seed)

    if len(normal) + len(outlier) <= max_papers:
        return list(dict.fromkeys(normal + outlier))

    normal_quota = min(len(normal), max_papers // 2)
    outlier_quota = min(len(outlier), max_papers - normal_quota)

    if normal_quota + outlier_quota < max_papers:
        remain = max_papers - (normal_quota + outlier_quota)
        extra_normal = min(len(normal) - normal_quota, remain)
        normal_quota += extra_normal
        remain -= extra_normal
        outlier_quota += min(len(outlier) - outlier_quota, remain)

    picked = rng.sample(normal, normal_quota) + rng.sample(outlier, outlier_quota)
    return list(dict.fromkeys(picked))


def build_graph(author_ids, author_data, pub_data, max_papers=30):
    paper_to_authors, paper_to_roles = build_global_paper_to_authors(author_data)
    graph = nx.Graph()
    selected_authors = set(author_ids)
    selected_papers = set()
    paper_status_for_focus = {}

    for idx, author_id in enumerate(author_ids):
        info = author_data[author_id]
        graph.add_node(author_id, node_type="author", label=info["name"], focus=True)
        sampled = sample_papers(info, max_papers=max_papers, seed=42 + idx)
        for pid in sampled:
            selected_papers.add(pid)
            if pid in info.get("normal_data", []):
                paper_status_for_focus[(author_id, pid)] = "normal"
            elif pid in info.get("outliers", []):
                paper_status_for_focus[(author_id, pid)] = "outlier"

    for pid in selected_papers:
        paper = pub_data.get(pid, {})
        title = paper.get("title", pid)
        graph.add_node(pid, node_type="paper", label=shorten(title, 20), focus=False)

        attached_authors = sorted(paper_to_authors.get(pid, []))
        for aid in attached_authors:
            if aid not in graph:
                graph.add_node(aid, node_type="author", label=author_data[aid]["name"], focus=False)
            edge_role = "shared"
            if (aid, pid) in paper_status_for_focus:
                edge_role = paper_status_for_focus[(aid, pid)]
            graph.add_edge(aid, pid, role=edge_role)

    return graph


def multipartite_layout_by_panels(graph, focus_author_ids):
    pos = {}
    panel_gap = 10.0

    # Put focus authors and their immediate papers/authors into visible panels.
    for panel_idx, focus_author in enumerate(focus_author_ids):
        x0 = panel_idx * panel_gap
        pos[focus_author] = (x0, 0.0)

        paper_neighbors = sorted([n for n in graph.neighbors(focus_author) if graph.nodes[n]["node_type"] == "paper"])
        ys = list(reversed([i - (len(paper_neighbors) - 1) / 2 for i in range(len(paper_neighbors))]))
        for pid, y in zip(paper_neighbors, ys):
            pos[pid] = (x0 + 3.0, y * 0.5)

        other_authors = sorted(
            {
                n2
                for pid in paper_neighbors
                for n2 in graph.neighbors(pid)
                if graph.nodes[n2]["node_type"] == "author" and n2 != focus_author
            }
        )
        ys2 = list(reversed([i - (len(other_authors) - 1) / 2 for i in range(len(other_authors))]))
        for aid, y in zip(other_authors, ys2):
            if aid not in pos:
                pos[aid] = (x0 + 6.0, y * 0.42)

    # Any remaining shared paper/author gets a fallback position.
    fallback_x = len(focus_author_ids) * panel_gap + 2.0
    leftovers = [n for n in graph.nodes if n not in pos]
    for idx, node in enumerate(leftovers):
        pos[node] = (fallback_x, idx * 0.35)

    return pos


def draw_graph(graph, focus_author_ids, output_path):
    pos = multipartite_layout_by_panels(graph, focus_author_ids)

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.axis("off")

    edge_colors = []
    edge_widths = []
    for u, v, data in graph.edges(data=True):
        role = data.get("role", "shared")
        if role == "normal":
            edge_colors.append("#4c78a8")
            edge_widths.append(2.4)
        elif role == "outlier":
            edge_colors.append("#e45756")
            edge_widths.append(2.4)
        else:
            edge_colors.append("#7f8c8d")
            edge_widths.append(1.5)

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.75)

    author_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "author"]
    paper_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "paper"]
    focus_authors = [n for n, d in graph.nodes(data=True) if d["node_type"] == "author" and d.get("focus")]
    non_focus_authors = [n for n in author_nodes if n not in focus_authors]

    nx.draw_networkx_nodes(graph, pos, nodelist=paper_nodes, node_color="#d9dde3", edgecolors="white", linewidths=1.0, node_size=900, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=non_focus_authors, node_color="#222222", edgecolors="white", linewidths=1.0, node_size=850, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=focus_authors, node_color="#111111", edgecolors="#f4d35e", linewidths=2.0, node_size=1050, ax=ax)

    labels = {n: graph.nodes[n]["label"] for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7.5, font_color="black", ax=ax)

    # Repaint author labels white for readability.
    for n in author_nodes:
        x, y = pos[n]
        ax.text(x, y, shorten(graph.nodes[n]["label"], 16), fontsize=7.5, ha="center", va="center", color="white", zorder=5)

    for panel_idx, focus_author in enumerate(focus_author_ids):
        x0 = panel_idx * 10.0
        ax.text(x0, 8.2, f"focus author {panel_idx + 1}", fontsize=10, ha="center", color="#666666")
        ax.text(x0 + 3.0, 8.2, "papers", fontsize=10, ha="center", color="#666666")
        ax.text(x0 + 6.0, 8.2, "authors", fontsize=10, ha="center", color="#666666")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#111111", markeredgecolor="#f4d35e", markersize=11, label="focus author"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#222222", markersize=10, label="author"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d9dde3", markersize=10, label="paper"),
        plt.Line2D([0], [0], color="#4c78a8", linewidth=2.4, label="normal edge"),
        plt.Line2D([0], [0], color="#e45756", linewidth=2.4, label="outlier edge"),
        plt.Line2D([0], [0], color="#7f8c8d", linewidth=1.5, label="shared edge"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.suptitle(
        "NetworkX Author-Paper Graph\n"
        "Real many-to-many relations from train_author.json",
        fontsize=18,
        y=0.98,
    )
    fig.text(0.02, 0.02, "One paper can attach to multiple authors; this visualization preserves those shared-paper relations.", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize NetworkX author-paper graph from train_author.json.")
    parser.add_argument("--author-data", default="data/IND-WhoIsWho/train_author.json")
    parser.add_argument("--pub-data", default="data/IND-WhoIsWho/pid_to_info_all.json")
    parser.add_argument("--author-ids", nargs="*", default=DEFAULT_AUTHOR_IDS)
    parser.add_argument("--max-papers", type=int, default=30)
    parser.add_argument("--output", default="output/stage3_author_graphs/networkx_author_paper.png")
    args = parser.parse_args()

    with open(args.author_data, "r", encoding="utf-8") as f:
        author_data = json.load(f)
    with open(args.pub_data, "r", encoding="utf-8") as f:
        pub_data = json.load(f)

    graph = build_graph(args.author_ids, author_data, pub_data, max_papers=args.max_papers)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    draw_graph(graph, args.author_ids, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
