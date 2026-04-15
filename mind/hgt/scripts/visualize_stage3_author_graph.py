import argparse
import json
import math
import os
from collections import Counter
import random

import matplotlib.pyplot as plt


DEFAULT_AUTHORS = [
    "0ne0AhGR",  # normal-dominant
    "H4sLQRZO",  # relatively balanced
    "cnGw5I4m",  # outlier-dominant
]


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def collect_coauthors(paper, author_name):
    target = normalize_name(author_name)
    coauthors = []
    for author in paper.get("authors", []):
        candidate = (author.get("name") or "").strip()
        if not candidate:
            continue
        if normalize_name(candidate) != target:
            coauthors.append(candidate)
    return coauthors


def shorten(text: str, limit: int = 26) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def circular_positions(nodes, radius, start_angle=0.0):
    if not nodes:
        return {}
    step = 2 * math.pi / len(nodes)
    return {
        node: (
            radius * math.cos(start_angle + idx * step),
            radius * math.sin(start_angle + idx * step),
        )
        for idx, node in enumerate(nodes)
    }


def build_author_graph(author_id, author_info, pub_info):
    author_name = author_info["name"]
    normal = set(author_info.get("normal_data", []))
    outliers = set(author_info.get("outliers", []))
    papers = list(normal | outliers)

    paper_nodes = []
    coauthor_nodes = set()
    paper_to_coauthors = {}

    for pid in papers:
        paper = pub_info.get(pid)
        if not paper:
            continue
        label = "normal" if pid in normal else "outlier"
        title = paper.get("title", pid)
        paper_nodes.append((pid, title, label))
        coauthors = collect_coauthors(paper, author_name)
        if coauthors:
            paper_to_coauthors[pid] = coauthors
            coauthor_nodes.update(coauthors)

    return {
        "author_id": author_id,
        "author_name": author_name,
        "paper_nodes": sorted(paper_nodes, key=lambda x: (x[2], x[1].lower())),
        "coauthor_nodes": sorted(coauthor_nodes),
        "paper_to_coauthors": paper_to_coauthors,
    }


def sample_author_graph(graph_data, max_papers=30, seed=42):
    paper_nodes = graph_data["paper_nodes"]
    if len(paper_nodes) <= max_papers:
        kept = paper_nodes
    else:
        normal = [node for node in paper_nodes if node[2] == "normal"]
        outlier = [node for node in paper_nodes if node[2] == "outlier"]
        rng = random.Random(seed)

        normal_quota = min(len(normal), max_papers // 2)
        outlier_quota = min(len(outlier), max_papers - normal_quota)

        if normal_quota + outlier_quota < max_papers:
            remaining = max_papers - (normal_quota + outlier_quota)
            extra_normal = min(len(normal) - normal_quota, remaining)
            normal_quota += extra_normal
            remaining -= extra_normal
            outlier_quota += min(len(outlier) - outlier_quota, remaining)

        kept = rng.sample(normal, normal_quota) + rng.sample(outlier, outlier_quota)
        kept = sorted(kept, key=lambda x: (x[2], x[1].lower()))

    kept_ids = {pid for pid, _, _ in kept}
    kept_paper_to_coauthors = {pid: names for pid, names in graph_data["paper_to_coauthors"].items() if pid in kept_ids}
    kept_coauthors = sorted({name for names in kept_paper_to_coauthors.values() for name in names})

    sampled = dict(graph_data)
    sampled["paper_nodes"] = kept
    sampled["coauthor_nodes"] = kept_coauthors
    sampled["paper_to_coauthors"] = kept_paper_to_coauthors
    return sampled


def draw_author_graph(graph_data, output_path):
    author_name = graph_data["author_name"]
    paper_nodes = graph_data["paper_nodes"]
    coauthor_nodes = graph_data["coauthor_nodes"]
    paper_to_coauthors = graph_data["paper_to_coauthors"]

    fig, ax = plt.subplots(figsize=(18, 18))
    ax.set_aspect("equal")
    ax.axis("off")

    author_pos = {"author": (0.0, 0.0)}
    paper_ids = [pid for pid, _, _ in paper_nodes]
    coauthor_ids = coauthor_nodes

    paper_pos = circular_positions(paper_ids, radius=5.5, start_angle=math.pi / 2)
    coauthor_pos = circular_positions(coauthor_ids, radius=9.5, start_angle=math.pi / 2)

    # Draw edges first so nodes stay on top.
    for pid in paper_ids:
        px, py = paper_pos[pid]
        ax.plot(
            [author_pos["author"][0], px],
            [author_pos["author"][1], py],
            color="#3a4147",
            linewidth=1.8,
            alpha=0.75,
            zorder=1,
        )
        for coauthor in paper_to_coauthors.get(pid, []):
            if coauthor in coauthor_pos:
                ox, oy = coauthor_pos[coauthor]
                ax.plot(
                    [px, ox],
                    [py, oy],
                    color="#6b7280",
                    linewidth=1.35,
                    alpha=0.6,
                    zorder=1,
                )

    # Draw author nodes.
    for coauthor in coauthor_ids:
        x, y = coauthor_pos[coauthor]
        ax.scatter([x], [y], s=240, color="#222222", edgecolors="white", linewidths=1.0, zorder=3)
        ax.text(x, y, shorten(coauthor, 22), fontsize=7, ha="center", va="center", color="white", zorder=4)

    # Draw paper nodes.
    color_map = {"normal": "#4c78a8", "outlier": "#e45756"}
    for pid, title, label in paper_nodes:
        x, y = paper_pos[pid]
        ax.scatter([x], [y], s=130, color=color_map[label], edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(x, y, shorten(title, 20), fontsize=6.5, ha="center", va="center", color="white", zorder=4)

    # Draw author node at center.
    ax.scatter([0], [0], s=900, color="#222222", edgecolors="white", linewidths=1.5, zorder=5)
    ax.text(0, 0, shorten(author_name, 18), fontsize=11, ha="center", va="center", color="white", zorder=6)

    normal_count = sum(1 for _, _, label in paper_nodes if label == "normal")
    outlier_count = sum(1 for _, _, label in paper_nodes if label == "outlier")
    title = (
        f"Author- Paper- Org Local Graph\n"
        f"{author_name} ({graph_data['author_id']}) | papers={len(paper_nodes)} "
        f"| normal={normal_count} | outlier={outlier_count} | authors={len(coauthor_ids) + 1}"
    )
    ax.set_title(title, fontsize=16, pad=18)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#222222", markersize=12, label="author"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8", markersize=10, label="normal paper"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e45756", markersize=10, label="outlier paper"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    note = (
        "Node types in this figure: author and paper. Edge types: author--paper.\n"
        "paper--refpaper is not drawn because IND-WhoIsWho does not contain reference-edge data."
    )
    fig.text(0.02, 0.02, note, fontsize=10, color="#444444")

    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_combined_author_graph(graph_list, output_path):
    fig, ax = plt.subplots(figsize=(24, 12))
    ax.axis("off")

    panel_centers = [-14.0, 0.0, 14.0]
    color_map = {"normal": "#4c78a8", "outlier": "#e45756"}

    for idx, graph_data in enumerate(graph_list):
        cx = panel_centers[idx]
        author_name = graph_data["author_name"]
        paper_nodes = graph_data["paper_nodes"]
        coauthor_nodes = graph_data["coauthor_nodes"]
        paper_to_coauthors = graph_data["paper_to_coauthors"]

        author_pos = (cx - 4.8, 0.0)
        paper_x = cx
        coauthor_x = cx + 4.8

        paper_sorted = sorted(paper_nodes, key=lambda x: (x[2], x[1].lower()))
        paper_ys = list(reversed([i - (len(paper_sorted) - 1) / 2 for i in range(len(paper_sorted))]))
        paper_pos = {pid: (paper_x, y * 0.45) for (pid, _, _), y in zip(paper_sorted, paper_ys)}

        coauthor_sorted = sorted(coauthor_nodes)
        coauthor_ys = list(reversed([i - (len(coauthor_sorted) - 1) / 2 for i in range(len(coauthor_sorted))]))
        coauthor_pos = {name: (coauthor_x, y * 0.38) for name, y in zip(coauthor_sorted, coauthor_ys)}

        # Subtle panel guide.
        ax.plot([cx - 7.2, cx - 7.2], [-7.5, 7.5], color="#e5e7eb", linewidth=1.0, zorder=0)
        ax.plot([cx + 7.2, cx + 7.2], [-7.5, 7.5], color="#e5e7eb", linewidth=1.0, zorder=0)
        ax.text(cx - 4.8, 7.9, "author", fontsize=10, ha="center", color="#666666")
        ax.text(cx, 7.9, "papers", fontsize=10, ha="center", color="#666666")
        ax.text(cx + 4.8, 7.9, "authors", fontsize=10, ha="center", color="#666666")

        for pid in [pid for pid, _, _ in paper_sorted]:
            px, py = paper_pos[pid]
            ax.plot(
                [author_pos[0], px],
                [author_pos[1], py],
                color="#2f363d",
                linewidth=2.2,
                alpha=0.9,
                zorder=1,
            )
            for coauthor in paper_to_coauthors.get(pid, []):
                if coauthor in coauthor_pos:
                    ox, oy = coauthor_pos[coauthor]
                    ax.plot(
                        [px, ox],
                        [py, oy],
                        color="#5c6670",
                        linewidth=1.6,
                        alpha=0.72,
                        zorder=1,
                    )

        for coauthor in coauthor_sorted:
            x, y = coauthor_pos[coauthor]
            ax.scatter([x], [y], s=320, color="#222222", edgecolors="white", linewidths=1.2, zorder=3)
            ax.text(x, y, shorten(coauthor, 14), fontsize=6.5, ha="center", va="center", color="white", zorder=4)

        for pid, title, label in paper_sorted:
            x, y = paper_pos[pid]
            ax.scatter([x], [y], s=180, color=color_map[label], edgecolors="white", linewidths=1.0, zorder=3)
            ax.text(x, y, shorten(title, 13), fontsize=6.2, ha="center", va="center", color="white", zorder=4)

        ax.scatter([author_pos[0]], [author_pos[1]], s=1100, color="#222222", edgecolors="white", linewidths=1.6, zorder=5)
        ax.text(author_pos[0], author_pos[1], shorten(author_name, 16), fontsize=11, ha="center", va="center", color="white", zorder=6)

        normal_count = sum(1 for _, _, label in paper_nodes if label == "normal")
        outlier_count = sum(1 for _, _, label in paper_nodes if label == "outlier")
        subtitle = f"{graph_data['author_id']}\n30 papers | N={normal_count} O={outlier_count}"
        ax.text(cx, -8.4, subtitle, fontsize=10, ha="center", va="center", color="#333333")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#222222", markersize=12, label="author"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8", markersize=10, label="normal paper"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e45756", markersize=10, label="outlier paper"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    ax.set_title(
        "Three Author Local Graphs in One View\n"
        "Each author keeps 30 papers. Layered author-paper-author view.",
        fontsize=18,
        pad=20,
    )
    fig.text(
        0.02,
        0.02,
        "paper--refpaper is not drawn because IND-WhoIsWho does not provide real reference-edge data.",
        fontsize=11,
        color="#444444",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize author-paper-org local graphs for stage3 analysis.")
    parser.add_argument("--author-data", default="data/IND-WhoIsWho/train_author.json")
    parser.add_argument("--pub-data", default="data/IND-WhoIsWho/pid_to_info_all.json")
    parser.add_argument("--author-ids", nargs="*", default=DEFAULT_AUTHORS)
    parser.add_argument("--output-dir", default="output/stage3_author_graphs")
    parser.add_argument("--max-papers", type=int, default=30)
    args = parser.parse_args()

    with open(args.author_data, "r", encoding="utf-8") as f:
        author_data = json.load(f)
    with open(args.pub_data, "r", encoding="utf-8") as f:
        pub_data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    sampled_graphs = []
    for idx, author_id in enumerate(args.author_ids):
        if author_id not in author_data:
            print(f"skip missing author_id={author_id}")
            continue
        graph_data = build_author_graph(author_id, author_data[author_id], pub_data)
        graph_data = sample_author_graph(graph_data, max_papers=args.max_papers, seed=42 + idx)
        sampled_graphs.append(graph_data)
        safe_name = graph_data["author_name"].replace(" ", "_")
        output_path = os.path.join(args.output_dir, f"{author_id}_{safe_name}.png")
        draw_author_graph(graph_data, output_path)
        print(output_path)

    if sampled_graphs:
        combined_path = os.path.join(args.output_dir, "combined_three_authors.png")
        draw_combined_author_graph(sampled_graphs[:3], combined_path)
        print(combined_path)


if __name__ == "__main__":
    main()
