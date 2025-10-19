"""Post-process ARACNe cluster outputs with Pearson correlations.

This script reproduces the "pearson_compare" notebook workflow for every
stage/cluster combination produced by ``run_aracne_by_cluster.py``. For each
cluster it:

1. Regenerates the Pearson correlation networks (positive/negative) for the
   subset of samples that belong to the cluster.
2. Identifies MI-only (non-linear) and Pearson-only edges by comparing the
   ARACNe network with the Pearson graph.
3. Annotates the ARACNe network with Pearson correlation, p-value, and a sign
   label, exporting the combined edge list.

Outputs are written next to the ARACNe files inside
``outputARACNE/cluster_runs/<stage>/cluster_XX``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


CELL_TYPE_KEY = "cell_type_adjusted"
SAMPLE_KEY = "sample"
COUNT_COLUMN = "0"

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = REPO_ROOT / "metadata"
NB_OUTPUT_DIR = REPO_ROOT / "nb_graphAnalysis" / "output"
DEFAULT_CLUSTER_ROOT = REPO_ROOT / "outputARACNE" / "cluster_runs"


@dataclass(frozen=True)
class StageConfig:
    name: str
    membership_csv: Path
    time_label: str


STAGE_CONFIGS = {
    "early": StageConfig(
        name="early",
        membership_csv=REPO_ROOT / "nb_graphAnalysis/output/membership_by_cluster_early.csv",
        time_label="I-II_leidenwu",
    ),
    "late": StageConfig(
        name="late",
        membership_csv=REPO_ROOT / "nb_graphAnalysis/output/membership_by_cluster_late.csv",
        time_label="III-IV_leidenwu",
    ),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate ARACNe cluster networks with Pearson correlations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=sorted(STAGE_CONFIGS.keys()),
        default=sorted(STAGE_CONFIGS.keys()),
        help="Stages to process (early, late).",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        type=int,
        help="Optional list of cluster ids to process (applies to all selected stages).",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.2,
        help="Absolute Pearson correlation threshold for retaining edges.",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="Maximum Pearson p-value to consider significant.",
    )
    parser.add_argument(
        "--cluster-root",
        type=Path,
        default=DEFAULT_CLUSTER_ROOT,
        help="Directory containing per-stage cluster subfolders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if the annotated network file already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity to warnings only.",
    )

    return parser.parse_args(argv)


def setup_logging(quiet: bool = False) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def read_membership_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Membership file not found: {path}")
    df = pd.read_csv(path)
    required = {SAMPLE_KEY, "membership"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Membership CSV {path} missing columns: {missing}")
    return df


def derive_cluster_ids(df: pd.DataFrame) -> List[int]:
    return sorted(df["membership"].astype(int).unique())


def filter_cluster_list(all_clusters: Iterable[int], selection: Optional[Sequence[int]]) -> List[int]:
    if selection is None:
        return list(all_clusters)
    sel = sorted(set(selection))
    return [cluster for cluster in all_clusters if cluster in sel]


def draw_graph(G: nx.Graph, ax: plt.Axes, title: str) -> None:
    if G.number_of_nodes() == 0:
        ax.set_title(f"{title}\n(no edges)")
        ax.axis("off")
        return
    pos = nx.spring_layout(G, seed=0)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=170,
        font_size=7,
        font_family="sans-serif",
        edge_color="black",
        ax=ax,
    )
    edge_labels = {edge: f"{weight:.2f}" for edge, weight in nx.get_edge_attributes(G, "weight").items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, ax=ax)
    ax.set_title(title)
    ax.axis("off")


def plot_corr_networks_cluster(
    stage: StageConfig,
    cluster_id: int,
    cluster_samples: Sequence[str],
    *,
    corr_threshold: float,
    p_threshold: float,
    cluster_dir: Path,
) -> Optional[Tuple[nx.Graph, nx.Graph, nx.Graph]]:
    groups_path = METADATA_DIR / f"groups_{stage.time_label}.csv"
    if not groups_path.exists():
        raise FileNotFoundError(f"Groups file not found: {groups_path}")

    groups = pd.read_csv(groups_path)
    groups = groups[groups[SAMPLE_KEY].isin(cluster_samples)]
    if groups.empty:
        logging.warning("Stage %s cluster %d: no matching samples in groups file", stage.name, cluster_id)
        return None

    if COUNT_COLUMN not in groups.columns:
        raise ValueError(f"Expected count column '{COUNT_COLUMN}' in {groups_path}")

    groups = groups[[CELL_TYPE_KEY, SAMPLE_KEY, COUNT_COLUMN, "dataset"]]
    groups = groups.rename(columns={COUNT_COLUMN: "count"})
    groups = groups[groups["count"] > 1]
    if groups.empty:
        logging.warning("Stage %s cluster %d: all counts ≤ 1 after cutoff", stage.name, cluster_id)
        return None

    matrix_counts = (
        groups.pivot_table(index=SAMPLE_KEY, columns=CELL_TYPE_KEY, values="count", aggfunc="sum")
        .fillna(0)
    )
    matrix_counts = matrix_counts.loc[:, (matrix_counts.sum(axis=0) > 0)]
    if matrix_counts.shape[0] < 3 or matrix_counts.shape[1] < 2:
        logging.warning(
            "Stage %s cluster %d: need ≥3 samples and ≥2 cell types (got %s)",
            stage.name,
            cluster_id,
            matrix_counts.shape,
        )
        return None

    dataset_map = (
        groups[[SAMPLE_KEY, "dataset"]]
        .drop_duplicates()
        .set_index(SAMPLE_KEY)["dataset"]
        .reindex(matrix_counts.index)
        .fillna("unknown")
    )

    corr_types = matrix_counts.div(matrix_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    counts_plus1 = matrix_counts + 1
    corr_plus1_norm = counts_plus1.div(counts_plus1.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    pearson_df, pearson_p_values_df = compute_pearson_matrices(corr_types)
    G, G_negative, G_positive = build_pearson_graphs(
        pearson_df, pearson_p_values_df, corr_threshold, p_threshold
    )

    # save_pearson_outputs(
    #     cluster_dir,
    #     stage,
    #     cluster_id,
    #     corr_types,
    #     counts_plus1,
    #     corr_plus1_norm,
    #     dataset_map,
    #     pearson_df,
    #     pearson_p_values_df,
    #     matrix_counts,
    #     G,
    #     G_negative,
    #     G_positive,
    # )

    return G, G_negative, G_positive


def compute_pearson_matrices(corr_types: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = corr_types.to_numpy().T
    n_variables = data.shape[0]

    pearson_matrix = np.identity(n_variables)
    pearson_p_values = np.zeros((n_variables, n_variables))

    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            correlation, p_value = pearsonr(data[i], data[j])
            if np.isnan(correlation):
                correlation = 0.0
                p_value = 1.0
            pearson_matrix[i, j] = pearson_matrix[j, i] = correlation
            pearson_p_values[i, j] = pearson_p_values[j, i] = p_value

    cols = corr_types.columns
    pearson_df = pd.DataFrame(pearson_matrix, index=cols, columns=cols)
    pearson_p_df = pd.DataFrame(pearson_p_values, index=cols, columns=cols)
    return pearson_df, pearson_p_df


def build_pearson_graphs(
    pearson_df: pd.DataFrame,
    pearson_p_df: pd.DataFrame,
    corr_threshold: float,
    p_threshold: float,
) -> Tuple[nx.Graph, nx.Graph, nx.Graph]:
    G = nx.Graph()
    cols = list(pearson_df.columns)
    for i, u in enumerate(cols):
        for j in range(i + 1, len(cols)):
            v = cols[j]
            weight = float(pearson_df.iloc[i, j])
            pval = float(pearson_p_df.iloc[i, j])
            if abs(weight) >= corr_threshold and pval <= p_threshold:
                G.add_edge(u, v, weight=round(weight, 2), pvalue=pval)

    nodes_without_edges = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(nodes_without_edges)

    G.graph["corr_threshold"] = corr_threshold
    G.graph["p_threshold"] = p_threshold

    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > 0]
    G_negative = G.edge_subgraph(negative_edges).copy()
    G_positive = G.edge_subgraph(positive_edges).copy()

    return G, G_negative, G_positive


def save_pearson_outputs(
    cluster_dir: Path,
    stage: StageConfig,
    cluster_id: int,
    corr_types: pd.DataFrame,
    counts_plus1: pd.DataFrame,
    corr_plus1_norm: pd.DataFrame,
    dataset_map: pd.Series,
    pearson_df: pd.DataFrame,
    pearson_p_values_df: pd.DataFrame,
    matrix_counts: pd.DataFrame,
    G: nx.Graph,
    G_negative: nx.Graph,
    G_positive: nx.Graph,
) -> None:
    cluster_dir.mkdir(parents=True, exist_ok=True)

    time_label = stage.time_label
    pearson_df.to_csv(cluster_dir / f"pearson_{time_label}_cluster{cluster_id}.csv")
    pearson_p_values_df.to_csv(cluster_dir / f"pearson_p_values_{time_label}_cluster{cluster_id}.csv")

    corr_types_out = corr_types.copy()
    corr_types_out.insert(0, SAMPLE_KEY, corr_types_out.index)
    corr_types_out.insert(1, "dataset", dataset_map.loc[corr_types.index].tolist())
    corr_types_out.to_csv(cluster_dir / f"corr_types_{time_label}_cluster{cluster_id}.csv", index=False)

    counts_plus1_out = counts_plus1.copy()
    counts_plus1_out.insert(0, SAMPLE_KEY, counts_plus1_out.index)
    counts_plus1_out.insert(1, "dataset", dataset_map.loc[counts_plus1.index].tolist())
    counts_plus1_out.to_csv(cluster_dir / f"counts_plus1_{time_label}_cluster{cluster_id}.csv", index=False)

    corr_plus1_norm_out = corr_plus1_norm.copy()
    corr_plus1_norm_out.insert(0, SAMPLE_KEY, corr_plus1_norm_out.index)
    corr_plus1_norm_out.insert(1, "dataset", dataset_map.loc[corr_plus1_norm.index].tolist())
    corr_plus1_norm_out.to_csv(
        cluster_dir / f"corr_plus1_norm_{time_label}_cluster{cluster_id}.csv", index=False
    )

    counts_with_dataset = matrix_counts.copy()
    counts_with_dataset.insert(0, SAMPLE_KEY, counts_with_dataset.index)
    counts_with_dataset.insert(1, "dataset", dataset_map.loc[matrix_counts.index].tolist())

    feature_cols = [col for col in counts_with_dataset.columns if col not in {SAMPLE_KEY, "dataset"}]
    zeros_per_dataset = counts_with_dataset.groupby("dataset")[feature_cols].apply(lambda df: (df > 0).sum())
    zeros_per_dataset_T = zeros_per_dataset.T
    ax = zeros_per_dataset_T.plot(kind="bar", stacked=True, figsize=(14, 9))
    ax.set_ylabel("# Samples")
    ax.set_title("# Samples with Cell Type")
    ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(cluster_dir / f"0_samples_{time_label}_cluster{cluster_id}.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    draw_graph(G, axes[0], f"Pearson Corr {time_label}")
    draw_graph(G_negative, axes[1], f"Negative Corr {time_label}")
    draw_graph(G_positive, axes[2], f"Positive Corr {time_label}")
    plt.tight_layout()
    plt.savefig(cluster_dir / f"pearson_networks_{time_label}_cluster{cluster_id}.png", dpi=200)
    plt.close()


def get_nonlinear_edges(
    stage: StageConfig,
    cluster_id: int,
    cluster_dir: Path,
    G_pear: nx.Graph,
) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]], nx.Graph]:
    net_path = cluster_dir / f"net_{stage.time_label}_cluster{cluster_id}.txt"
    if not net_path.exists():
        raise FileNotFoundError(f"ARACNe network not found: {net_path}")

    G_MI = nx.read_edgelist(net_path, data=(("MI", float), ("p", float)), delimiter="\t")
    edges_MI = {(min(u, v), max(u, v)) for u, v in G_MI.edges()}
    edges_pear = {(min(u, v), max(u, v)) for u, v in G_pear.edges()}

    non_linear_edges = edges_MI - edges_pear
    non_signif_edges = edges_pear - edges_MI
    threshold = G_pear.graph.get("corr_threshold", float("nan"))
    logging.info(
        "Stage %s cluster %d: MI edges %d, Pearson edges %d, non-linear %d",
        stage.name,
        cluster_id,
        len(edges_MI),
        len(edges_pear),
        len(non_linear_edges),
    )
    if not np.isnan(threshold):
        logging.info(
            "Stage %s cluster %d: corr threshold %.3f", stage.name, cluster_id, threshold
        )

    return non_linear_edges, non_signif_edges, G_MI


def add_pearson_edges(
    stage: StageConfig,
    cluster_id: int,
    cluster_dir: Path,
    G_MI: nx.Graph,
    corr_threshold: float,
    p_threshold: float,
    export: bool = True,
) -> nx.Graph:
    pearson_path = cluster_dir / f"pearson_{stage.time_label}_cluster{cluster_id}.csv"
    pearson_p_path = cluster_dir / f"pearson_p_values_{stage.time_label}_cluster{cluster_id}.csv"
    if not pearson_path.exists() or not pearson_p_path.exists():
        raise FileNotFoundError(
            f"Pearson outputs missing for cluster {cluster_id}: {pearson_path} / {pearson_p_path}"
        )

    pearson_df = pd.read_csv(pearson_path, index_col=0)
    pearson_p_df = pd.read_csv(pearson_p_path, index_col=0)

    for u, v in G_MI.edges():
        if u not in pearson_df.index or v not in pearson_df.columns:
            continue
        corr = float(pearson_df.loc[u, v])
        pval = float(pearson_p_df.loc[u, v])

        if pval <= p_threshold:
            if corr <= -corr_threshold:
                sign = "negative_significant"
            elif corr >= corr_threshold:
                sign = "positive_significant"
            else:
                sign = "neither"
        else:
            sign = "neither"

        G_MI[u][v]["pearson"] = corr
        G_MI[u][v]["pvalue"] = pval
        G_MI[u][v]["sign"] = sign

    if export:
        export_path = cluster_dir / f"net_{stage.time_label}_cluster{cluster_id}_MI_pearson.txt"
        nx.write_edgelist(
            G_MI,
            export_path,
            data=["MI", "p", "pearson", "pvalue", "sign"],
            delimiter="\t",
        )

    return G_MI


def export_edge_set(
    edges: Iterable[Tuple[str, str]],
    graph: nx.Graph,
    path: Path,
    *,
    attrs: Optional[Iterable[str]] = None,
) -> None:
    records = []
    for u, v in edges:
        data = graph.get_edge_data(u, v, default={})
        record = {"source": u, "target": v}
        if attrs is None:
            for key, value in data.items():
                record[key] = value
        else:
            for key in attrs:
                record[key] = data.get(key)
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)


def process_cluster(
    stage: StageConfig,
    cluster_id: int,
    cluster_samples: Sequence[str],
    args: argparse.Namespace,
) -> None:
    cluster_dir = args.cluster_root / stage.name / f"cluster_{cluster_id:02d}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    annotated_network = cluster_dir / f"net_{stage.time_label}_cluster{cluster_id}_MI_pearson.txt"
    if annotated_network.exists() and not args.force:
        logging.info(
            "Stage %s cluster %d: annotated network already exists (use --force to re-run)",
            stage.name,
            cluster_id,
        )
        return

    result = plot_corr_networks_cluster(
        stage,
        cluster_id,
        cluster_samples,
        corr_threshold=args.corr_threshold,
        p_threshold=args.p_threshold,
        cluster_dir=cluster_dir,
    )

    if result is None:
        logging.warning("Stage %s cluster %d: skipping remainder due to insufficient data", stage.name, cluster_id)
        return

    G_pear, _, _ = result

    non_linear_edges, non_signif_edges, G_MI = get_nonlinear_edges(stage, cluster_id, cluster_dir, G_pear)

    export_edge_set(
        non_linear_edges,
        G_MI,
        cluster_dir / f"nonlinear_edges_{stage.time_label}_cluster{cluster_id}.csv",
        attrs=["MI", "p"],
    )
    export_edge_set(
        non_signif_edges,
        G_pear,
        cluster_dir / f"pearson_only_edges_{stage.time_label}_cluster{cluster_id}.csv",
        attrs=["weight", "pvalue"],
    )

    G_MI_annotated = add_pearson_edges(
        stage,
        cluster_id,
        cluster_dir,
        G_MI,
        corr_threshold=args.corr_threshold,
        p_threshold=args.p_threshold,
        export=True,
    )

    logging.info(
        "Stage %s cluster %d: annotated %d MI edges",
        stage.name,
        cluster_id,
        G_MI_annotated.number_of_edges(),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.quiet)

    for stage_name in args.stages:
        stage = STAGE_CONFIGS[stage_name]
        membership_df = read_membership_csv(stage.membership_csv)
        cluster_ids = filter_cluster_list(derive_cluster_ids(membership_df), args.clusters)
        logging.info(
            "Processing stage '%s' clusters: %s",
            stage.name,
            ", ".join(map(str, cluster_ids)) if cluster_ids else "(none)",
        )

        for cluster_id in cluster_ids:
            cluster_samples = (
                membership_df.loc[membership_df["membership"] == cluster_id, SAMPLE_KEY]
                .dropna()
                .astype(str)
                .tolist()
            )
            if not cluster_samples:
                logging.warning(
                    "Stage %s cluster %d: no samples listed in membership file",
                    stage.name,
                    cluster_id,
                )
                continue
            process_cluster(stage, cluster_id, cluster_samples, args)

    logging.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
