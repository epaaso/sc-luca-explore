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
import math
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


@dataclass(frozen=True)
class ClusterAnalysisResult:
    pearson_graph: nx.Graph
    pearson_negative: nx.Graph
    pearson_positive: nx.Graph
    corr_types: pd.DataFrame
    counts_plus1: pd.DataFrame
    corr_plus1_norm: pd.DataFrame
    dataset_map: pd.Series
    pearson_df: pd.DataFrame
    pearson_p_values_df: pd.DataFrame
    matrix_counts: pd.DataFrame


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
    "late2": StageConfig(
        name="late",
        membership_csv=REPO_ROOT / "nb_graphAnalysis/output/membership_by_cluster_late2.csv",
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
) -> Optional[ClusterAnalysisResult]:
    groups_path = METADATA_DIR / f"groups_{stage.time_label}_funcnames.csv"
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

    # TODO this would need to filter the corrs
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

    return ClusterAnalysisResult(
        pearson_graph=G,
        pearson_negative=G_negative,
        pearson_positive=G_positive,
        corr_types=corr_types,
        counts_plus1=counts_plus1,
        corr_plus1_norm=corr_plus1_norm,
        dataset_map=dataset_map,
        pearson_df=pearson_df,
        pearson_p_values_df=pearson_p_values_df,
        matrix_counts=matrix_counts,
    )


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
            G.add_edge(u, v, weight=weight, pvalue=pval)

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

    counts_with_dataset = matrix_counts.copy()
    counts_with_dataset.insert(0, SAMPLE_KEY, counts_with_dataset.index)
    counts_with_dataset.insert(1, "dataset", dataset_map.loc[matrix_counts.index].tolist())

    # feature_cols = [col for col in counts_with_dataset.columns if col not in {SAMPLE_KEY, "dataset"}]
    # zeros_per_dataset = counts_with_dataset.groupby("dataset")[feature_cols].apply(lambda df: (df > 0).sum())
    # zeros_per_dataset_T = zeros_per_dataset.T
    # ax = zeros_per_dataset_T.plot(kind="bar", stacked=True, figsize=(14, 9))
    # ax.set_ylabel("# Samples")
    # ax.set_title("# Samples with Cell Type")
    # ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig(cluster_dir / f"0_samples_{time_label}_cluster{cluster_id}.png", dpi=200)
    # plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    draw_graph(G, axes[0], f"Pearson Corr {time_label}")
    draw_graph(G_negative, axes[1], f"Negative Corr {time_label}")
    draw_graph(G_positive, axes[2], f"Positive Corr {time_label}")
    plt.tight_layout()
    plt.savefig(cluster_dir / f"pearson_networks_{time_label}_cluster{cluster_id}.png", dpi=200)
    plt.close()


def load_mi_graph(
    stage: StageConfig,
    cluster_id: int,
    cluster_dir: Path,
) -> nx.Graph:
    net_path = cluster_dir / f"net_{stage.time_label}_cluster{cluster_id}.txt"
    if not net_path.exists():
        raise FileNotFoundError(f"ARACNe network not found: {net_path}")

    G_MI = nx.read_edgelist(net_path, data=(("MI", float), ("p", float)), delimiter="\t")
    logging.info(
        "Stage %s cluster %d: loaded MI network with %d edges",
        stage.name,
        cluster_id,
        G_MI.number_of_edges(),
    )
    return G_MI


def annotate_mi_graph(
    stage: StageConfig,
    cluster_id: int,
    cluster_dir: Path,
    G_MI: nx.Graph,
    pearson_df: pd.DataFrame,
    pearson_p_df: pd.DataFrame,
    corr_threshold: float,
    p_threshold: float,
    export: bool = True,
) -> Tuple[nx.Graph, pd.DataFrame]:
    net_path = cluster_dir / f"net_{stage.time_label}_cluster{cluster_id}.txt"
    if not net_path.exists():
        raise FileNotFoundError(f"MI network missing for cluster {cluster_id}: {net_path}")

    try:
        df_edges = pd.read_csv(net_path, sep="\t", header=None, engine="python")
    except pd.errors.EmptyDataError:
        logging.warning(
            "Stage %s cluster %d: MI network file is empty %s",
            stage.name,
            cluster_id,
            net_path,
        )
        return G_MI, pd.DataFrame()

    if df_edges.shape[1] < 4:
        raise ValueError(
            f"Expected at least 4 columns in MI network {net_path}, got {df_edges.shape[1]}"
        )

    base_cols = ["source", "target", "MI", "p"]
    extra_cols = [f"extra_{i}" for i in range(df_edges.shape[1] - len(base_cols))]
    df_edges.columns = base_cols + extra_cols

    def canonical_pair(u: str, v: str) -> tuple[str, str]:
        return tuple(sorted((u, v)))

    df_edges["_edge_key"] = [canonical_pair(s, t) for s, t in zip(df_edges["source"], df_edges["target"])]
    df_edges = df_edges.loc[~df_edges["_edge_key"].duplicated(keep="first")].copy()

    def lookup_stats(u: str, v: str) -> tuple[float, float]:
        if u in pearson_df.index and v in pearson_df.columns:
            return float(pearson_df.loc[u, v]), float(pearson_p_df.loc[u, v])
        if v in pearson_df.index and u in pearson_df.columns:
            return float(pearson_df.loc[v, u]), float(pearson_p_df.loc[v, u])
        logging.debug(
            "Pearson stats missing for edge (%s, %s); defaulting to corr=0.0, p=1.0",
            u,
            v,
        )
        return 0.0, 1.0

    pearson_values = []
    pvalue_values = []
    sign_values = []

    for row in df_edges.itertuples(index=False):
        corr, pval = lookup_stats(row.source, row.target)

        if np.isnan(corr) or np.isnan(pval):
            sign = "neither"
        elif pval <= p_threshold:
            if corr <= -corr_threshold:
                sign = "negative_significant"
            elif corr >= corr_threshold:
                sign = "positive_significant"
            else:
                sign = "neither"
        else:
            sign = "neither"

        pearson_values.append(corr)
        pvalue_values.append(pval)
        sign_values.append(sign)

        u, v = row.source, row.target
        if G_MI.has_edge(u, v):
            edge_u, edge_v = u, v
        elif G_MI.has_edge(v, u):
            edge_u, edge_v = v, u
        else:
            edge_u = edge_v = None

        if edge_u is not None:
            G_MI[edge_u][edge_v]["pearson"] = corr
            G_MI[edge_u][edge_v]["pvalue"] = pval
            G_MI[edge_u][edge_v]["sign"] = sign

    df_edges["pearson"] = pearson_values
    df_edges["pvalue"] = pvalue_values
    df_edges["sign"] = sign_values
    df_edges.drop(columns="_edge_key", inplace=True)

    if export:
        export_path = cluster_dir / f"net_{stage.time_label}_cluster{cluster_id}_MI_pearson.txt"
        out_cols = base_cols + extra_cols + ["pearson", "pvalue", "sign"]
        df_edges.to_csv(
            export_path,
            sep="\t",
            header=False,
            index=False,
            columns=out_cols,
            float_format="%.15g",
        )

    return G_MI, df_edges


def plot_nonlinear_scatter(
    stage: StageConfig,
    cluster_id: int,
    edges: Sequence[Tuple[str, str]],
    analysis: ClusterAnalysisResult,
    G_MI: nx.Graph,
    cluster_dir: Path,
    batch_size: int = 12,
) -> None:
    if not edges:
        logging.info("Stage %s cluster %d: no non-linear edges to plot", stage.name, cluster_id)
        return

    corr_types = analysis.corr_types.copy()
    counts_plus1 = analysis.counts_plus1.reindex(corr_types.index)
    corr_plus1_norm = analysis.corr_plus1_norm.reindex(corr_types.index)
    dataset_series = analysis.dataset_map.reindex(corr_types.index).fillna("unknown")

    unique_datasets = pd.unique(dataset_series)
    cmap = plt.get_cmap("tab20", max(len(unique_datasets), 1))
    color_mapping = {dataset: cmap(i) for i, dataset in enumerate(unique_datasets)}
    dataset_colors = dataset_series.map(color_mapping)

    pearson_df = analysis.pearson_df
    pearson_p_df = analysis.pearson_p_values_df

    edges_sorted = [(min(u, v), max(u, v)) for u, v in edges]
    edges_sorted = sorted(set(edges_sorted))

    variants = [
        {"key": "hist2d", "title": "2D Histogram of log10(counts + 1) 1s removed", "needs_legend": False},
        {"key": "logscatter", "title": "Scatter of normalized log10(counts + 1)", "needs_legend": True},
        {"key": "scatter", "title": "Scatter of normalized counts", "needs_legend": True},
    ]

    for variant in variants:
        for batch_index in range(0, len(edges_sorted), batch_size):
            batch = edges_sorted[batch_index : batch_index + batch_size]
            if not batch:
                continue

            n_edges = len(batch)
            ncols = 4
            nrows = max(1, math.ceil(n_edges / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
            if isinstance(axes, np.ndarray):
                axes_iter = axes.ravel()
            else:
                axes_iter = np.array([axes])

            for ax, (var1, var2) in zip(axes_iter, batch):
                corr = pearson_df.loc[var1, var2] if var1 in pearson_df.index and var2 in pearson_df.columns else np.nan
                pval = pearson_p_df.loc[var1, var2] if var1 in pearson_p_df.index and var2 in pearson_p_df.columns else np.nan
                mi_data = G_MI.get_edge_data(var1, var2) or G_MI.get_edge_data(var2, var1)
                mi = mi_data.get("MI") if mi_data and "MI" in mi_data else np.nan

                if variant["key"] == "hist2d":
                    if var1 not in counts_plus1.columns or var2 not in counts_plus1.columns:
                        logging.warning(
                            "Stage %s cluster %d: variables %s/%s missing from counts_plus1",
                            stage.name,
                            cluster_id,
                            var1,
                            var2,
                        )
                        ax.axis("off")
                        continue

                    x_vals = np.log10(counts_plus1[var1].to_numpy())
                    y_vals = np.log10(counts_plus1[var2].to_numpy())
                    mask = (x_vals == 0) & (y_vals == 0)
                    x_vals = x_vals[~mask]
                    y_vals = y_vals[~mask]

                    if x_vals.size == 0 or y_vals.size == 0:
                        ax.axis("off")
                        ax.set_title("No data", fontsize=9)
                        continue

                    ax.hist2d(x_vals, y_vals, bins=30, cmap="viridis")
                    ax.set_xlabel(f"log10({var1} + 1)")
                    ax.set_ylabel(f"log10({var2} + 1)")

                elif variant["key"] == "logscatter":
                    if var1 not in corr_plus1_norm.columns or var2 not in corr_plus1_norm.columns:
                        logging.warning(
                            "Stage %s cluster %d: variables %s/%s missing from corr_plus1_norm",
                            stage.name,
                            cluster_id,
                            var1,
                            var2,
                        )
                        ax.axis("off")
                        continue

                    x_vals = corr_plus1_norm[var1].to_numpy()
                    y_vals = corr_plus1_norm[var2].to_numpy()
                    colors = dataset_colors.to_numpy()

                    ax.scatter(x_vals, y_vals, s=6, c=colors)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel(var1)
                    ax.set_ylabel(var2)

                else:  # scatter of normalized counts
                    if var1 not in corr_types.columns or var2 not in corr_types.columns:
                        logging.warning(
                            "Stage %s cluster %d: variables %s/%s missing from corr_types",
                            stage.name,
                            cluster_id,
                            var1,
                            var2,
                        )
                        ax.axis("off")
                        continue

                    colors = dataset_colors.to_numpy()
                    ax.scatter(corr_types[var1].to_numpy(), corr_types[var2].to_numpy(), s=6, c=colors)
                    ax.set_xlabel(var1)
                    ax.set_ylabel(var2)

                ax.set_title(f"P corr: {corr:.2f}, pval: {pval:.4f}, MI: {mi:.2f}", fontsize=9)

            for ax in axes_iter[n_edges:]:
                ax.axis("off")

            fig.suptitle(variant["title"], fontsize=14, y=1.02)

            if variant["needs_legend"]:
                handles = [
                    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=6, label=dataset)
                    for dataset, color in color_mapping.items()
                ]
                fig.legend(handles=handles, loc="center right", ncol=1, fontsize="small")
                fig.tight_layout(rect=[0.05, 0.05, 0.85, 0.95])
            else:
                fig.tight_layout()

            part = batch_index // batch_size + 1
            suffix = variant["key"]
            outfile = cluster_dir / f"nonlinear_{suffix}_{stage.time_label}_cluster{cluster_id}_part{part}.png"
            fig.savefig(outfile, dpi=180)
            logging.info(
                "Stage %s cluster %d: saved %s plot (%d pairs) to %s",
                stage.name,
                cluster_id,
                variant["key"],
                len(batch),
                outfile.name,
            )
            plt.close(fig)

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

    analysis = plot_corr_networks_cluster(
        stage,
        cluster_id,
        cluster_samples,
        corr_threshold=args.corr_threshold,
        p_threshold=args.p_threshold,
        cluster_dir=cluster_dir,
    )

    if analysis is None:
        logging.warning("Stage %s cluster %d: skipping remainder due to insufficient data", stage.name, cluster_id)
        return

    G_MI = load_mi_graph(stage, cluster_id, cluster_dir)

    G_MI_annotated, annotated_df = annotate_mi_graph(
        stage,
        cluster_id,
        cluster_dir,
        G_MI,
        analysis.pearson_df,
        analysis.pearson_p_values_df,
        corr_threshold=args.corr_threshold,
        p_threshold=args.p_threshold,
        export=True,
    )

    non_linear_edges = {
        tuple(sorted((row.source, row.target)))
        for row in annotated_df.itertuples(index=False)
        if row.sign == "neither"
    }

    logging.info(
        "Stage %s cluster %d: %d edges labelled as non-linear (sign == neither)",
        stage.name,
        cluster_id,
        len(non_linear_edges),
    )

    plot_nonlinear_scatter(
        stage,
        cluster_id,
        sorted(non_linear_edges),
        analysis,
        G_MI_annotated,
        cluster_dir,
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
