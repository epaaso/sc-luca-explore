#!/usr/bin/env python
"""Run dataset-stratified CellPhoneDB DEG analyses and aggregate consensus."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellphonedb.src.core.methods import cpdb_degs_analysis_method
from nb_DE_wilcox.modal_DE import CommonConfig, DataLoader, DataLoaderConfig
from shared.paths import (
    CLUSTER_GRAPH_RESULTS,
    EXTERNAL_CELLPHONEDB_DATABASE,
    EXTERNAL_CELLPHONEDB_ROOT,
    EXTERNAL_DATA_DIR,
    EXTERNAL_MARKER_ROOT,
    MEMBERSHIP_FILES,
)


DATA_DIR = EXTERNAL_DATA_DIR
MARKER_DIR = EXTERNAL_MARKER_ROOT
GRAPH_DIR = CLUSTER_GRAPH_RESULTS
OUTPUT_DIR = EXTERNAL_CELLPHONEDB_ROOT
CPDB_FILE = EXTERNAL_CELLPHONEDB_DATABASE
TIME_FOLDERS = {"I-II": "early", "III-IV": "late"}
TUMOR_TERMS = ("Tumor", "Ciliated", "AT2", "AT1", "Club")


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    pred_name: str
    stage_key: str
    skip_stages: bool = False
    gene_feature: str | None = None
    obs_unique: bool = False


SURGERY_SPECS = {
    "Zuani_2024_NSCLC": DatasetSpec("Zuani_2024_NSCLC", "Zuani", "stage"),
    "Deng_Liu_LUAD_2024": DatasetSpec(
        "Deng_Liu_LUAD_2024", "Deng", "Pathological stage"
    ),
    "Hu_Zhang_2023_NSCLC": DatasetSpec(
        "Hu_Zhang_2023_NSCLC", "Hu", "Clinical Stage"
    ),
    "Trinks_Bishoff_2021_NSCLC": DatasetSpec(
        "Trinks_Bishoff_2021_NSCLC", "Bishoff", "stage", skip_stages=True
    ),
}


def dataset_spec(dataset: str) -> DatasetSpec:
    if dataset in SURGERY_SPECS:
        return SURGERY_SPECS[dataset]
    return DatasetSpec(dataset, "Atlas", "uicc_stage", gene_feature="feature_name")


def marker_path(time: str, dataset: str, cluster: int, contrast: str) -> Path:
    return MARKER_DIR / contrast / f"{time}_{dataset}_cluster_{cluster}_auc.npy"


def discover_runs(
    times: list[str] | None = None,
    clusters: list[int] | None = None,
    datasets: list[str] | None = None,
) -> list[tuple[str, int, str]]:
    pattern = re.compile(r"^(I-II|III-IV)_(.+)_cluster_([0-2])_auc\.npy$")
    found = []
    for path in (MARKER_DIR / "tumor-vs-all").glob("*_cluster_*_auc.npy"):
        match = pattern.match(path.name)
        if not match:
            continue
        time, dataset, cluster_text = match.groups()
        cluster = int(cluster_text)
        if times and time not in times:
            continue
        if clusters and cluster not in clusters:
            continue
        if datasets and dataset not in datasets:
            continue
        found.append((time, cluster, dataset))
    return sorted(set(found))


def load_degs(time: str, cluster: int, dataset: str, auc_threshold: float) -> pd.DataFrame:
    rows = []
    for contrast in ("normal-vs-normal", "tumor-vs-all"):
        path = marker_path(time, dataset, cluster, contrast)
        if not path.exists():
            if contrast == "normal-vs-normal":
                logging.warning("Normal marker file missing: %s", path)
            continue
        region = np.load(path, allow_pickle=True).item()
        names = region["names"]
        scores = region["scores"]
        for cell_type in names.dtype.names:
            is_tumor = any(term in cell_type for term in TUMOR_TERMS)
            if (contrast == "normal-vs-normal") == is_tumor:
                continue
            genes = np.asarray(names[cell_type], dtype=str)
            aucs = np.asarray(scores[cell_type], dtype=float)
            keep = np.isfinite(aucs) & (aucs >= auc_threshold)
            rows.extend(zip(np.repeat(cell_type, keep.sum()), genes[keep]))
    if not rows:
        return pd.DataFrame(columns=["cluster", "gene"])
    return pd.DataFrame(rows, columns=["cluster", "gene"]).drop_duplicates()


def make_loader(time: str, cluster: int, spec: DatasetSpec) -> DataLoader:
    common = CommonConfig()
    common.ext_name = spec.dataset
    common.file_ext_name = spec.dataset
    common.name = spec.dataset
    common.time = time
    common.backup_dir = str(DATA_DIR)
    config = DataLoaderConfig(
        pred_name=spec.pred_name,
        obs_unique=spec.obs_unique,
        cell_key="cell_type_adjusted",
        stage_key=spec.stage_key,
        skip_stages=spec.skip_stages,
        log_layer=False,
        gene_feature=spec.gene_feature,
        cluster_id=cluster,
        membership_csv=str(MEMBERSHIP_FILES[time]),
    )
    return DataLoader(common, config)


def stages_for_time(time: str, skip_stages: bool) -> list[str] | None:
    if skip_stages:
        return None
    if time == "I-II":
        return ["IA1", "IB", "IA2", "IA3", "IIB", "II", "I"]
    return ["IIIA", "IIIB", "III", "III or IV", "IV"]


def find_patient_column(adata: ad.AnnData, membership_file: Path) -> str:
    membership = pd.read_csv(membership_file)
    samples = set(membership["sample"].astype(str))
    preferred = ["sample", "Sample", "Patient", "Patient Number", "orig.ident", "patient_id"]
    for column in preferred + [c for c in adata.obs.columns if c not in preferred]:
        if column in adata.obs and set(adata.obs[column].dropna().astype(str)) & samples:
            return column
    raise ValueError("No AnnData observation column matches ARACNE membership samples")


def balance_cells(
    adata: ad.AnnData,
    patient_column: str,
    cap: int,
    seed: int,
    min_cells: int,
    min_patients: int,
) -> tuple[ad.AnnData, pd.DataFrame]:
    obs = adata.obs.copy()
    obs["_position"] = np.arange(adata.n_obs)
    obs["_patient"] = obs[patient_column].astype(str)
    obs["_cell_type"] = obs["type_tissue"].astype(str)
    rng = np.random.default_rng(seed)
    positions = []
    for _, group in obs.groupby(["_patient", "_cell_type"], observed=True, sort=True):
        values = group["_position"].to_numpy()
        if len(values) > cap:
            values = rng.choice(values, cap, replace=False)
        positions.extend(values.tolist())
    balanced = adata[np.sort(positions)].copy()
    support = (
        balanced.obs.assign(
            patient=balanced.obs[patient_column].astype(str),
            cell_type=balanced.obs["type_tissue"].astype(str),
        )
        .groupby("cell_type", observed=True)
        .agg(cells=("cell_type", "size"), patients=("patient", "nunique"))
        .reset_index()
    )
    valid = support.loc[
        (support["cells"] >= min_cells) & (support["patients"] >= min_patients),
        "cell_type",
    ]
    balanced = balanced[balanced.obs["type_tissue"].astype(str).isin(valid)].copy()
    support["included"] = support["cell_type"].isin(valid)
    return balanced, support


def collapse_genes(adata: ad.AnnData, gene_feature: str | None) -> ad.AnnData:
    genes = (
        adata.var[gene_feature].astype(str).to_numpy()
        if gene_feature and gene_feature in adata.var
        else adata.var_names.astype(str).to_numpy()
    )
    genes = np.char.strip(genes.astype(str))
    valid = pd.notna(genes) & (genes != "") & (genes != "nan")
    adata = adata[:, valid].copy()
    genes = genes[valid]

    unique_genes, inverse = np.unique(genes, return_inverse=True)
    if len(unique_genes) == len(genes):
        adata.var_names = genes
        return adata

    matrix = adata.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    mapper = sparse.csr_matrix(
        (np.ones(len(inverse)), (np.arange(len(inverse)), inverse)),
        shape=(len(inverse), len(unique_genes)),
    )
    collapsed = matrix @ mapper
    return ad.AnnData(
        X=collapsed.tocsr(),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=unique_genes),
    )


def prepare_run(
    time: str,
    cluster: int,
    dataset: str,
    auc_threshold: float,
    patient_cap: int,
    seed: int,
    min_cells: int,
    min_patients: int,
) -> Path:
    spec = dataset_spec(dataset)
    output = (
        OUTPUT_DIR / TIME_FOLDERS[time] / f"cluster_{cluster:02d}" / "datasets" / dataset
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    counts_path = output / "counts.h5ad"
    meta_path = output / "metadata.tsv"
    degs_path = output / "degs.tsv"
    support_path = output / "cell_type_support.csv"
    if manifest_path.exists() and counts_path.exists() and meta_path.exists() and degs_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") in {"prepared", "complete"}:
            logging.info("Skipping prepared dataset input: %s", output)
            return output

    loader = make_loader(time, cluster, spec)
    preds = loader.load_predictions()
    adata = loader.load_anndata(preds, stages_for_time(time, spec.skip_stages))
    if adata is None or adata.n_obs == 0:
        raise ValueError("No cells remain after dataset, stage, and cluster filtering")

    patient_column = find_patient_column(adata, MEMBERSHIP_FILES[time])
    cluster_membership = pd.read_csv(MEMBERSHIP_FILES[time])
    expected_patients = set(
        cluster_membership.loc[
            (cluster_membership["membership"] == cluster)
            & (cluster_membership["dataset"].astype(str).str.contains(dataset.split("_NSCLC")[0])),
            "sample",
        ].astype(str)
    )
    observed_patients = set(adata.obs[patient_column].astype(str))
    patient_count = len(observed_patients)
    if patient_count < 3:
        raise ValueError(f"Only {patient_count} patients available; at least 3 required")
    if expected_patients and not observed_patients.issubset(expected_patients):
        raise ValueError("Filtered cells include patients outside the requested ARACNE cluster")

    balanced, support = balance_cells(
        adata, patient_column, patient_cap, seed, min_cells, min_patients
    )
    if balanced.n_obs == 0 or balanced.obs["type_tissue"].nunique() < 2:
        raise ValueError("Fewer than two QC-passing cell types remain after balancing")

    raw = balanced.layers["count"] if "count" in balanced.layers else balanced.X
    balanced.X = raw.copy()
    balanced = collapse_genes(balanced, spec.gene_feature)
    sc.pp.normalize_total(balanced, target_sum=10_000)
    sc.pp.log1p(balanced)
    balanced.obs_names_make_unique()
    balanced.obs["cell_type"] = balanced.obs["type_tissue"].astype(str)

    degs = load_degs(time, cluster, dataset, auc_threshold)
    valid_types = set(balanced.obs["cell_type"])
    valid_genes = set(balanced.var_names)
    degs = degs[
        degs["cluster"].isin(valid_types) & degs["gene"].astype(str).isin(valid_genes)
    ].drop_duplicates()
    if degs.empty:
        raise ValueError("No marker genes pass AUC and input-expression filters")

    balanced.write_h5ad(counts_path)
    balanced.obs[["cell_type"]].to_csv(meta_path, sep="\t", index=True, index_label="Cell")
    degs.to_csv(degs_path, sep="\t", index=False)
    support.to_csv(support_path, index=False)
    manifest = {
        "status": "prepared",
        "time": time,
        "cluster": cluster,
        "dataset": dataset,
        "patient_column": patient_column,
        "patients": patient_count,
        "cells": balanced.n_obs,
        "cell_types": sorted(valid_types),
        "degs": len(degs),
        "parameters": {
            "auc_threshold": auc_threshold,
            "marker_contrasts": ["normal-vs-normal", "tumor-vs-all"],
            "patient_cap": patient_cap,
            "seed": seed,
            "min_cells": min_cells,
            "min_patients": min_patients,
            "normalization": "normalize_total_10000_log1p",
        },
        "dataset_spec": asdict(spec),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return output


def make_long_results(results: dict, degs: pd.DataFrame) -> pd.DataFrame:
    relevant = results["relevant_interactions"]
    means = results["significant_means"]
    pair_columns = [column for column in relevant.columns if "|" in str(column)]
    id_columns = [column for column in relevant.columns if column not in pair_columns]
    rel_long = relevant.melt(
        id_vars=id_columns,
        value_vars=pair_columns,
        var_name="interacting_cells",
        value_name="relevant",
    )
    rel_long = rel_long[pd.to_numeric(rel_long["relevant"], errors="coerce").fillna(0) > 0]
    mean_ids = [column for column in means.columns if column not in pair_columns]
    mean_long = means.melt(
        id_vars=mean_ids,
        value_vars=[c for c in pair_columns if c in means],
        var_name="interacting_cells",
        value_name="significant_mean",
    )
    merge_keys = [c for c in ["id_cp_interaction", "interacting_pair", "interacting_cells"] if c in rel_long and c in mean_long]
    long = rel_long.merge(
        mean_long[merge_keys + ["significant_mean"]],
        on=merge_keys,
        how="left",
    )
    long[["sender_cell_type", "receiver_cell_type"]] = long["interacting_cells"].str.split(
        "|", n=1, expand=True
    )
    deg_map = degs.groupby("cluster")["gene"].agg(set).to_dict()
    gene_a = long["gene_a"] if "gene_a" in long else pd.Series("", index=long.index)
    gene_b = long["gene_b"] if "gene_b" in long else pd.Series("", index=long.index)
    long["ligand_deg"] = [
        gene in deg_map.get(cell, set())
        for gene, cell in zip(gene_a, long["sender_cell_type"])
    ]
    long["receptor_deg"] = [
        gene in deg_map.get(cell, set())
        for gene, cell in zip(gene_b, long["receiver_cell_type"])
    ]
    keep = [
        "id_cp_interaction", "interacting_pair", "partner_a", "partner_b",
        "gene_a", "gene_b", "directionality", "classification", "interacting_cells",
        "sender_cell_type", "receiver_cell_type", "significant_mean",
        "ligand_deg", "receptor_deg",
    ]
    return long[[column for column in keep if column in long]].drop_duplicates()


def run_cpdb(output: Path, threads: int) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result_path = output / "dataset_lr_pairs.csv"
    if manifest.get("status") == "complete" and result_path.exists():
        logging.info("Skipping completed CellPhoneDB run: %s", output)
        return
    results = cpdb_degs_analysis_method.call(
        cpdb_file_path=str(CPDB_FILE),
        meta_file_path=str(output / "metadata.tsv"),
        counts_file_path=str(output / "counts.h5ad"),
        degs_file_path=str(output / "degs.tsv"),
        counts_data="hgnc_symbol",
        threshold=0.1,
        result_precision=3,
        output_path=str(output),
        output_suffix="dataset",
        score_interactions=False,
        threads=threads,
    )
    degs = pd.read_csv(output / "degs.tsv", sep="\t")
    make_long_results(results, degs).to_csv(result_path, index=False)
    manifest["status"] = "complete"
    manifest["interactions"] = len(pd.read_csv(result_path))
    manifest_path.write_text(json.dumps(manifest, indent=2))


def graph_nodes(time: str, cluster: int) -> set[str]:
    folder = GRAPH_DIR / TIME_FOLDERS[time] / f"cluster_{cluster:02d}"
    for path in folder.glob("*.txt"):
        if f"net_{time}_leidenwu_cluster" in path.name and "pearson" in path.name:
            graph = nx.read_edgelist(path, delimiter="\t", nodetype=str, data=False)
            return set(graph.nodes)
    logging.warning("No graph found for %s cluster %s; consensus will not be graph-filtered", time, cluster)
    return set()


def aggregate_cluster(time: str, cluster: int) -> None:
    cluster_dir = OUTPUT_DIR / TIME_FOLDERS[time] / f"cluster_{cluster:02d}"
    dataset_dirs = sorted((cluster_dir / "datasets").glob("*"))
    manifests = []
    interactions = []
    support_rows = []
    for directory in dataset_dirs:
        manifest_path = directory / "manifest.json"
        result_path = directory / "dataset_lr_pairs.csv"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        manifests.append(manifest)
        for cell_type in manifest.get("cell_types", []):
            support_rows.append({"dataset": manifest["dataset"], "cell_type": cell_type})
        if manifest.get("status") == "complete" and result_path.exists():
            frame = pd.read_csv(result_path)
            frame["dataset"] = manifest["dataset"]
            interactions.append(frame)

    pd.DataFrame(manifests).to_csv(cluster_dir / "dataset_run_summary.csv", index=False)
    support = pd.DataFrame(support_rows)
    support.to_csv(cluster_dir / "cell_type_dataset_support.csv", index=False)
    if not interactions:
        logging.warning("No completed interactions to aggregate for %s cluster %s", time, cluster)
        return

    all_interactions = pd.concat(interactions, ignore_index=True)
    nodes = graph_nodes(time, cluster)
    if nodes:
        all_interactions = all_interactions[
            all_interactions["sender_cell_type"].isin(nodes)
            & all_interactions["receiver_cell_type"].isin(nodes)
        ].copy()
    all_interactions["dataset_percentile_rank"] = all_interactions.groupby("dataset")[
        "significant_mean"
    ].rank(pct=True)
    key = ["interacting_pair", "sender_cell_type", "receiver_cell_type"]

    eligible = {}
    for _, row in all_interactions[key].drop_duplicates().iterrows():
        eligible[tuple(row)] = support.groupby("dataset")["cell_type"].apply(
            lambda values: row["sender_cell_type"] in set(values)
            and row["receiver_cell_type"] in set(values)
        ).sum()
    grouped = (
        all_interactions.groupby(key, observed=True)
        .agg(
            supporting_datasets=("dataset", "nunique"),
            consensus_score=("dataset_percentile_rank", "mean"),
            mean_significant_mean=("significant_mean", "mean"),
            ligand_deg=("ligand_deg", "max"),
            receptor_deg=("receptor_deg", "max"),
            datasets=("dataset", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
    )
    grouped["eligible_datasets"] = [eligible[tuple(row)] for _, row in grouped[key].iterrows()]
    grouped["support_fraction"] = grouped["supporting_datasets"] / grouped["eligible_datasets"]
    grouped = grouped.sort_values(
        ["supporting_datasets", "support_fraction", "consensus_score"],
        ascending=False,
    )
    consensus = grouped[
        (grouped["supporting_datasets"] >= 2) & (grouped["support_fraction"] >= 0.5)
    ]
    single = grouped[grouped["eligible_datasets"] == 1]
    consensus.to_csv(cluster_dir / "consensus_lr_pairs.csv", index=False)
    single.to_csv(cluster_dir / "single_dataset_evidence.csv", index=False)


def cluster_route_activity(time: str, cluster: int) -> pd.DataFrame:
    """Calculate route activity, counting testable-but-absent datasets as zero."""
    cluster_dir = OUTPUT_DIR / TIME_FOLDERS[time] / f"cluster_{cluster:02d}"
    dataset_dirs = sorted((cluster_dir / "datasets").glob("*"))
    interactions = []
    cell_types_by_dataset = {}
    nodes = graph_nodes(time, cluster)
    for directory in dataset_dirs:
        manifest_path = directory / "manifest.json"
        result_path = directory / "dataset_lr_pairs.csv"
        if not manifest_path.exists() or not result_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            continue
        dataset = manifest["dataset"]
        cell_types_by_dataset[dataset] = set(manifest.get("cell_types", []))
        frame = pd.read_csv(result_path)
        if nodes:
            frame = frame[
                frame["sender_cell_type"].isin(nodes)
                & frame["receiver_cell_type"].isin(nodes)
            ].copy()
        frame["dataset"] = dataset
        frame["dataset_percentile_rank"] = frame["significant_mean"].rank(pct=True)
        interactions.append(frame)
    if not interactions:
        return pd.DataFrame()

    all_interactions = pd.concat(interactions, ignore_index=True)
    key = ["interacting_pair", "sender_cell_type", "receiver_cell_type"]
    grouped = (
        all_interactions.groupby(key, observed=True)
        .agg(
            supporting_datasets=("dataset", "nunique"),
            rank_sum=("dataset_percentile_rank", "sum"),
            ligand_deg=("ligand_deg", "max"),
            receptor_deg=("receptor_deg", "max"),
            datasets=("dataset", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
    )
    grouped["eligible_datasets"] = [
        sum(
            row["sender_cell_type"] in cell_types
            and row["receiver_cell_type"] in cell_types
            for cell_types in cell_types_by_dataset.values()
        )
        for _, row in grouped.iterrows()
    ]
    grouped["prevalence"] = grouped["supporting_datasets"] / grouped["eligible_datasets"]
    grouped["activity"] = grouped["rank_sum"] / grouped["eligible_datasets"]
    grouped["time"] = time
    grouped["cluster"] = cluster
    return grouped


def completed_cell_types(time: str, cluster: int) -> dict[str, set[str]]:
    cluster_dir = OUTPUT_DIR / TIME_FOLDERS[time] / f"cluster_{cluster:02d}"
    result = {}
    for manifest_path in (cluster_dir / "datasets").glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            result[manifest["dataset"]] = set(manifest.get("cell_types", []))
    return result


def complete_route_universe(
    routes: pd.DataFrame,
    universe: pd.DataFrame,
    cell_types_by_dataset: dict[str, set[str]],
    time: str,
    cluster: int,
) -> pd.DataFrame:
    """Add zero-activity rows for routes testable but absent in a cluster."""
    key = ["interacting_pair", "sender_cell_type", "receiver_cell_type"]
    completed = universe.copy()
    completed["eligible_datasets"] = [
        sum(
            row["sender_cell_type"] in cell_types
            and row["receiver_cell_type"] in cell_types
            for cell_types in cell_types_by_dataset.values()
        )
        for _, row in completed.iterrows()
    ]
    completed = completed[completed["eligible_datasets"] > 0]
    if not routes.empty:
        observed = routes.drop(columns=["eligible_datasets"], errors="ignore")
        completed = completed.merge(observed, on=key, how="left")
    for column, default in {
        "supporting_datasets": 0,
        "rank_sum": 0.0,
        "ligand_deg": False,
        "receptor_deg": False,
        "datasets": "",
        "prevalence": 0.0,
        "activity": 0.0,
    }.items():
        if column not in completed:
            completed[column] = default
        else:
            completed[column] = completed[column].fillna(default)
    completed["time"] = time
    completed["cluster"] = cluster
    return completed


def score_cluster_specificity(
    target: pd.DataFrame,
    others: list[pd.DataFrame],
    key: list[str],
) -> pd.DataFrame:
    """Score target activity against the strongest testable same-stage cluster."""
    if target.empty:
        return target
    result = target.copy()
    other_activity_columns = []
    other_eligible_columns = []
    for other in others:
        if other.empty:
            continue
        cluster = int(other["cluster"].iloc[0])
        comparison = other[key + ["activity", "eligible_datasets"]].rename(
            columns={
                "activity": f"activity_cluster_{cluster}",
                "eligible_datasets": f"eligible_cluster_{cluster}",
            }
        )
        result = result.merge(comparison, on=key, how="left")
        other_activity_columns.append(f"activity_cluster_{cluster}")
        other_eligible_columns.append(f"eligible_cluster_{cluster}")

    if not other_activity_columns:
        result["testable_other_clusters"] = 0
        result["max_other_activity"] = np.nan
        result["specificity"] = np.nan
        return result

    result["testable_other_clusters"] = result[other_eligible_columns].notna().sum(axis=1)
    result["max_other_activity"] = result[other_activity_columns].fillna(0).max(axis=1)
    result["specificity"] = result["activity"] - result["max_other_activity"]
    return result


def pair_activity_from_routes(routes: pd.DataFrame) -> pd.DataFrame:
    """Represent each LR pair by its strongest directional route in a cluster."""
    if routes.empty:
        return routes
    strongest = routes.sort_values("activity", ascending=False).drop_duplicates(
        "interacting_pair"
    )
    return strongest[
        [
            "interacting_pair", "sender_cell_type", "receiver_cell_type",
            "supporting_datasets", "eligible_datasets", "prevalence", "activity",
            "ligand_deg", "receptor_deg", "datasets", "time", "cluster",
        ]
    ].rename(
        columns={
            "sender_cell_type": "best_sender_cell_type",
            "receiver_cell_type": "best_receiver_cell_type",
        }
    )


def aggregate_specificity(time: str) -> None:
    """Write route- and LR-pair-level same-stage cluster specificity rankings."""
    observed_route_tables = {
        cluster: cluster_route_activity(time, cluster)
        for cluster in (0, 1, 2)
    }
    nonempty = [table for table in observed_route_tables.values() if not table.empty]
    if not nonempty:
        return
    route_key = ["interacting_pair", "sender_cell_type", "receiver_cell_type"]
    universe = pd.concat([table[route_key] for table in nonempty]).drop_duplicates()
    route_tables = {
        cluster: complete_route_universe(
            observed_route_tables[cluster],
            universe,
            completed_cell_types(time, cluster),
            time,
            cluster,
        )
        for cluster in (0, 1, 2)
    }
    pair_tables = {
        cluster: pair_activity_from_routes(routes)
        for cluster, routes in route_tables.items()
    }
    pair_key = ["interacting_pair"]

    for cluster in (0, 1, 2):
        cluster_dir = OUTPUT_DIR / TIME_FOLDERS[time] / f"cluster_{cluster:02d}"
        if route_tables[cluster].empty:
            continue
        routes = score_cluster_specificity(
            route_tables[cluster],
            [table for other, table in route_tables.items() if other != cluster],
            route_key,
        ).sort_values(["specificity", "activity", "prevalence"], ascending=False)
        pairs = score_cluster_specificity(
            pair_tables[cluster],
            [table for other, table in pair_tables.items() if other != cluster],
            pair_key,
        ).sort_values(["specificity", "activity", "prevalence"], ascending=False)

        route_specific = routes[
            (routes["supporting_datasets"] >= 2)
            & (routes["prevalence"] >= 0.5)
            & (routes["specificity"] >= 0.2)
            & (routes["testable_other_clusters"] >= 1)
        ]
        pair_specific = pairs[
            (pairs["supporting_datasets"] >= 2)
            & (pairs["prevalence"] >= 0.5)
            & (pairs["specificity"] >= 0.2)
            & (pairs["testable_other_clusters"] >= 1)
        ]
        routes.to_csv(cluster_dir / "directional_route_specificity_all.csv", index=False)
        route_specific.to_csv(
            cluster_dir / "directional_route_specificity_filtered.csv", index=False
        )
        route_specific.head(10).to_csv(
            cluster_dir / "directional_route_specificity_top10.csv", index=False
        )
        pairs.to_csv(cluster_dir / "lr_pair_specificity_all.csv", index=False)
        pair_specific.to_csv(cluster_dir / "lr_pair_specificity_filtered.csv", index=False)
        pair_specific.head(10).to_csv(
            cluster_dir / "lr_pair_specificity_top10.csv", index=False
        )


def record_failure(time: str, cluster: int, dataset: str, error: Exception) -> None:
    cluster_dir = OUTPUT_DIR / TIME_FOLDERS[time] / f"cluster_{cluster:02d}"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    path = cluster_dir / "skipped_runs.csv"
    row = pd.DataFrame([{
        "time": time,
        "cluster": cluster,
        "dataset": dataset,
        "reason": str(error),
        "traceback": traceback.format_exc(),
    }])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    global MARKER_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=["prepare", "run", "aggregate", "specificity", "all"],
        nargs="?",
        default="all",
    )
    parser.add_argument("--time", choices=["I-II", "III-IV"], action="append")
    parser.add_argument("--cluster", choices=[0, 1, 2], type=int, action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--auc-threshold", type=float, default=0.65)
    parser.add_argument("--patient-cap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--min-patients", type=int, default=2)
    parser.add_argument("--threads", type=int, default=7)
    parser.add_argument("--marker-root", type=Path, default=MARKER_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover eligible dataset runs without creating or modifying outputs.",
    )
    args = parser.parse_args()
    MARKER_DIR = args.marker_root
    OUTPUT_DIR = args.output_root
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    runs = discover_runs(args.time, args.cluster, args.dataset)
    logging.info("Discovered %d eligible dataset runs", len(runs))
    if args.dry_run:
        for time, cluster, dataset in runs:
            print(f"{time}\tcluster_{cluster:02d}\t{dataset}")
        return
    touched_clusters = set()
    for time, cluster, dataset in runs:
        touched_clusters.add((time, cluster))
        try:
            output = (
                OUTPUT_DIR / TIME_FOLDERS[time] / f"cluster_{cluster:02d}" / "datasets" / dataset
            )
            if args.phase in {"prepare", "all"}:
                output = prepare_run(
                    time, cluster, dataset, args.auc_threshold, args.patient_cap,
                    args.seed, args.min_cells, args.min_patients,
                )
            if args.phase in {"run", "all"}:
                run_cpdb(output, args.threads)
        except Exception as error:
            logging.exception("Failed %s cluster %s dataset %s", time, cluster, dataset)
            record_failure(time, cluster, dataset, error)

    if args.phase in {"aggregate", "all"}:
        for time, cluster in sorted(touched_clusters):
            aggregate_cluster(time, cluster)
    if args.phase in {"specificity", "all"}:
        times = args.time or ["I-II", "III-IV"]
        for time in times:
            aggregate_specificity(time)


if __name__ == "__main__":
    main()
