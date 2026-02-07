"""Automate ARACNe-AP mutual information runs for LUCA clusters.

This script reproduces the last section of `grouping_MI.ipynb` by taking the
pre-computed sample-by-type matrices stored under `outputARACNE/` and running
ARACNe-AP for every community assignment present in the
`nb_graphAnalysis/output/membership_by_cluster_*.csv` files. Each cluster run
receives its own subfolder inside `outputARACNE/cluster_runs/<stage>/` with the
matrix, transcription factor list, ARACNe outputs, and the final edge list with
the header removed (matching the behaviour in the notebook).

Example usage (dry run for cluster 0 early stage):

    python utils/run_aracne_by_cluster.py --stages early --clusters 0 --dry-run

Run all stages and clusters with 200 bootstraps and 2 GiB of JVM memory:

    python utils/run_aracne_by_cluster.py --bootstraps 100 --java-memory 1G

Note: Running ARACNe with 500 bootstraps per cluster can take a long time.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class StageConfig:
    name: str
    membership_csv: Path
    matrix_path: Path
    time_label: str  # e.g. "I-II_leidenwu"


DEFAULT_OUTPUT_ROOT = Path("outputARACNE")
STAGE_CONFIGS = {
    "early": StageConfig(
        name="early",
        membership_csv=Path("nb_graphAnalysis/output/membership_by_cluster_early.csv"),
        matrix_path=DEFAULT_OUTPUT_ROOT / "matrixI-II_leidenwu_funcnames.txt",
        time_label="I-II_leidenwu",
    ),
    "late": StageConfig(
        name="late",
        membership_csv=Path("nb_graphAnalysis/output/membership_by_cluster_late.csv"),
        matrix_path=DEFAULT_OUTPUT_ROOT / "matrixIII-IV_leidenwu_funcnames.txt",
        time_label="III-IV_leidenwu",
    ),
    "late2": StageConfig(
        name="late",
        membership_csv=Path("nb_graphAnalysis/output/membership_by_cluster_late2.csv"),
        matrix_path=DEFAULT_OUTPUT_ROOT / "matrixIII-IV_leidenwu_funcnames.txt",
        time_label="III-IV_leidenwu",
    ),
}


# ---------------------------------------------------------------------------
# Utility helpers


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ARACNe-AP MI inference per cluster for early/late stages.",
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
        "--min-samples",
        type=int,
        default=5,
        help="Minimum number of samples required in a cluster to run ARACNe.",
    )
    parser.add_argument(
        "--min-nonzero-types",
        type=int,
        default=3,
        help="Minimum number of cell types with non-zero counts required after filtering.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=100,
        help="Number of bootstrap runs per cluster (the notebook used 500).",
    )
    parser.add_argument(
        "--pvalue",
        default="1E-5",
        help="P-value threshold passed to ARACNe.",
    )
    parser.add_argument(
        "--java-memory",
        default="1G",
        help="Maximum heap size for the JVM, passed as -Xmx (e.g. 5G, 8192m).",
    )
    parser.add_argument(
        "--jar",
        type=Path,
        default=Path("/root/host_home/ARACNe-AP/dist/aracne.jar"),
        help="Path to the ARACNe-AP jar file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "cluster_runs",
        help="Directory where per-cluster outputs will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run clusters even if a final network file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare inputs but skip all Java executions (useful for testing).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity to warnings only.",
    )

    return parser.parse_args(argv)


def setup_logging(quiet: bool = False) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def read_membership_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Membership file not found: {path}")
    df = pd.read_csv(path)
    required = {"sample", "membership"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Membership CSV {path} missing columns: {missing}")
    return df


def read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    return df


def ensure_java_available() -> None:
    if shutil.which("java") is None:
        raise RuntimeError("Java executable not found on PATH; required to run ARACNe.")


def run_command(cmd: Sequence[str], *, cwd: Path) -> None:
    logging.debug("Running command: %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)


def subset_cluster_samples(
    matrix: pd.DataFrame,
    samples: Sequence[str],
) -> pd.DataFrame:
    available = [sample for sample in samples if sample in matrix.columns]
    missing = sorted(set(samples) - set(available))
    if missing:
        logging.warning("%d samples missing from matrix: %s", len(missing), ", ".join(missing))
    if not available:
        logging.warning("No samples available for this cluster after filtering.")
        return pd.DataFrame()
    cluster_df = matrix.loc[:, available]
    cluster_df = cluster_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    cluster_df = cluster_df.loc[cluster_df.sum(axis=1) > 0]
    return cluster_df


def build_aracne_matrix(cluster_df: pd.DataFrame) -> pd.DataFrame:
    cluster_df = cluster_df.copy()
    cluster_df.index = cluster_df.index.astype(str)
    cluster_df.index.name = "types"
    return cluster_df


def write_matrix_and_tfs(
    cluster_dir: Path,
    stage: StageConfig,
    cluster_id: int,
    matrix_for_aracne: pd.DataFrame,
) -> tuple[Path, Path, List[str]]:
    matrix_name = f"matrix_{stage.time_label}_cluster{cluster_id}.txt"
    matrix_path = cluster_dir / matrix_name
    matrix_for_aracne.to_csv(matrix_path, sep="\t")

    tf_list = matrix_for_aracne.index.tolist()
    tf_path = cluster_dir / f"tfs_{stage.time_label}_cluster{cluster_id}.txt"
    with tf_path.open("w") as fh:
        for tf in tf_list:
            fh.write(f"{tf}\n")

    return matrix_path, tf_path, list(tf_list)


def derive_cluster_ids(df: pd.DataFrame) -> List[int]:
    return sorted(df["membership"].astype(int).unique())


def filter_cluster_list(
    all_clusters: Iterable[int],
    selection: Optional[Sequence[int]],
) -> List[int]:
    if selection is None:
        return list(all_clusters)
    sel = sorted(set(selection))
    return [cluster for cluster in all_clusters if cluster in sel]


def generate_stage_output_dir(output_root: Path, stage: StageConfig) -> Path:
    stage_dir = output_root / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def existing_network_path(cluster_dir: Path, stage: StageConfig, cluster_id: int) -> Path:
    return cluster_dir / f"net_{stage.time_label}_cluster{cluster_id}.txt"


def clean_bootstrap_artifacts(cluster_dir: Path) -> None:
    for path in cluster_dir.glob("bootstrapNetwork_*"):
        path.unlink(missing_ok=True)


def write_network_without_header(network_path: Path, dest_path: Path) -> None:
    if not network_path.exists():
        raise FileNotFoundError(f"Expected ARACNe network output missing: {network_path}")
    with network_path.open("r") as src, dest_path.open("w") as dst:
        next(src, None)  # skip header
        for line in src:
            dst.write(line)


def run_aracne_for_cluster(
    stage: StageConfig,
    cluster_dir: Path,
    matrix_path: Path,
    tf_path: Path,
    args: argparse.Namespace,
) -> None:
    jar_path = args.jar.resolve()
    if not jar_path.exists():
        raise FileNotFoundError(f"ARACNe jar not found at {jar_path}")

    cluster_dir_abs = cluster_dir.resolve()
    matrix_abs = matrix_path.resolve()
    tf_abs = tf_path.resolve()

    cmd_base = [
        "java",
        f"-Xmx{args.java_memory}",
        "-jar",
        str(jar_path),
        "-e",
        str(matrix_abs),
        "-o",
        str(cluster_dir_abs),
        "--tfs",
        str(tf_abs),
        "--pvalue",
        str(args.pvalue),
    ]

    threshold_cmd = cmd_base + ["--seed", "1", "--calculateThreshold"]
    run_command(threshold_cmd, cwd=cluster_dir_abs)

    for seed in range(1, args.bootstraps + 1):
        bootstrap_cmd = cmd_base + ["--seed", str(seed)]
        run_command(bootstrap_cmd, cwd=cluster_dir_abs)

    consolidate_cmd = [
        "java",
        f"-Xmx{args.java_memory}",
        "-jar",
        str(jar_path),
        "-o",
        str(cluster_dir_abs),
        "--consolidate",
    ]
    run_command(consolidate_cmd, cwd=cluster_dir_abs)


def process_stage(stage: StageConfig, args: argparse.Namespace) -> None:
    logging.info("Processing stage '%s'", stage.name)
    membership_df = read_membership_csv(stage.membership_csv)
    matrix_df = read_matrix(stage.matrix_path)

    cluster_ids = derive_cluster_ids(membership_df)
    cluster_ids = filter_cluster_list(cluster_ids, args.clusters)
    logging.info("Stage '%s' clusters: %s", stage.name, ", ".join(map(str, cluster_ids)))

    stage_output_dir = generate_stage_output_dir(args.output_root, stage)
    if not args.dry_run:
        ensure_java_available()

    for cluster_id in cluster_ids:
        logging.info("Stage '%s' cluster %d", stage.name, cluster_id)
        cluster_output_dir = stage_output_dir / f"cluster_{cluster_id:02d}"
        cluster_output_dir.mkdir(parents=True, exist_ok=True)
        cluster_samples = membership_df.loc[
            membership_df["membership"] == cluster_id, "sample"
        ].tolist()
        cluster_matrix = subset_cluster_samples(matrix_df, cluster_samples)

        if cluster_matrix.shape[1] < args.min_samples:
            logging.warning(
                "Skipping cluster %d (only %d samples, need >= %d)",
                cluster_id,
                cluster_matrix.shape[1],
                args.min_samples,
            )
            continue

        if (cluster_matrix.sum(axis=1) > 0).sum() < args.min_nonzero_types:
            logging.warning(
                "Skipping cluster %d (fewer than %d non-zero cell types)",
                cluster_id,
                args.min_nonzero_types,
            )
            continue

        aracne_matrix = build_aracne_matrix(cluster_matrix)
        matrix_path, tf_path, tf_list = write_matrix_and_tfs(
            cluster_output_dir, stage, cluster_id, aracne_matrix
        )
        logging.info(
            "Prepared matrix (%d x %d) and TF list (%d entries)",
            aracne_matrix.shape[0],
            aracne_matrix.shape[1],
            len(tf_list),
        )

        final_network_path = existing_network_path(cluster_output_dir, stage, cluster_id)
        if final_network_path.exists() and not args.force:
            logging.info(
                "Final network already exists for cluster %d (use --force to re-run)",
                cluster_id,
            )
            continue

        if args.dry_run:
            logging.info(
                "Dry-run enabled; skipping ARACNe execution for cluster %d",
                cluster_id,
            )
            continue

        run_aracne_for_cluster(stage, cluster_output_dir, matrix_path, tf_path, args)

        network_txt = cluster_output_dir / "network.txt"
        write_network_without_header(network_txt, final_network_path)
        network_txt.unlink(missing_ok=True)
        clean_bootstrap_artifacts(cluster_output_dir)

        logging.info("Cluster %d network saved to %s", cluster_id, final_network_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.quiet)

    for stage_name in args.stages:
        stage = STAGE_CONFIGS[stage_name]
        process_stage(stage, args)

    logging.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
