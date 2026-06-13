"""Canonical repository and production-storage paths."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR = REPO_ROOT / "metadata"
CELL_MAPPINGS = METADATA_DIR / "cell_mappings.json"

RESULTS_DIR = REPO_ROOT / "results"
GRAPH_GENERATION_RESULTS = RESULTS_DIR / "graph_generation"
GLOBAL_GRAPH_RESULTS = GRAPH_GENERATION_RESULTS / "global"
CLUSTER_GRAPH_RESULTS = GRAPH_GENERATION_RESULTS / "cluster_runs"
LEGACY_CLUSTER_GRAPH_RESULTS = GRAPH_GENERATION_RESULTS / "legacy" / "cluster_runs2"
ECOTYPE_CLUSTERING_RESULTS = RESULTS_DIR / "ecotype_clustering"
GRAPH_ANALYSIS_RESULTS = RESULTS_DIR / "graph_analysis"
MARKER_GENE_RESULTS = RESULTS_DIR / "marker_genes"
CELLPHONEDB_RESULTS = RESULTS_DIR / "cellphonedb"
OPTIONAL_DE_SCT_RESULTS = RESULTS_DIR / "optional" / "de_sct"

EARLY_MEMBERSHIP = ECOTYPE_CLUSTERING_RESULTS / "membership_by_cluster_early.csv"
LATE_MEMBERSHIP = ECOTYPE_CLUSTERING_RESULTS / "membership_by_cluster_late.csv"
LATE2_MEMBERSHIP = ECOTYPE_CLUSTERING_RESULTS / "membership_by_cluster_late2.csv"
MEMBERSHIP_FILES = {
    "I-II": EARLY_MEMBERSHIP,
    "III-IV": LATE_MEMBERSHIP,
}

# Full production runs intentionally stay outside the repository.
EXTERNAL_DATA_DIR = Path("/datos/migccl/neto_maestria/luca_explore/surgeries")
EXTERNAL_ANALYSIS_ROOT = Path(
    "/datos/migccl/neto_maestria/luca_explore/analysis_runs/marker_lr_consensus"
)
EXTERNAL_MARKER_ROOT = EXTERNAL_ANALYSIS_ROOT / "marker_genes"
EXTERNAL_CELLPHONEDB_ROOT = EXTERNAL_ANALYSIS_ROOT / "cellphonedb"
EXTERNAL_CELLPHONEDB_DATABASE = Path(
    "/datos/migccl/neto_maestria/luca_explore/cellphoneDB/db/v5.0.0/cellphonedb.zip"
)
