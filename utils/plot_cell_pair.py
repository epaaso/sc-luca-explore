"""Plot scatter plots for a pair of cells in a specific cluster/stage.
Generates two plots of Relative Abundance:
1. Cell1 vs Cell2 with Log Y axis (Linear X).
2. Cell1 vs Cell2 with Log X axis (Linear Y).

Usage:
    python utils/plot_cell_pair.py --stage [early|late] --cluster [ID] --cell1 "CellA" --cell2 "CellB"
"""

import argparse
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration mirrored from postprocess_cluster_networks.py
REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = REPO_ROOT / "metadata"

CELL_TYPE_KEY = "cell_type_adjusted"
SAMPLE_KEY = "sample"
COUNT_COLUMN = "0"

class StageConfig:
    def __init__(self, name, membership_csv, time_label):
        self.name = name
        self.membership_csv = membership_csv
        self.time_label = time_label

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

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_data(stage_name, cluster_id):
    if stage_name not in STAGE_CONFIGS:
        raise ValueError(f"Unknown stage: {stage_name}. Choose from {list(STAGE_CONFIGS.keys())}")
    
    stage = STAGE_CONFIGS[stage_name]
    
    logging.info(f"Loading membership for stage {stage_name}...")
    # Load membership
    if not stage.membership_csv.exists():
        raise FileNotFoundError(f"Membership file not found: {stage.membership_csv}")
    
    membership_df = pd.read_csv(stage.membership_csv)
    
    # Filter for cluster
    cluster_samples = membership_df.loc[membership_df["membership"] == cluster_id, SAMPLE_KEY]
    if cluster_samples.empty:
         raise ValueError(f"No samples found for cluster {cluster_id} in stage {stage_name}")
         
    cluster_samples = cluster_samples.astype(str).tolist()
    logging.info(f"Found {len(cluster_samples)} samples in cluster {cluster_id}")

    # Load groups/counts
    groups_path = METADATA_DIR / f"groups_{stage.time_label}_funcnames.csv"
    logging.info(f"Loading groups data from {groups_path}...")
    if not groups_path.exists():
        raise FileNotFoundError(f"Groups file not found: {groups_path}")
        
    groups = pd.read_csv(groups_path)
    # Check if we need to cast sample to string to match membership
    groups[SAMPLE_KEY] = groups[SAMPLE_KEY].astype(str)
    
    # Filter by samples in the cluster
    groups = groups[groups[SAMPLE_KEY].isin(cluster_samples)]
    
    if groups.empty:
        raise ValueError("No matching data found in groups file for the selected clusters/samples.")
        
    # Pivot to sample x cell_type matrix
    if COUNT_COLUMN not in groups.columns:
         raise ValueError(f"Count column '{COUNT_COLUMN}' not found in groups file.")

    matrix_counts = (
        groups.pivot_table(index=SAMPLE_KEY, columns=CELL_TYPE_KEY, values=COUNT_COLUMN, aggfunc="sum")
        .fillna(0)
    )
    
    # Get datasets for coloring
    dataset_map = groups[[SAMPLE_KEY, "dataset"]].drop_duplicates().set_index(SAMPLE_KEY)["dataset"]
    
    return matrix_counts, dataset_map

def plot_pair(df, cell1, cell2, output_path, dataset_map, axis_label):
    # Verify cells exist
    available_cells = set(df.columns)
    if cell1 not in available_cells:
        raise ValueError(f"Cell '{cell1}' not found in dataset. Available cells: {list(available_cells)[:5]}...")
    if cell2 not in available_cells:
        raise ValueError(f"Cell '{cell2}' not found in dataset.")

    x = df[cell1]
    y = df[cell2]
    
    # Prepare colors
    datasets = dataset_map.reindex(df.index).fillna("unknown")
    unique_datasets = sorted(datasets.unique())
    cmap = plt.get_cmap("tab10", len(unique_datasets) if len(unique_datasets) > 0 else 1)
    color_map = {d: cmap(i) for i, d in enumerate(unique_datasets)}
    colors = datasets.map(color_map)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Log Y, Linear X
    sc1 = ax1.scatter(x, y, c=colors, alpha=0.7, s=30, edgecolor='k', linewidth=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('linear')
    ax1.set_xlabel(f"{cell1} ({axis_label})")
    ax1.set_ylabel(f"{cell2} ({axis_label}) [Log]")
    ax1.set_title(f"{cell1} vs {cell2}\n(Log Y-axis)")
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot 2: Log X, Linear Y
    sc2 = ax2.scatter(x, y, c=colors, alpha=0.7, s=30, edgecolor='k', linewidth=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('linear')
    ax2.set_xlabel(f"{cell1} ({axis_label}) [Log]")
    ax2.set_ylabel(f"{cell2} ({axis_label})")
    ax2.set_title(f"{cell1} vs {cell2}\n(Log X-axis)")
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add legend
    if unique_datasets:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[d], label=d, markersize=10) for d in unique_datasets]
        # Place legend to the right of the plots
        fig.legend(handles=handles, loc='center left', title="Dataset", bbox_to_anchor=(0.85, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    logging.info(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot cell pair correlations (Relative Abundance) for a specific cluster/stage.")
    parser.add_argument("--stage", required=True, choices=["early", "late"], help="Cancer stage (early=I-II, late=III-IV)")
    parser.add_argument("--cluster", required=True, type=int, help="Cluster ID")
    parser.add_argument("--cell1", required=True, help="Name of the first cell type")
    parser.add_argument("--cell2", required=True, help="Name of the second cell type")
    parser.add_argument("--output", default=None, help="Output image file path (default: auto-generated from inputs)")
    parser.add_argument("--no-plus1", action="store_true", help="Do not add 1 to counts before normalization. Zeros will not be shown on log scales.")
    
    args = parser.parse_args()
    setup_logging()

    if args.output is None:
        safe_cell1 = args.cell1.replace(" ", "_").replace("/", "-")
        safe_cell2 = args.cell2.replace(" ", "_").replace("/", "-")
        args.output = f"{safe_cell1}_vs_{safe_cell2}_{args.stage}_cluster{args.cluster}.png"
    
    try:
        matrix_counts, dataset_map = load_data(args.stage, args.cluster)

        if args.no_plus1:
            # Relative Abundance of raw counts
            row_sums = matrix_counts.sum(axis=1).replace(0, np.nan)
            df = matrix_counts.div(row_sums, axis=0).fillna(0)
            axis_label = "Rel. Abundance"
        else:
            # Relative Abundance of (Counts + 1)
            # Standard "log1p" style normalization practice
            counts_plus1 = matrix_counts + 1
            row_sums = counts_plus1.sum(axis=1).replace(0, np.nan)
            df = counts_plus1.div(row_sums, axis=0).fillna(0)
            axis_label = "Rel. Abundance (Counts+1)"

        plot_pair(df, args.cell1, args.cell2, args.output, dataset_map, axis_label)
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
