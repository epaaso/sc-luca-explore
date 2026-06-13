# Marker And Ligand-Receptor Consensus

This directory contains compact scientific summaries from the June 12, 2026
analysis. Full inputs, intermediates, outputs, logs, and plots are in:

```text
/datos/migccl/neto_maestria/luca_explore/analysis_archive/marker_lr_consensus_20260612/
```

Markers are calculated separately by dataset, stage, and ARACNE patient cluster.
The retained contrasts are `normal-vs-normal` and `tumor-vs-all`. Consensus
includes genes with AUC >= 0.65 and reports top-10 rankings with dataset support,
eligible datasets, mean/min/max AUC, and support fraction.

CellPhoneDB uses the dataset-stratified DEG method. Per-dataset evidence is
combined into cross-dataset consensus and ranked for same-stage cluster
specificity. Filtered specificity requires at least two supporting datasets,
prevalence >= 0.5, specificity >= 0.2, and at least one testable other cluster.

Clusters with few patients or insufficient cell-type support are skipped.
Specificity is relative to other testable clusters in the same stage and does
not establish a causal interaction.

## Reproduction

```bash
python nb_DE_wilcox/run_marker_extraction.py --dry-run
python nb_DE_wilcox/run_marker_extraction.py \
  --contrast normal-vs-normal --contrast tumor-vs-all \
  --output-root /datos/migccl/neto_maestria/luca_explore/analysis_runs/marker_lr_consensus/marker_genes

python nb_cellphoneDB/run_cellphone_dataset_consensus.py --dry-run \
  --marker-root /datos/migccl/neto_maestria/luca_explore/analysis_runs/marker_lr_consensus/marker_genes
python nb_cellphoneDB/run_cellphone_dataset_consensus.py all \
  --marker-root /datos/migccl/neto_maestria/luca_explore/analysis_runs/marker_lr_consensus/marker_genes \
  --output-root /datos/migccl/neto_maestria/luca_explore/analysis_runs/marker_lr_consensus/cellphonedb
```

The archive manifest records the exact June 12 parameters, software versions,
source inventory, and checksums. Future runs must use a new run ID.
