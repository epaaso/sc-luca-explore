# Agent Notes For This Repo

This repo contains the exploratory and analytical notebooks/scripts for the LUCA single-cell pipeline.

## Architectural Insights & Future Work

### Per-Cluster CellphoneDB & Differential Expression
**Canonical State**:
- `nb_DE_wilcox/run_marker_extraction.py` computes explicit `normal-vs-normal` and
  `tumor-vs-all` marker contrasts separately by dataset, stage, and ARACNE
  patient cluster.
- `nb_cellphoneDB/run_cellphone_dataset_consensus.py` runs CellPhoneDB's dataset-stratified DEG
  method for each cluster and builds cross-dataset consensus and same-stage
  cluster-specificity rankings.
- Full production outputs belong in external storage. Compact summaries are in
  `results/marker_genes/` and `results/cellphonedb/`.
