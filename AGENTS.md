# Agent Notes For This Repo

This repo contains the exploratory and analytical notebooks/scripts for the LUCA single-cell pipeline.

## Architectural Insights & Future Work

### Per-Cluster CellphoneDB & Differential Expression
**Current State**: 
- `nb_DE_wilcox/modal_DE.py` computes differential expression (AUC scores/marker genes) for each cell type globally across all patients in a given stage (e.g., III-IV).
- `run_cellphone_full.py` runs CellphoneDB globally using this aggregated data, and then filters the resulting significant ligand-receptor pairs for each ARACNE patient cluster based on which cell types co-occur in that cluster network.

**The Flaw**:
An ARACNE cluster represents a specific subgroup/ecotype of patients. Because we use the global marker genes, we assume a cell type (like a Macrophage) has the exact same expression profile across all patient clusters. This ignores the intra-cluster biology that distinguishes these patient groups.

**Next Steps / Required Fix**:
To correctly compute ligand-receptor interactions for each ARACNE cluster:
1. Divide the single-cell expression data based on the patient ARACNE clusters (using `nb_graphAnalysis/output/membership_by_cluster_*.csv`).
2. Run the differential expression logic (`modal_DE.py` / `aggregate_markers.py`) *separately* for the cells within each patient cluster to generate cluster-specific `auc_count_cellphonedb` matrices.
3. Run CellphoneDB separately for each cluster, using its own distinct marker gene expression profile.
