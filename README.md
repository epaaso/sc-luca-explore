# LUCA Single-Cell Exploration

Exploratory notebooks and reproducible scripts for the LUCA single-cell
pipeline. Run commands from the repository root. Repository paths are defined
in `shared/paths.py`; full marker and CellPhoneDB production runs remain on
external storage.

## Canonical Workflow

1. **Data acquisition: `nb_dataAcquisition`**
   - Entrypoints: `get_data.ipynb`, `run_EmptyDrop.R`, `run_EmptyDrop_batch.R`
   - Prerequisites: dataset manifests and source download credentials
   - Produces: downloaded/raw data and EmptyDrops matrices on external storage
   - Command: `jupyter lab nb_dataAcquisition/get_data.ipynb`

2. **Quality filtering: `nb_filters`**
   - Entrypoint: `filter_param.ipynb` and dataset notebooks
   - Prerequisite: acquired count matrices
   - Produces: filtered AnnData files on external storage
   - Command: `jupyter lab nb_filters/filter_param.ipynb`

3. **Reference atlas training: `nb_refAtlas`**
   - Entrypoint: `scANVImodel.ipynb`
   - Prerequisite: filtered reference datasets
   - Produces: trained atlas model on external storage
   - Command: `jupyter lab nb_refAtlas/scANVImodel.ipynb`

4. **Dataset annotation: `nb_annotRefatlas`**
   - Entrypoint: `labelTransfer.ipynb` and dataset notebooks
   - Prerequisites: filtered datasets and reference atlas
   - Produces: transferred cell-type annotations on external storage
   - Command: `jupyter lab nb_annotRefatlas/labelTransfer.ipynb`

5. **Tumor subclustering: `nb_subCluster`**
   - Entrypoint: `Tumor_subcluster.ipynb`
   - Prerequisite: annotated tumor cells
   - Produces: tumor subcluster predictions on external storage
   - Command: `jupyter lab nb_subCluster/Tumor_subcluster.ipynb`

6. **Global coabundance and ARACNe generation: `nb_graphGeneration`**
   - Entrypoint: `grouping_MI.ipynb`
   - Prerequisite: cell annotations and sample metadata
   - Produces: `results/graph_generation/global/`
   - Command: `jupyter lab nb_graphGeneration/grouping_MI.ipynb`

7. **Patient/ecotype clustering: `nb_ecotypeClustering`**
   - Entrypoint: `abund_clusters.ipynb`
   - Prerequisite: global coabundance matrices
   - Produces: `results/ecotype_clustering/`
   - Command: `jupyter lab nb_ecotypeClustering/abund_clusters.ipynb`

8. **Per-cluster ARACNe and Pearson post-processing: `nb_graphGeneration`**
   - Entrypoints: `run_aracne_by_cluster.py`, `postprocess_cluster_networks.py`
   - Prerequisites: global matrices and ecotype membership CSVs
   - Produces: `results/graph_generation/cluster_runs/`
   - Commands:
     ```bash
     python nb_graphGeneration/run_aracne_by_cluster.py --dry-run
     python nb_graphGeneration/run_aracne_by_cluster.py
     python nb_graphGeneration/postprocess_cluster_networks.py
     ```

9. **Graph layouts and downstream analysis: `nb_graphAnalysis`**
   - Entrypoints: `graph_layouts.ipynb`, `sbm_cluster.ipynb`, `circos.ipynb`
   - Prerequisite: global and per-cluster graph outputs
   - Produces: `results/graph_analysis/`
   - Command: `jupyter lab nb_graphAnalysis/graph_layouts.ipynb`

10. **Cluster-specific marker extraction: `nb_DE_wilcox`**
    - Entrypoint: `run_marker_extraction.py`
    - Prerequisites: annotated AnnData files and ecotype membership CSVs
    - Produces: full external marker runs; compact summaries in
      `results/marker_genes/`
    - Commands:
      ```bash
      python nb_DE_wilcox/run_marker_extraction.py --dry-run
      python nb_DE_wilcox/run_marker_extraction.py \
        --contrast normal-vs-normal --contrast tumor-vs-all
      ```

11. **Dataset-consensus CellPhoneDB: `nb_cellphoneDB`**
    - Entrypoint: `run_cellphone_dataset_consensus.py`
    - Prerequisites: external marker runs, cluster graphs, and memberships
    - Produces: full external CellPhoneDB runs; compact summaries in
      `results/cellphonedb/`
    - Commands:
      ```bash
      python nb_cellphoneDB/run_cellphone_dataset_consensus.py --dry-run
      python nb_cellphoneDB/run_cellphone_dataset_consensus.py all
      ```

## Shared Code And Results

- `shared/paths.py`: canonical repository and external production paths
- `shared/functions.py`: helpers used across workflow phases
- `tools/cull-kernels.ipynb`: operational kernel cleanup
- `results/`: all tracked generated results
- `metadata/`: intentional static metadata and mappings

## Optional And Deprecated Workflows

- `nb_infercnv/`: optional InferCNV analysis
- `nb_ikarus/`: optional Ikarus analysis
- `nb_DE_SCT/`: optional SCT differential expression; tracked output is in
  `results/optional/de_sct/`
- `nb_tumorUMAP/`: optional tumor UMAP checks
- `nb_annot/`: older annotation notebooks
- Deprecated marker and global/AUC-as-expression CellPhoneDB outputs remain
  archived and are not restored.

The immutable June 12, 2026 marker/LR archive is documented in
`results/README.md`.
