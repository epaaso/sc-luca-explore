# Graph Generation

This phase is entered twice. `grouping_MI.ipynb` first creates global
sample-by-cell-type matrices and ARACNe networks in
`results/graph_generation/global/`. After ecotype clustering creates membership
CSVs in `results/ecotype_clustering/`, the per-cluster scripts generate and
post-process networks in `results/graph_generation/cluster_runs/`.

```bash
jupyter lab nb_graphGeneration/grouping_MI.ipynb
python nb_graphGeneration/run_aracne_by_cluster.py --dry-run
python nb_graphGeneration/run_aracne_by_cluster.py
python nb_graphGeneration/postprocess_cluster_networks.py
```

`pearson_compare.ipynb`, `plot_cell_pair.py`, and `change_cells.py` support this
phase. Historical second-pass cluster runs are retained under
`results/graph_generation/legacy/cluster_runs2/`.
