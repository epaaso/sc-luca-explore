# Marker Analysis

`run_marker_extraction.py` computes explicit `normal-vs-normal` and
`tumor-vs-all` marker contrasts by dataset, stage, and ecotype cluster.
`modal_DE.py` is the analysis library; the notebooks are exploratory.

```bash
python nb_DE_wilcox/run_marker_extraction.py --dry-run
python nb_DE_wilcox/run_marker_extraction.py \
  --contrast normal-vs-normal \
  --contrast tumor-vs-all
pytest -q nb_DE_wilcox/test_pipeline.py
```

Full production output remains in external storage. Compact retained summaries
are in `results/marker_genes/`.
