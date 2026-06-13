# CellPhoneDB Consensus

`run_cellphone_dataset_consensus.py` runs the dataset-stratified DEG method for
each ecotype cluster, then builds cross-dataset consensus and same-stage
cluster-specificity rankings.

```bash
python nb_cellphoneDB/run_cellphone_dataset_consensus.py --dry-run
python nb_cellphoneDB/run_cellphone_dataset_consensus.py all
pytest -q nb_cellphoneDB/test_cellphone_dataset_consensus.py
```

Full production output remains in external storage. Compact retained summaries
are in `results/cellphonedb/`. `cellphone.ipynb` is exploratory.
