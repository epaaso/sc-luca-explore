import anndata as ad
import numpy as np
import pandas as pd
from nb_cellphoneDB import run_cellphone_dataset_consensus as consensus

from nb_cellphoneDB.run_cellphone_dataset_consensus import (
    balance_cells,
    complete_route_universe,
    load_degs,
    score_cluster_specificity,
)


def test_balance_cells_caps_patients_and_filters_cell_types():
    obs = pd.DataFrame(
        {
            "patient": ["p1"] * 5 + ["p2"] * 4 + ["p1"],
            "type_tissue": ["A"] * 5 + ["A"] * 4 + ["B"],
        },
        index=[f"c{i}" for i in range(10)],
    )
    data = ad.AnnData(np.ones((10, 2)), obs=obs)

    balanced, support = balance_cells(
        data, "patient", cap=2, seed=42, min_cells=3, min_patients=2
    )

    assert balanced.n_obs == 4
    assert set(balanced.obs["type_tissue"]) == {"A"}
    assert support.set_index("cell_type").loc["A", "included"]


def test_load_degs_combines_normal_and_tumor_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(consensus, "MARKER_DIR", tmp_path)
    for contrast, cell_type, genes, scores in [
        ("normal-vs-normal", "Macrophage", ["N1", "N2"], [0.95, 0.2]),
        ("tumor-vs-all", "Tumor LUAD", ["T1", "T2"], [0.91, 0.1]),
    ]:
        directory = tmp_path / contrast
        directory.mkdir()
        region = {
            "names": np.core.records.fromarrays([genes], dtype=[(cell_type, "O")]),
            "scores": np.core.records.fromarrays([scores], dtype=[(cell_type, float)]),
        }
        np.save(directory / "I-II_UKIM-V_cluster_0_auc.npy", region)

    degs = load_degs("I-II", 0, "UKIM-V", auc_threshold=0.9)

    assert not degs.empty
    assert degs.columns.tolist() == ["cluster", "gene"]
    assert degs.duplicated().sum() == 0
    assert set(degs["gene"]) == {"N1", "T1"}


def test_specificity_penalizes_testable_absence_and_ignores_untestable_cluster():
    target = pd.DataFrame(
        {
            "interacting_pair": ["A_B"],
            "activity": [0.8],
            "eligible_datasets": [4],
            "cluster": [0],
        }
    )
    testable_absence = pd.DataFrame(
        {
            "interacting_pair": ["other_pair"],
            "activity": [0.5],
            "eligible_datasets": [3],
            "cluster": [1],
        }
    )
    observed_other = pd.DataFrame(
        {
            "interacting_pair": ["A_B"],
            "activity": [0.3],
            "eligible_datasets": [2],
            "cluster": [2],
        }
    )

    scored = score_cluster_specificity(
        target, [testable_absence, observed_other], ["interacting_pair"]
    )

    assert scored.loc[0, "testable_other_clusters"] == 1
    assert scored.loc[0, "max_other_activity"] == 0.3
    assert scored.loc[0, "specificity"] == 0.5


def test_complete_route_universe_adds_testable_absence_as_zero():
    universe = pd.DataFrame(
        {
            "interacting_pair": ["A_B", "C_D"],
            "sender_cell_type": ["T", "missing"],
            "receiver_cell_type": ["M", "M"],
        }
    )

    completed = complete_route_universe(
        pd.DataFrame(),
        universe,
        {"dataset": {"T", "M"}},
        "I-II",
        1,
    )

    assert completed["interacting_pair"].tolist() == ["A_B"]
    assert completed.loc[0, "eligible_datasets"] == 1
    assert completed.loc[0, "activity"] == 0
