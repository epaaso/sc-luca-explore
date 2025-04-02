# test_pipeline.py
import pytest
import pandas as pd
import numpy as np
import anndata as ad
import os

from unittest.mock import patch, MagicMock

# Import your classes and configs from your main module:
from nb_DE_wilcox.modal_DE import (
    DEConfig, DataLoader, DEProcessor, DEVisualizer
)

# --------------------------------------------------------------------------
# Test Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def config():
    """
    Returns a DEConfig object pre-populated with the parameter overrides
    you specified:

        "ext_name": 'Trinks_Bishoff_2021_NSCLC',
        "name": 'Bishoff',
        "pred_name": 'Subcluster_wu/Bishoff',
        "time": "III-IV",
        "skip_stages": True,
        "cell_key": "cell_type_adjusted",
        "stage_key": "Pathological stage"
    """
    cfg = DEConfig()
    cfg.common.ext_name = "Trinks_Bishoff_2021_NSCLC"
    cfg.common.name = "Bishoff"
    cfg.dataloader.pred_name = "Subcluster_wu/Bishoff"
    cfg.common.time = "III-IV"
    cfg.dataloader.skip_stages = True
    cfg.dataloader.cell_key = "cell_type_adjusted"
    cfg.dataloader.stage_key = "Pathological stage"
    # call __post_init__ in case you rely on it to set defaults
    cfg.__post_init__()
    return cfg


@pytest.fixture
def fake_preds(config):
    """
    Returns a dummy preds DataFrame that mimics the structure of your
    real prediction CSV. By default, let's say it includes a 'batch'
    column if "Atlas" might appear, and a 'cell_type_adjusted' column.
    """
    data = {
        "batch": ["Trinks_Bishoff_2021_NSCLC_1", "Trinks_Bishoff_2021_NSCLC_2", "Trinks_Bishoff_2021_NSCLC_2"],
        config.dataloader.cell_key: ["Tumor", "Tumor", "Tumor"],  # or other cell types
    }
    index = ["cell_1", "cell_2", "cell_3"]
    df = pd.DataFrame(data, index=index)
    return df


@pytest.fixture
def fake_adata():
    """
    Returns a minimal AnnData object with 2 cells, matching the
    fake_preds above.
    """
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [1.0,4.0]])  # 3 cells, 2 genes
    var = pd.DataFrame(index=["GeneA", "GeneB"])
    obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
    adata = ad.AnnData(X=X, var=var, obs=obs)
    return adata


@pytest.fixture
def fake_de_regions():
    """
    Returns a dictionary with scores, names, and params for 2 regions.
    """
    de_region = {
        "scores": np.core.records.fromarrays(
            [[1.0, 2.0],[2, 4]],
            dtype=[("Tumor", float),("Tumor2", float)]
        ),
        "names": np.core.records.fromarrays(
            [["GeneA", "GeneB"], ["GeneC", "GeneD"]],
            dtype=[("Tumor", "O"), ("Tumor2", "O")]
        ),
        "params": {
            "groupby": "type_tissue",
            "reference": "tumorall",
            "method": "wilcoxon",
            "use_raw": False,
            "layer": None,
            "corr_method": "benjamini-hochberg"
        },
    }
    return de_region


@pytest.fixture
def fake_de_summary():
    """
    Returns a dictionary with cell types and all the scores for every gene
    """
    de_summary = {
        'Tumor1': {
            "GeneA": np.array(
                [1.0, 122.0, -4.0]
            ),
            "GeneB": np.array(
                [1.0, 2.0, -4.0]
            ),
        
        },
        'Tumor2': {
            "GeneA": np.array(
                [10., 22.0, -44.0]
            ),
            "GeneB": np.array(
                [5.0, -12.0, -44.0]
            ),
        
        }
    }
    return de_summary


@pytest.fixture
def fake_de_pair():
    """
    Returns a dictionary with scores, names, logfoldchanges, and pvals for 1 region.
    """
    de_pair = {
        'Tumor1_vs_Tumor2': {
        "scores": np.array(
            [1.0, 2.0, 4.0]
        ),
        "names": np.array(
            ["GeneA", "GeneB", "GeneD"]
        ),
        'pvals': np.array(
            [1e-3, 0.02, 0.04]
        ),
        'logfoldchanges': np.array(
            [1.0, 2.0, 4.0]
        ),
        'pvals_adjusted': np.array(
            [2e-3, 0.01, 0.06]
        ),
        }
    }
    return de_pair

# --------------------------------------------------------------------------
# DataLoader Tests
# --------------------------------------------------------------------------

def test_dataloader_load_predictions(config, fake_preds):
    """
    Test that load_predictions() reads the correct CSV path and returns
    the predicted DataFrame. We'll mock `pd.read_csv` to avoid file I/O.
    """
    loader = DataLoader(config.common, config.dataloader)

    mock_csv = patch("pandas.read_csv", return_value=fake_preds)
    with mock_csv as mocked_read:
        preds = loader.load_predictions()
        # Check that the correct CSV path was constructed:
        expected_suffix = "early" if "I-II" in config.common.time else "late"
        expected_path = os.path.join(config.common.backup_dir,
                                     f"{config.dataloader.pred_name}_predicted_leiden_{expected_suffix}.csv")
        mocked_read.assert_called_once_with(expected_path, index_col=0)
    
    # Since pred_name includes "Bishoff", no "Atlas" filtering is expected.
    assert len(preds) == 3, "Fake preds should have 3 rows"


def test_dataloader_load_anndata(config, fake_preds, fake_adata):
    """
    Test that load_anndata() attempts to read the correct h5ad path,
    merges predictions, and applies transformations.
    We'll mock anndata.read_h5ad to return our fake_adata.
    """
    loader = DataLoader(config.common, config.dataloader)
    with patch("anndata.read_h5ad", return_value=fake_adata) as mock_read:
        # We won't test file-based stage filtering in detail here,
        # but let's check if stage filtering is skipped since skip_stages=True.
        stages = loader.determine_stages()  # might be None
        adata = loader.load_anndata(fake_preds, stages)
        # Check the AnnData reading path
        expected_h5ad = os.path.join(config.common.backup_dir, 
                                     f"filtered_{config.common.ext_name}.h5ad")
        mock_read.assert_called_once_with(expected_h5ad)

        # Make sure the AnnData is updated with the new cell_key
        assert "type_tissue" in adata.obs.columns
        assert adata.obs["type_tissue"].tolist() == ["Tumor", "Tumor", "Tumor"]

        # TODO If log_layer is default "do_log1p", check whether adata.X was log-transformed.
        # For a minimal test, let's just ensure code doesn't crash.
        # A real test might check if the values in X reflect log1p transformation.


def test_dataloader_filter_preds_noadata(config, fake_preds):
    """
    If no_adata is True, check that filter_preds_noadata() is invoked
    and modifies 'fake_preds' accordingly. We'll mock h5py File read.
    """
    # Mark no_adata as True to skip loading AnnData.
    config.dataloader.no_adata = True
    loader = DataLoader(config.common, config.dataloader)

    # Suppose the stage is "Pathological stage". We'll set skip_stages=False here
    # to test the filtering logic. We'll also mock the obs matrix read.
    config.dataloader.skip_stages = False
    stages = loader.determine_stages()  # This might yield some stage list

    mock_h5py = patch("h5py.File", MagicMock())
    with mock_h5py:
        # We'll patch read_elem to return a DataFrame-like object with the stage column
        with patch("anndata.experimental.read_elem", return_value=pd.DataFrame({
            "Pathological stage": ["IIIA", "IV", "III"]  # matching our 2 cells
        }, index=["cell_1", "cell_2", "cell_3"])):
            new_preds = loader.filter_preds_noadata(fake_preds, stages)
            # If skip_stages=False and time="III-IV", stages might be 
            # ['IIIA', 'IIIB','III', 'III or IV', 'IV'] => so both cells pass
            assert len(new_preds) == 3, "Expect both cells to pass stage filter for III-IV"


# --------------------------------------------------------------------------
# DEProcessor Tests
# --------------------------------------------------------------------------

def test_processor_determine_tumor_types(config, fake_preds):
    """
    Test that we can identify tumor types from the preds DataFrame.
    """
    processor = DEProcessor(config.common, config.processor, config.dataloader)
    valid_types, tumor_types = processor.determine_tumor_types(fake_preds)
    # Our fake_preds had "Tumor" in cell_type_adjusted, so that's a valid & tumor type.
    assert "Tumor" in valid_types
    assert tumor_types == ["Tumor"]


def test_processor_compute_pairwise(config, fake_adata):
    """
    Test that compute_pairwise either loads from an existing file or
    calls rank_genes_groups_pairwise.
    We'll mock np.load and check if it tries to read the correct file.
    """
    config.processor.load_pair = True
    processor = DEProcessor(config.common, config.processor, config.dataloader)

    with patch("os.path.exists", return_value=True):
        mock_np_load = patch("numpy.load", return_value=type("ObjWithItem", (object,), {
            "item": lambda self: {"mock_key": "mock_value"}
        })()
        )
        with mock_np_load as load_patch:
            de_pair = processor.compute_pairwise(fake_adata, ["Tumor"], ["Tumor"])
            # If file exists and load_pair is True, it loads from the file (if load_summary is false).
            # By default, config.processor.load_summary is False, so it tries to load pairwise data.
            load_patch.assert_called_once()
            assert de_pair == {"mock_key": "mock_value"}


def test_processor_compute_summary(config, fake_adata, fake_de_pair):
    """
    Test that compute_summary either loads a summary or regenerates one if needed.
    We'll mock np.load to simulate a stored summary.
    """
    processor = DEProcessor(config.common, config.processor, config.dataloader)

    # We'll mock an existing summary file
    with patch("os.path.exists", return_value=True):
        de_summary = processor.compute_summary(fake_adata, fake_de_pair, ["Tumor"], ["Tumor"])
        assert "Tumor" in de_summary
        assert "GeneA" in de_summary["Tumor"]
        

    config.processor.load_summary = True
    with patch("os.path.exists", return_value=True):
        mock_np_load = patch("numpy.load", return_value=type("ObjWithItem", (object,), {
            "item": lambda self: {"Tumor": {
                "GeneA": [0.5, 0.1]
                }
            }
            })())
        with mock_np_load as load_patch:
            de_summary = processor.compute_summary(fake_adata, fake_de_pair, ["Tumor"], ["Tumor"])
            load_patch.assert_called_once()
            assert "Tumor" in de_summary
            assert "GeneA" in de_summary["Tumor"]


def test_processor_compute_regions(config, fake_adata, fake_de_summary):
    """
    Tests that compute_regions either loads from file or creates a new region-level
    dictionary. We'll mock np.load for the existing file scenario.
    """
    processor = DEProcessor(config.common, config.processor, config.dataloader)

    # Suppose a file is not found => it must create one
    with patch("os.path.exists", return_value=False):
        mock_np_save = patch("numpy.save", MagicMock())
        with mock_np_save as save_patch:
            de_region = processor.compute_regions(fake_adata, fake_de_summary)
            save_patch.assert_called_once()
            assert "Tumor" in de_region["scores"].dtype.names


# --------------------------------------------------------------------------
# DEVisualizer Tests
# --------------------------------------------------------------------------

def test_visualizer_plot_marker_genes(config, fake_de_regions):
    """
    We'll ensure that plot_marker_genes doesn't crash and attempts to save
    a figure to the correct path. We'll mock plt.savefig.
    """
    vis = DEVisualizer(config.common, config.visualizer)

    # Minimal de_region structure
    de_region = fake_de_regions
    valid_types = ["Tumor", "Tumor2"]
    with patch("matplotlib.pyplot.savefig", MagicMock()) as mock_save:
        vis.plot_marker_genes(de_region, valid_types)
        mock_save.assert_called_once()
        # You could also check the file path used in config.common.w_folder, etc.


def test_visualizer_plot_gsea(config, fake_de_regions):
    """
    Test that plot_gsea attempts to load or compute GSEA, then calls savefig.
    We'll mock file existence checks and skip the actual GSEA logic.
    """
    vis = DEVisualizer(config.common, config.visualizer)
    vis.config.load_gsea_heatmap = False
    de_region = fake_de_regions
    valid_types = ["Tumor", "Tumor2"]

    # Suppose the CSV doesn't exist => load_gsea_heatmap => calls get_gseas_df
    with patch("os.path.exists", side_effect=[True, True, False]):  # first for gmt_file, then for folder existence
        with patch("matplotlib.pyplot.savefig", MagicMock()) as mock_save:
            mock_np_load = patch(
                "numpy.load",
                return_value=type("ObjWithItem", (object,), {
                    "item": lambda self: {
                        "Term": pd.Series(["prefixTermA", "prefixTermB"]),
                        "NES": pd.Series([2.0, 3.0])
                    }
                })()
            )
            with mock_np_load as load_patch:
                vis.plot_gsea(de_region, valid_types)
                assert load_patch.call_count == 2
                mock_save.assert_called_once()
