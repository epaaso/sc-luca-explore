import os
from typing import Any, Iterable, List, Literal, Optional, Union, Dict, Tuple
from dataclasses import dataclass, field
import logging
import multiprocessing
import json

import numpy as np
import numba
import anndata as ad
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scanpy.tools import rank_genes_groups
import scanpy as sc
import gseapy

# TODO implement for different region names and masks

local = True

if not local:
    import modal
    app = modal.App("DE - wilcox")

local_dir = '/root/datos/maestria/netopaas/luca_explore/surgeries/'
# Define the remote path where the data will be available in the remote function
backup_dir = "/data" if not local else local_dir
w_folder = '/root/host_home/luca/nb_DE_wilcox/wilcoxon_DE' if local else backup_dir

if not local:
    scvi_image = modal.Image.from_registry(
        "ghcr.io/scverse/scvi-tools:py3.11-cu12-1.2.x-runtime")\
        .pip_install('gprofiler-official==1.0.0', 'gseapy==1.1.1', 'GEOparse==2.0.4')\
    .pip_install('scanpy','matplotlib', 'seaborn').pip_install('numba')

    # To delete mutiple files: modal volume ls --json DE-vol | jq -r '.[] | select(.Filename | test("^Tumor")) | .Filename' | xargs -I {} sh -c 'echo Deleting: {}; modal volume rm DE-vol "/{}"'
    vol = modal.Volume.from_name("DE-vol", create_if_missing=True)

def get_gseas_df(de_regions: dict, valid_types: List[str],
                 types: List[str], id_: str, load_gsea: bool = False,
                 gsea_folder:str = 'gseapy_gsea', gene_mapping: dict = None,
                 num_threads:int = 1) -> pd.DataFrame:
    """
    Generates a combined DataFrame from GSEA results for different cell types.

    Args:
    adata (ad.AnnData): An AnnData object containing the dataset.
    valid_types (List[str]): A list of valid cell types to filter the cell types.
    types (List[str]): A list of cell types to be considered for analysis.
    id_ (str): Identifier used in generating the GSEA path.
    load_gsea (bool, optional): Flag to load existing GSEA results. Defaults to False.
    gene_mapping (dict): A dictionary mapping gene names to other ones.

    Returns:
    pd.DataFrame: A combined DataFrame with GSEA results pivoted by cell type and hallmark.

    The function uses the provided 'types', intersects them with 'valid_types',
    and computes GSEA for each type. The final DataFrame is a pivot table with cell types as rows
    and GSEA hallmarks as columns.
    """
    # Intersect types with valid types
    types = set(types).intersection(set(valid_types))
    dfs = []

    for typer in types:
        ranks = de_regions
        type_path = typer.replace('/','_')
        gsea_path = f'{gsea_folder}/{type_path}_{id_}'

        # Generate gene scores
        if gene_mapping:
            gene_scores = {gene_mapping.get(gene, gene): ranks['scores'][typer][i] for i, gene
                            in enumerate(ranks['names'][typer])}
        else:
            gene_scores = {gene: ranks['scores'][typer][i] for i, gene
                            in enumerate(ranks['names'][typer])}
        gene_scores = pd.Series(gene_scores)

        # Run GSEA
        if load_gsea:
            gseas = np.load(gsea_path, allow_pickle='TRUE').item()
        else:
            pre_res = gseapy.prerank(rnk=gene_scores,
                                     gene_sets='h.all.v2023.2.Hs.symbols.gmt',
                                     threads=num_threads, permutation_num=100,
                                     seed=6, no_plot=True)
            # TODO  expose seed and permut num
            gseas = pre_res.res2d
            # We remove this because it clutters the folders and  though it has more info it is not used
            # np.save(gsea_path, gseas)

        data1 = {'hallmark': list(gseas['Term'].str.slice(9)), 'score': list(gseas['NES'])}
        df_celltype1 = pd.DataFrame(data1)
        df_celltype1['cell_type'] = typer
        dfs.append(df_celltype1)

    # Combine all DataFrames
    combined_df = pd.concat(dfs).reset_index(drop=True).pivot(index='cell_type',
                                                               columns='hallmark', values='score')
    return combined_df


def process_gene(group1, results, groups2):
    scores = {}
    genes = next(iter(results.items()))[1]['names']
    comparisons = [(group1, group2) for group2 in groups2 if group1 != group2]
    
    for gene in genes:
        comparison_scores = []
        for comparison in comparisons:
            comp_key = f'{comparison[0]}_vs_{comparison[1]}'
            gene_index = np.where(results[comp_key]['names'] == gene)[0]
            comparison_scores.append(results[comp_key]['scores'][gene_index][0])
        scores[gene] = comparison_scores
    return group1, scores


def compare_groups(adata: ad.AnnData, groupby: str, group1: str, group2: str,
                   method:str='wilcoxon', use_raw:bool=False, parallel:bool=False
                   ,n_jobs_inner:int=10):

    key = f'{group1}_vs_{group2}'
    if parallel:
        print(f'Started copying {key}')
    else:
        print(f'Comparing {key}')
    adata_temp = adata.copy() if parallel else adata # Make a copy to avoid modifying the shared adata THIS IS DUMB TODO FIX, maybe using `sc.aggregate`
    if parallel:
        print(f'Ended copying {key}')
    
    numba.set_num_threads(n_jobs_inner)
    rank_genes_groups(adata_temp, groupby=groupby, groups=[group1], reference=group2,
                    method=method, use_raw=use_raw, key_added=key, n_jobs=n_jobs_inner)

    current_scores = adata_temp.uns[key]['scores'][group1]

    return key, {
        'scores': current_scores,
        'names': adata_temp.uns[key]['names'][group1],
        'pvals': adata_temp.uns[key]['pvals'][group1],
        'logfoldchanges': adata_temp.uns[key]['logfoldchanges'][group1],
        'pvals_adj': adata_temp.uns[key]['pvals_adj'][group1]
    }


def rank_genes_groups_pairwise(adata: ad.AnnData, groupby: str, 
                               groups: Union[Literal['all'], Iterable[str]] = 'all', 
                               subgroups: Optional[ Iterable[str]] = None,
                               use_raw: Optional[bool] = None,
                               method: Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']] = 'wilcoxon',
                               parallel: bool = False,
                               n_jobs: int = 1,
                               n_jobs_inner: int = 10) -> dict:
    """
    Perform pairwise comparison of marker genes between specified groups. Expects log data.

    Parameters:
    - adata: AnnData object containing the data.
    - groupby: The key of the column in .obs where the annotations of the groups are
    - groups: List of groups to include in pairwise comparisons.
    - subgroups: List of groups to restric te first element in comparison to.
    - method: The statistical method to use for the test ('t-test', 'wilcoxon', etc.).
    - n_jobs: Number of jobs to run in parallel. -1 means using all processors.

    Returns:
    - Returns a dict for all pairwise groups and its statistics
    """

    pairwise_results = {}
    summary_stats = {}
    results = []
    if subgroups:
        comparisons = [(group1, group2) for group1 in subgroups for group2 in groups if group1 != group2]
    else:
        comparisons = [(group1, group2) for group1 in groups for group2 in groups if group1 != group2]

    if parallel:
        with multiprocessing.Pool(n_jobs) as pool:
            results = pool.starmap(compare_groups, [(adata, groupby, group1, group2, method, use_raw, parallel, n_jobs_inner) for group1, group2 in comparisons])
    else:
        for comparison in comparisons:
            group1, group2 = comparison
            results.append(compare_groups(adata, groupby, group1, group2, method, use_raw, parallel, n_jobs_inner))

    for key, result in results:
        pairwise_results[key] = result

    return pairwise_results


def rank_genes_group(
    de_regions: dict,
    group_name: str,
    n_genes: int = 20,
    gene_mapping: Optional[dict] = None,
    gene_names: Optional[List[str]] = None,
    fontsize: int = 8,
    titlesize: int = 10,
    show: Optional[bool] = None,
    ax: Optional[Axes] = None,
    **kwds,
) -> Axes:
    """
    Visualizes the ranking of genes for a specified group from an AnnData object.

    Parameters:
        de_regions: dict
            The dictionary containing the dataset and analysis results.
        group_name: str
            The name of the group for which to rank genes.
        n_genes: int, optional
            The number of top genes to display (default is 20).
        gene_mapping: Optional[dict], optional
            A dictionary mapping gene names to other ones
        gene_names: Optional[List[str]], optional
            Explicit list of gene names to use for plotting.
        fontsize: int, optional
            Font size for gene names (default is 8).
        titlesize: int, optional
            Font size for the title (default is 10).
        show: Optional[bool], optional
            If True, show the plot immediately.
        ax: Optional[Axes], optional
            A matplotlib axes object to plot on. If None, a new figure is created.
        **kwds:
            Additional keyword arguments to pass to plotting functions.

    Returns:
        Axes:
            The matplotlib axes with the plot.

    Raises:
        ValueError:
            If n_genes is less than 1.
    """
    if n_genes < 1:
        raise ValueError(f"n_genes must be positive; received n_genes={n_genes}.")

    reference = str(de_regions['params']['reference'])

    gene_names = gene_names if gene_names is not None else de_regions['names'][group_name][:n_genes]
    gene_mask = np.isin(de_regions['names'][group_name], gene_names)
    scores = de_regions['scores'][group_name][gene_mask]

    if gene_mapping:
        gene_names = [gene_mapping.get(gene, gene) for gene in gene_names]

    ymin = np.min(scores)
    ymax = np.max(scores)
    ymax += 0.3 * (ymax - ymin)

    ax = ax if ax is not None else plt.subplot(111)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.9, n_genes - 0.1)

    for ig, gene_name in enumerate(gene_names):
        ax.text(
            ig, scores[ig], gene_name,
            rotation='vertical',
            verticalalignment='bottom',
            horizontalalignment='center',
            fontsize=fontsize
        )

    ax.set_title(f'{group_name} vs. {reference}', fontsize=titlesize)
    ax.set_xticklabels([])
    ax.set_ylabel('score')
    ax.set_xlabel('genes')

    if show:
        plt.show()

    return ax


def cond_plot(de_regions: dict, cond_types, valid_types, n_genes,
               ax: Optional[Axes] = None, fontsize=9, titlesize=14,
                gene_mapping:dict=None, **kwds):
    if set(cond_types).issubset(valid_types):
        rank_genes_group(de_regions, cond_types[0], n_genes=n_genes,
                            ax=ax, sharey=False, show=False,
                            fontsize=fontsize, titlesize=titlesize, gene_mapping=gene_mapping)
    else:
        # pass
        # Draw an empty plot with a message
        if ax:
            ax.text(0.5, 0.5, f'Missing cells: {cond_types}', color='red',
                        ha='center', va='center', transform=ax.transAxes) 
            ax.axis('off')



# -----------------------------------------------------------------------------
# region Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class CommonConfig:
    """
    Configuration for common vars used in all the pipeline.

    Attributes
    ----------
    ext_name : str, optional
        External name or ID for the dataset (default "Zuani_2024_NSCLC").
    name : str, optional
        Study or dataset name (default "Zuani").
    time : str, optional
        Stage designation (e.g., "I-II" or "III-IV") to control filtering
        (default "I-II").
    gene_mapping : Union[str, dict], optional
        obs column name or dictionary for gene name mapping (default None).
    
    """
    ext_name: str = "Zuani_2024_NSCLC"
    name: str = "Zuani"
    time: str = "I-II"
    backup_dir = "/data" if not local else local_dir
    w_folder = '/root/host_home/luca/nb_DE_wilcox/wilcoxon_DE' if local else backup_dir
    gene_mapping: Union[str, Dict, None] = None

@dataclass
class DataLoaderConfig:
    """
    Configuration for data loading.

    Attributes
    ----------
    pred_name : str, optional
        Suffix of the prediction file to use for filtering (default None).
    obs_has_name : bool, optional
        If True, the obs index has the gene name as prefix... remove it from preds df for index harmonization (default False).
    obs_unique : bool, optional
        If True, makes the obs index unique (default False).
    cell_key : str, optional
        Column name in external annotation (predictions) with cell identities
        (default "cell_type_adjusted").
    no_adata : bool, optional
        If True, does not load or integrate with an AnnData object (default False).
    stage_key : str, optional
        Column name in adata.obs with stage annotations (default "stage").
    skip_stages : bool, optional
        If True, skip filtering by specific stages (default False).
    log_layer : str or bool, optional
        Layer to log-transform, or set to 'do_log1p' to apply sc.pp.log1p on
        the main matrix (default "do_log1p").
    gene_feature : str, optional
            Feature name to assign to adata.var.index (default None).
    avoid_ensembl : bool, optional
        If True, avoids using Ensembl by giving an error and possible column names to use (default False).
    """
    pred_name: Optional[str] = None  # If None, will default to CommonConfig.name.
    obs_has_name: bool = False
    obs_unique: bool = False
    cell_key: str = "cell_type_adjusted"
    no_adata: bool = False
    stage_key: str = "stage"
    skip_stages: bool = False
    log_layer: Union[str, bool] = "do_log1p"
    gene_feature: Optional[str] = None
    avoid_ensembl: bool = False

@dataclass
class ProcessorConfig:
    """
    Configuration for the data processing vars.

    Attributes
    ----------
    tumor_is_int : bool, optional
        If True, identifies tumor subpopulations as integer labels (default False).
    load_pair : bool, optional
        Whether to load existing pairwise comparisons from file (default True).
    load_summary : bool, optional
        Whether to load existing DE summary from file (default True).
    load_regions : bool, optional
        Whether to load existing region-level results from file (default True).
    regions_AUC : bool, optional
        If True, computes region-level AUCs instead of mean z-scores (default False).
    parallel_pair : bool, optional
        Enables pairwise Wilcoxon parallel execution if True/ NOT RECOMMENDED, MUST COPY THE WHOLE ANNDATA (default False).
    parallel_summary : bool, optional
        Enables parallel execution for summary statistics (default False).
    update_symbols_summary : bool, str optional
            If True, updates gene identifiers using a default path symbol map CSV. If str imports that CSV from the path str (default False).
    num_processes : int, optional
        Number of parallel processes (default 1). For running mutiple pairwise comparisons at the same time. NOT RECOMMENDED, MUST COPY THE WHOLE ANNDATA.
        But it is useful for the summary statistics. When parallel_summary is True it will be used but is not useful if it > number of cell types.
        Also used in the GSEA computation.
    n_jobs_inner : int, optional
        Internal parallelization for pairwise comparisons. Works thanks to numba.set_threads() (default 10).
    """
    tumor_is_int: bool = False
    load_pair: bool = False
    load_summary: bool = False
    load_regions: bool = False
    regions_AUC: bool = False
    parallel_pair: bool = False
    parallel_summary: bool = False
    update_symbols_summary: Union[bool, str] = False
    num_processes: int = 1
    n_jobs_inner: int = 1

@dataclass
class VisualizerConfig:
    """
    Configuration for data visualization tasks, including GSEA and region-level mapping.

    Attributes
    ----------
    region_mapping : str
        Path to a JSON file that maps older region names to new ones before plotting.
    load_gsea : bool
        Indicates whether to load existing GSEA data from disk.
    load_gsea_heatmap : bool
        If True, attempts to load saved GSEA heatmap data, skipping computation.
    """
    region_mapping: str = ""
    load_gsea: bool = True
    load_gsea_heatmap: bool = True

@dataclass
class DEConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)

    def __post_init__(self):
        # Ensure pred_name is set.
        if self.dataloader.pred_name is None:
            self.dataloader.pred_name = self.common.name

        if self.processor.update_symbols_summary==True:
            self.processor.update_symbols_summary = f'{self.common.backup_dir}/zuani_ensembl.csv'
# endregion

# -----------------------------------------------------------------------------
# region Data Loading Component
# -----------------------------------------------------------------------------

class DataLoader:
    def __init__(self, common_config: CommonConfig, dl_config: DataLoaderConfig):
        self.common = common_config
        self.config = dl_config

    def load_predictions(self) -> pd.DataFrame:
        time_suffix = "early" if "I-II" in self.common.time else "late"
        pred_file = os.path.join(
            self.common.backup_dir,
            f"{self.config.pred_name}_predicted_leiden_{time_suffix}.csv"
        )
        logging.info(f"Loading predictions from {pred_file}")
        preds = pd.read_csv(pred_file, index_col=0)
        if "Atlas" in self.config.pred_name:
            preds = preds[preds.batch.str.contains(self.common.ext_name)]
        if self.config.obs_has_name:
            preds.index = preds.index.str.split('_').str[1:].str.join('_')
        return preds

    def determine_stages(self) -> Optional[List[str]]:
        if 'I-II' in self.common.time:
            stages = ['IA1', 'IB', 'IA2', 'IA3', 'IIB', 'II', 'I']
        if 'III-IV' in self.common.time:
            stages = ['IIIA', 'IIIB','III', 'III or IV', 'IV']
        stages = None if self.config.skip_stages else stages

    def load_anndata(self, preds: pd.DataFrame, stages: Optional[List[str]]) -> ad.AnnData:
        if self.config.no_adata:
            return None
        adata_file = os.path.join(self.common.backup_dir, f"filtered_{self.common.ext_name}.h5ad")
        logging.info(f"Loading AnnData from {adata_file}")
        adata = ad.read_h5ad(adata_file)
        if self.config.obs_unique:
            adata.obs_names_make_unique()
        try:
            adata = adata[preds.index].copy()
            adata.obs.loc[preds.index, self.config.cell_key] = preds.loc[preds.index, self.config.cell_key]
        except Exception as e:
            raise Exception("Mismatch between predictions index and adata index") from e

        # Set a standardized key for cell types.
        adata.obs["type_tissue"] = adata.obs[self.config.cell_key]
        # Filter using stage information from adata.obs (CSV lacks stage info).
        if stages is not None:
            adata = adata[adata.obs[self.config.stage_key].isin(stages)].copy()

        # Apply log transformation.
        if self.config.log_layer == "do_log1p":
            sc.pp.log1p(adata)
        elif self.config.log_layer:
            adata.X = adata.layers[self.config.log_layer]
        
        preds = preds.loc[adata.obs.index]
            
        if self.config.gene_feature:
            print(f"Setting gene feature to {self.config.gene_feature}")
            adata.var.index = adata.var[self.config.gene_feature]
            print("First 10 genes in adata.var:")
            print(adata.var.index[:10])

        if isinstance(self.common.gene_mapping, str):
            self.common.gene_mapping = adata.var.loc[:, self.common.gene_mapping].to_dict()

        if self.config.avoid_ensembl and adata.var.index.str.contains('ENS').sum() > 400:
            raise Exception(f"The dataset contains Ensembl IDs. Please provide a gene feature to use from: {adata.var.columns} or a gene mapping")
        import gc
        gc.collect()

        logging.info("Data after log transformation:\n%s", adata[:10, 10:20].to_df().head())
        return adata
    
    def filter_preds_noadata(self, preds: pd.DataFrame, stages: Optional[List[str]]) -> pd.DataFrame:
        if stages is None and not isinstance(self.common.gene_mapping, str):
            return preds
        
        import h5py
        from anndata.experimental import read_elem
        adata_file = os.path.join(self.common.backup_dir, f"filtered_{self.common.ext_name}.h5ad")
        with h5py.File(adata_file, 'r') as f:
            # Check if 'obs' group exists (common for scRNA-seq data)
            if 'obs' in f:
                obs_matrix = read_elem(f['obs'])
            else:
                # Handle the case where 'obs' is not present
                raise("obs matrix not found in the h5ad file.")
            
        if isinstance(self.common.gene_mapping, str):
            self.common.gene_mapping = obs_matrix.loc[:, self.common.gene_mapping].to_dict()

        preds[self.config.stage_key] = obs_matrix[self.config.stage_key]
        preds = preds[preds[self.config.stage_key].isin(stages)].copy() if stages else preds

        del obs_matrix
        

# -----------------------------------------------------------------------------
# region Processing Component
# -----------------------------------------------------------------------------

class DEProcessor:
    def __init__(self, common_config: CommonConfig, proc_config: ProcessorConfig, dl_config: DataLoaderConfig):
        self.common = common_config
        self.config = proc_config
        self.dl_config = dl_config

    def determine_tumor_types(self, preds: pd.DataFrame) -> Tuple[List[str], List[str]]:
        valid_types = list(preds[self.dl_config.cell_key].value_counts().loc[lambda x: x > 2].index)
        if not self.config.tumor_is_int:
            tumor_types = [
                g for g in valid_types
                if any(x in g for x in ['Tumor', 'Ciliated', 'AT2', 'AT1', 'Club'])
            ]
        else:
            tumor_types = [g for g in valid_types if g.isdigit()]

        if len(tumor_types) == 0:
            raise Exception("No tumor groups found THE SUMMARIES WOULD BE EMPTY")
        return valid_types, tumor_types

    def compute_pairwise(self, adata: ad.AnnData, valid_types: List[str], tumor_types: List[str]) -> dict:
        logging.info("Computing pairwise differential expression")
        de_pair = {}
        if not self.config.load_summary:
            pair_file = os.path.join(
                self.common.w_folder,
                f"{self.common.time}_{self.common.ext_name}_tumorpair.npy"
            )
            if self.config.load_pair and os.path.exists(pair_file):
                de_pair = np.load(pair_file, allow_pickle=True).item()
            else:
                de_pair = rank_genes_groups_pairwise(
                    adata, "type_tissue", method="wilcoxon", use_raw=False,
                    groups=valid_types, subgroups=tumor_types,
                    parallel=self.config.parallel_pair,
                    n_jobs=self.config.num_processes // self.config.n_jobs_inner,
                    n_jobs_inner=self.config.n_jobs_inner
                )
                np.save(pair_file, de_pair)
        return de_pair

    def compute_summary(self, adata: Optional[ad.AnnData], de_pair: dict, valid_types: List[str], tumor_types: List[str]) -> dict:
        logging.info("Computing summary differential expression")
        summary_file = os.path.join(
            self.common.w_folder,
            f"{self.common.time}_{self.common.ext_name}_summary_tumorall.npy"
        )
        de_summary = {}
        if self.config.load_summary:
            if not os.path.exists(summary_file):
                logging.info(f"Summary file {summary_file} not found!"
                             )
                if self.config.load_pair:
                    raise Exception("Pairwise comparisons must be loaded to compute summary")
                logging.info("Computing summary from pairs")
            else:
                de_summary = np.load(summary_file, allow_pickle=True).item()

            if self.config.update_symbols_summary:
                gene_name_map = pd.read_csv(
                    self.config.update_symbols_summary)\
                    .iloc[:,:1]
                gene_name_map = gene_name_map.to_dict()['Unnamed: 0']
                gene_name_map = {str(k): v for k, v in gene_name_map.items()}
            
                logging.info('THE GENE MAP')
                logging.info(str(gene_name_map)[:100])
                for cell_type, genes_dict in de_summary.items():
                    updated_genes_dict = {}
                    for gene, scores in genes_dict.items():
                        new_gene_name = gene_name_map.get(gene, gene)
                        updated_genes_dict[new_gene_name] = scores
                    de_summary[cell_type] = updated_genes_dict

        elif not self.config.load_regions:
            if self.config.parallel_summary:
                with multiprocessing.Pool(self.config.num_processes) as pool:
                    for group_scores in pool.starmap(process_gene,
                            [(group1, de_pair, valid_types) for group1 in tumor_types]):
                        de_summary[group_scores[0]] = group_scores[1]
            else:
                for group in tumor_types:
                    grp, summary = process_gene(group, de_pair, valid_types)
                    de_summary[grp] = summary
            
            np.save(summary_file, de_summary)
            if not local:
                vol.commit()

        if adata:
            adata.uns[f'rank_genes_groups_summary_tumorall'] = de_summary

        return de_summary

    def compute_regions(self, adata: Optional[ad.AnnData], de_summary: dict, region_file: str=None) -> dict:
        logging.info("Computing region-level differential expression")
        if not region_file:
            region_file = os.path.join(
                self.common.w_folder,
                f"{self.common.time}_{self.common.ext_name}_tumorall.npy"
            )

        if self.config.load_regions and os.path.exists(region_file):
            de_region = np.load(region_file, allow_pickle=True).item()
        else:
            if not self.config.regions_AUC:
                regioner_sorted = {
                    ct: sorted(((gene, np.mean(vals)) for gene, vals in gdict.items()),
                            key=lambda x: x[1], reverse=True)
                    for ct, gdict in de_summary.items()
                }
            else:
                counts = {}
                if adata is None:
                    logging.warning("No adata provided, using a very dirty AUC approximation")
                for ct in de_summary.keys():
                    logging.info("Computing AUC for %s", ct)
                    if adata is not None:
                        n1 = np.sum(adata.obs["type_tissue"] == ct)
                        # TODO this is wrong! n2 should be of the size of the other cell type
                        n2 = np.sum(adata.obs["type_tissue"] != ct)
                        counts[ct] = (n1, n2)
                        # Replace np.mean(vals) with the mean of AUCs for each z-value in vals:
                        # For each z in vals, compute:
                        #   auc_z = ( z * sqrt(n1*n2*(n1+n2+1)/12) + (n1*n2)/2 ) / (n1*n2)
                        # Then take the mean of all auc_z for that gene.
                        regioner_sorted = {
                            ct: sorted(
                                (
                                    (
                                        gene,
                                        np.mean([
                                            (z * np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) + (n1 * n2) / 2) / (n1 * n2)
                                            for z in vals
                                        ])
                                    )
                                    for gene, vals in gdict.items()
                                ),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            for ct, gdict in de_summary.items()
                            for n1, n2 in [counts[ct]]
                        }
                    else:
                        # If we don't have n1 and n2, we do something very dirty and try to leave the scores between 0 and 1
                        # with 0.5 meaning no effect like with AUC. For that we do this transform:
                        # auc_z = (((z / max_val) * 0.9) + 1) / 2
                        regioner_sorted = {
                            ct: sorted(
                                (
                                    (
                                        gene,
                                        np.mean([(((z / abs_max) * 0.9) + 1) / 2 for z in vals])
                                    )
                                    for gene, vals in gdict.items()
                                    for abs_max in [max(abs(min(vals)), abs(max(vals)))]
                                ),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            for ct, gdict in de_summary.items()
                        }
                
        
            cell_types = list(regioner_sorted.keys())
            arr_scores, arr_names = [], []
            for ct in cell_types:
                g, scs = zip(*regioner_sorted[ct])
                arr_names.append(g)
                arr_scores.append(scs)
            arr_scores = np.array(arr_scores).T
            arr_names = np.array(arr_names).T
            dtypes_scores = [(ct, float) for ct in cell_types]
            dtypes_names = [(ct, "O") for ct in cell_types]

            de_region = {
                "params": {
                    "groupby": "type_tissue",
                    "reference": "tumorall",
                    "method": "wilcoxon",
                    "use_raw": False,
                    "layer": None,
                    "corr_method": "benjamini-hochberg"
                },
                "names": np.core.records.fromarrays(arr_names.T, dtype=dtypes_names),
                "scores": np.core.records.fromarrays(arr_scores.T, dtype=dtypes_scores)
            }

            np.save(region_file, de_region, allow_pickle=True)
            if not local:
                vol.commit()

        assert isinstance(de_region['names'], np.recarray) and isinstance(de_region['scores'], np.recarray)

        if adata:
            adata.uns[f'rank_genes_groups_tumorall'] = de_region
        return de_region

# -----------------------------------------------------------------------------
# Visualization Component
# -----------------------------------------------------------------------------

class DEVisualizer:
    def __init__(self, common_config: CommonConfig, vis_config: VisualizerConfig):
        self.common = common_config
        self.config = vis_config

    def plot_marker_genes(self, de_region: dict, valid_types: List[str]) -> None:
        logging.info("Plotting marker genes")

        if self.config.region_mapping:
            with open(self.config.region_mapping, "r") as f:
                mapper = json.load(f)

            valid_types = [mapper.get(t, t) for t in valid_types]
            types = list(de_region["scores"].dtype.names)
            new_names = tuple(mapper.get(name, name) for name in types)
            de_region["scores"].dtype.names = new_names
            de_region["names"].dtype.names = new_names

        types = de_region["scores"].dtype.names
        num_types = len(types)

        fig, axs = plt.subplots(
            (num_types + 1) // 2, 2, figsize=(16, 4.5 * ((num_types + 1) // 2))
        )
        axs = axs.ravel()
        n_genes = 20
        for i, cell_type in enumerate(types):
            cond_plot(
            de_region, [cell_type], valid_types, n_genes=n_genes,
            ax=axs[i], sharey=False, key="rank_genes_groups_tumorall",
            show=False, fontsize=6, titlesize=9, gene_mapping=self.common.gene_mapping
            )
        # Remove any extra axes
        if len(axs) > num_types:
            for j in range(num_types, len(axs)):
                fig.delaxes(axs[j])
        
        out_file = os.path.join(
            self.common.w_folder,
            f"markergenes_{self.common.name}_tumorall_{self.common.time}.png"
        )
        plt.savefig(out_file, bbox_inches="tight")
        if not local:
            vol.commit()
        logging.info(f"Marker genes plot saved to {out_file}")

    def plot_gsea(self, de_region: dict, valid_types: List[str]) -> None:
        logging.info("Plotting GSEA")
        gmt_file = "h.all.v2023.2.Hs.symbols.gmt"
        if not os.path.exists(gmt_file):
            import subprocess
            subprocess.run([
                "wget",
                "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt"
            ])

        gsea_folder = os.path.join(self.common.w_folder,'..', "gseapy_gsea")
        os.makedirs(gsea_folder, exist_ok=True)

        region = "tumorall"
        gsea_path = os.path.join(
            gsea_folder, f"heatmap_{self.common.name}_{region}_{self.common.time}.csv"
        )
        if self.config.load_gsea_heatmap and os.path.exists(gsea_path):
            combined_dfs = {region: pd.read_csv(gsea_path, index_col=0)}
        else:
            types = de_region["scores"].dtype.names
            combined_dfs = {region: get_gseas_df(
                de_region, valid_types, types,
                self.common.ext_name, load_gsea=self.config.load_gsea,
                gsea_folder=gsea_folder, gene_mapping=self.common.gene_mapping
            )}
            combined_dfs[region].to_csv(gsea_path)

        
        plt.figure(figsize=(15, 10))
        sns.heatmap(combined_dfs[region], cmap="viridis")
        plt.title(f"Hallmarks Scores by Cell Type for {region}")
        plt.xlabel("Hallmarks")
        plt.ylabel("Cell Types")
        out_gsea = os.path.join(
            gsea_folder, f"heatmap_{self.common.name}_{region}_{self.common.time}.png"
        )
        plt.savefig(out_gsea, bbox_inches="tight")
        if not local:
            vol.commit()
        logging.info(f"GSEA plot saved to {out_gsea}")

# -----------------------------------------------------------------------------
# Pipeline Orchestrator
# -----------------------------------------------------------------------------

class DEPipeline:
    def __init__(self, config: DEConfig):
        self.config = config
        logging.basicConfig(level=logging.INFO)
        self.loader = DataLoader(config.common, config.dataloader)
        self.processor = DEProcessor(config.common, config.processor, config.dataloader)
        self.visualizer = DEVisualizer(config.common, config.visualizer)

    def run(self) -> None:
        logging.info("Starting differential expression pipeline")
        preds = self.loader.load_predictions()
        stages = self.loader.determine_stages()
        adata = self.loader.load_anndata(preds, stages)
        if not adata:
            preds = self.loader.filter_preds_noadata(preds, stages)
        valid_types, tumor_types = self.processor.determine_tumor_types(preds)

        de_pair = self.processor.compute_pairwise(adata, valid_types, tumor_types)
        de_summary = self.processor.compute_summary(adata, de_pair, valid_types, tumor_types)
        de_region = self.processor.compute_regions(adata, de_summary)

        self.visualizer.plot_marker_genes(de_region, valid_types)
        self.visualizer.plot_gsea(de_region, valid_types)
        logging.info("Pipeline finished successfully")

# -----------------------------------------------------------------------------
# Modal Entry Point (Top-level Function)
# -----------------------------------------------------------------------------

def _get_de_impl(**kwargs):
    """
    Runs the differential expression pipeline.

    This function initializes the pipeline using a composite configuration that
    aggregates parameters from several configuration classes:
    
    - CommonConfig: General settings (e.g., dataset names, paths, runtime flags).
    - DataLoaderConfig: Parameters related to loading predictions and AnnData.
    - ProcessorConfig: Settings for data processing and differential expression
      computation.
    - VisualizerConfig: Parameters for generating plots and visualizations.
    
    Parameters
    ----------
    kwargs : dict
        Optional overrides for any configuration parameter. Refer to the
        respective configuration classes for details:
            - CommonConfig
            - DataLoaderConfig
            - ProcessorConfig
            - VisualizerConfig

    Examples
    --------
    >>> get_de(ext_name="Dataset_X", time="I-II", cell_key="cell_type")
    
    See Also
    --------
    DataLoaderConfig, ProcessorConfig, VisualizerConfig
    """

    # Instantiate the overall configuration.
    config = DEConfig()

    # Optionally update configuration fields if overrides are provided.
    for key, value in kwargs.items():
        if hasattr(config.common, key):
            setattr(config.common, key, value)
        elif hasattr(config.dataloader, key):
            setattr(config.dataloader, key, value)
        elif hasattr(config.processor, key):
            setattr(config.processor, key, value)
        elif hasattr(config.visualizer, key):
            setattr(config.visualizer, key, value)

    config.__post_init__()  # Ensure defaults are applied.
    pipeline = DEPipeline(config)
    pipeline.run()

if local:
    # Local definition without the Modal decorator.
    get_de = _get_de_impl
else:
    # Production definition with the Modal decorator.
    @app.function(
        image=scvi_image,
        timeout=86000,
        # cpu=40,
        volumes={backup_dir: vol}
    )
    def get_de(**kwargs):
        return _get_de_impl(**kwargs)


if __name__ == '__main__':
    print('Running function locally')

    common_kwargs = {"load_pair": True, "load_summary": True, "load_regions": False,
                "load_gsea": False, "load_gsea_heatmap": False, "tumor_is_int": False, "region_mapping": False,
                "n_jobs_inner": 8, "num_processes": 8, "parallel_pair": False, "parallel_summary": True,
                "gene_feature": None, "no_adata": False, "avoid_ensembl": True, "obs_has_name":True, "regions_AUC": True,
                "log_layer": None}

    # region Extended Atlas annots
    # get_de(**{
    #             "ext_name": 'Trinks_Bishoff_2021_NSCLC', "name": 'Bishoff', "pred_name": 'Subcluster_wu/Bishoff', "time": "III-IV", "skip_stages": True,
    #             "cell_key": "cell_type_adjusted", "stage_key": "Pathological stage",
    #             "log_layer": "do_log1p", **common_kwargs
    #         })

    # get_de(**{
    #             "ext_name": 'Hu_Zhang_2023_NSCLC', "name": 'Hu', "pred_name": 'Subcluster_wu/Hu', "time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "Clinical Stage",
    #             "log_layer": "do_log1p", **common_kwargs
    #         })    
    
    # get_de(**{
    #             "ext_name": 'Deng_Liu_LUAD_2024', "name": 'Deng', "pred_name": 'Subcluster_wu/Deng', "time": "I-II", "cell_key": "cell_type_adjusted", "stage_key": "Pathological stage",
    #             "log_layer": "data", **common_kwargs
    #         })


    # get_de(**{
    #             "ext_name": 'Deng_Liu_LUAD_2024', "name": 'Deng', "pred_name": 'Subcluster_wu/Deng', "time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "Pathological stage",
    #             "log_layer": "data", **common_kwargs
    #         })
    
    # get_de(**{
    #             "ext_name": 'Zuani_2024_NSCLC', "name": 'Zuani', "pred_name": 'Subcluster_wu/Zuani', "time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "stage",
    #             "log_layer": "do_log1p", **common_kwargs, "obs_unique": True
    #         })
    
    # get_de(**{
    #             "ext_name": 'Zuani_2024_NSCLC', "name": 'Zuani', "pred_name": 'Subcluster_wu/Zuani', "time": "I-II", "cell_key": "cell_type_adjusted", "stage_key": "stage",
    #             "log_layer": "do_log1p", **common_kwargs, "obs_unique": True
    #         })
    # endregion
    
    
    # region Altas annots
    
    # region Preamble
    no_adata = True
    if not no_adata:
        adata = sc.read_h5ad("/root/datos/maestria/netopaas/luca/data/atlas/extended_tumor_hvg.h5ad")
        dss = adata.obs["dataset"].unique()
        for ds in dss:
            if os.path.exists(f'{backup_dir}/filtered_{ds}.h5ad'):
                continue
            subset = adata[adata.obs["dataset"] == ds].copy()
            subset.write(f'{backup_dir}/filtered_{ds}.h5ad')
    else:
        import h5py
        from anndata.experimental import read_elem

        file_obj = h5py.File('/root/datos/maestria/netopaas/luca/data/atlas/extended_tumor_hvg.h5ad', 'r')
        
        obs_matrix = read_elem(file_obj['obs'])
        dss = obs_matrix['dataset'].unique()
        file_obj.close()
            

    dss = [ds for ds in dss if 'Wu' in ds]
    try:
        del adata
        del subset
        import gc
        gc.collect()
    except:
        pass
    # endregion
    # region Sequential
    for i, ds in enumerate(dss):
        try:
            # get_de(**{
            #     "ext_name": ds, "name": '-'.join(ds.split('_')[0:4:3]), "pred_name": 'Subcluster_wu/Atlas', "time": "I-II", "cell_key": "cell_type_adjusted", "stage_key": "uicc_stage",
            #     "log_layer": "do_log1p", **common_kwargs, "obs_has_name": False, "gene_feature": "feature_name"
            # })

            get_de(**{
                "ext_name": ds, "name": '-'.join(ds.split('_')[0:4:3]), "pred_name": 'Subcluster_wu/Atlas', "time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "uicc_stage",
                "log_layer": "do_log1p", **common_kwargs, "obs_has_name": False, "gene_feature": "feature_name"
            })
        except Exception as e:
            print(f'Error in {ds}: {e}')

    # endregion

    # region Parallel
    # def _run_get_de(params):
    #     get_de(**params)
    # futures = [None] * len(dss)
    # with multiprocessing.Pool(len(dss)) as pool:

    #     for i, ds in enumerate(dss):
    #         futures[i] = pool.apply_async(_run_get_de, [{
    #             "ext_name": ds, "name": '-'.join(ds.split('_')[0:4:3]), "pred_name": 'Atlas', "time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "uicc_stage",
    #             "log_layer": "do_log1p", "load_pair": False, "load_summary": False, "load_regions": False,
    #             "load_gsea": False, "load_gsea_heatmap": False, "tumor_is_int": False, "region_mapping": False,
    #             "n_jobs_inner": 10, "num_processes": 10, "parallel_pair": False, "parallel_summary": False,
    #             "gene_feature": "feature_name", "no_adata": False
    #         }])

    #     for future in futures:
    #         future.get()
    # endregion
    # endregion


## ACTIVATE THIS TO RUN THE FUNCTION IN MODAL
# @app.local_entrypoint()
def main():
    # Run the remote functions concurrently
    # We do this per dataset to avoid batch effects
    print("Starting differential expression analysis on the remote worker...")
    
    # Start both tasks
    future1 = get_de.spawn(ext_name="Zuani_2024_NSCLC", name='Zuani', time='I-II',
                           cell_key='cell_type_adjusted', stage_key='stage', log_layer='do_log1p',
                           load_pair = False, load_summary = False, load_regions = False,
                            load_gsea = False, load_gsea_heatmap = False,
                            tumor_is_int=False,
                            pred_name='Subcluster_wu/Zuani', n_jobs_inner=30, avoid_ensembl=True,
                              obs_has_name=True, parallel_summary=True, num_processes=30, obs_unique=True)
    
    # future2 = get_de.spawn(ext_name="Zuani_2024_NSCLC", name='Zuani', time='III-IV',
    #                        cell_key='cell_type_adjusted', stage_key='stage', log_layer='do_log1p',
    #                        load_pair = True, load_summary = False, load_regions = False,
    #                         load_gsea = False, load_gsea_heatmap = False,
    #                         tumor_is_int=True)
    
    # future3 = get_de.spawn(ext_name="Deng_Liu_LUAD_2024", name='Deng', time='III-IV',
    #                        cell_key='cell_type_adjusted', stage_key='Pathological stage', log_layer='data',
    #                        load_pair = False, load_summary = False, load_regions = False,
    #                         load_gsea = False, load_gsea_heatmap = False,
    #                         tumor_is_int=True)
    
    # future4 = get_de.spawn(ext_name="Deng_Liu_LUAD_2024", name='Deng', time='I-II',
    #                        cell_key='cell_type_adjusted', stage_key='Pathological stage', log_layer='data',
    #                        load_pair = False, load_summary = False, load_regions = False,
    #                         load_gsea = False, load_gsea_heatmap = False,
    #                         tumor_is_int=False,
    #                         pred_name='Subcluster_wu/Deng', n_jobs_inner=30, avoid_ensembl=True, obs_has_name=True, parallel_summary=True, num_processes=30)
    
    # future5 = get_de.spawn(ext_name="Hu_Zhang_2023_NSCLC", name='Hu', time='III-IV',
    #                        cell_key='cell_type_adjusted', stage_key='Clinical Stage', log_layer='do_log1p',
    #                        load_pair = False, load_summary = False, load_regions = False,
    #                         load_gsea = False, load_gsea_heatmap = False,
    #                         tumor_is_int=True, n_jobs_inner=5, parallel_pair=True)
    
    # future6 = get_de.spawn(ext_name="Trinks_Bishoff_2021_NSCLC", name='Bishoff', time='III-IV',
    #                        cell_key='cell_type_adjusted', skip_stages=True, log_layer='do_log1p',
    #                        load_pair = False, load_summary = False, load_regions = False,
    #                         load_gsea = False, load_gsea_heatmap = False,
    #                         tumor_is_int=True)
    
    # Wait for both tasks to complete
    future1.get()
    # future2.get()
    # future3.get()
    # future4.get()
    # future5.get()
    # future6.get()



