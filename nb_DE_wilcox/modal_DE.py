import os
from typing import Any, Iterable, List, Literal, Optional, Union
import warnings
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

import modal

app = modal.App("DE - wilcox")
local = True

local_dir = '/root/datos/maestria/netopaas/luca_explore/surgeries/'
# Define the remote path where the data will be available in the remote function
backup_dir = "/data" if not local else local_dir
w_folder = '/root/host_home/luca/nb_DE_wilcox/wilcoxon_DE' if local else backup_dir
subcluster_dir = f'{backup_dir}/Subcluster' if local else f'{backup_dir}/Subcluster'

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
            results.append(compare_groups(adata, groupby, group1, group2, method, use_raw, parallel))

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


scvi_image = modal.Image.from_registry(
    "ghcr.io/scverse/scvi-tools:py3.11-cu12-1.2.x-runtime")\
    .pip_install('gprofiler-official==1.0.0', 'gseapy==1.1.1', 'GEOparse==2.0.4')\
.pip_install('scanpy','matplotlib', 'seaborn')

# To delete mutiple files: modal volume ls --json DE-vol | jq -r '.[] | select(.Filename | test("^Tumor")) | .Filename' | xargs -I {} sh -c 'echo Deleting: {}; modal volume rm DE-vol "/{}"'
vol = modal.Volume.from_name("DE-vol", create_if_missing=True)


## ACTIVATE THIS TO RUN THE FUNCTION IN MODAL
# Remote function with GPU and mounted local directory
# @app.function(
#     image=scvi_image,
#     # gpu='any',
#     timeout=46000,
#     cpu=40,
#     volumes={backup_dir: vol}
# #    mounts=[
# #        modal.Mount.from_local_dir(
# #            local_path=local_data_dir,
# #            remote_path=remote_data_dir
# #        )
# #    ]
# )
def get_de(ext_name = "Zuani_2024_NSCLC", name = 'Zuani', time = 'I-II', skip_stages=False,
        cell_key = 'cell_type_adjusted', stage_key = 'stage', log_layer = 'do_log1p',
        num_processes = 1, load_pair = True, load_summary = True, load_regions = True,
        load_gsea = True, load_gsea_heatmap = True, zuani_symbols_summary = False,
        tumor_is_int = False, region_mapping='', n_jobs_inner=1, parallel_pair = False,
        no_adata = False, pred_name=None, parallel_summary=False,
        gene_mapping:Union[str,dict]=None, gene_feature=None
    ):
    """
    Computes differential expression analysis for cell subpopulations across
    specific timepoints and regions, with optional GSEA (Gene Set Enrichment
    Analysis) integration and marker genes.
    Parameters
        ----------
        ext_name : str, optional
            External name or ID for the dataset (default "Zuani_2024_NSCLC").
        name : str, optional
            Study or dataset name (default "Zuani").
        time : str, optional
            Stage designation (e.g., "I-II" or "III-IV") to control filtering
            (default "I-II").
        skip_stages : bool, optional
            If True, skip filtering by specific stages (default False).
        cell_key : str, optional
            Column name in external annotation (predictions) with cell identities
            (default "cell_type_adjusted").
        stage_key : str, optional
            Column name in adata.obs with stage annotations (default "stage").
        log_layer : str or bool, optional
            Layer to log-transform, or set to 'do_log1p' to apply sc.pp.log1p on
            the main matrix (default "do_log1p").
        num_processes : int, optional
            Number of parallel processes (default 1). For running mutiple pairwise comparisons at the same time. NOT RECOMMENDED, MUST COPY THE WHOLE ANNDATA.
            But it is useful for the summary statistics. When parallel_summary is True it will be used but is not useful if it > number of cell types.
            Also used in the GSEA computation.
        load_pair : bool, optional
            Whether to load existing pairwise comparisons from file (default True).
        load_summary : bool, optional
            Whether to load existing DE summary from file (default True).
        load_regions : bool, optional
            Whether to load existing region-level results from file (default True).
        load_gsea : bool, optional
            Whether to load existing GSEA results from file (default True).
        load_gsea_heatmap : bool, optional
            Whether to load existing GSEA heatmap data from file (default True).
        zuani_symbols_summary : bool, optional
            If True, updates gene identifiers using a symbol map CSV (default False).
        tumor_is_int : bool, optional
            If True, identifies tumor subpopulations as integer labels (default False).
        region_mapping : str, optional
            Path to JSON file for optional region label mapping. Only renames the regions object and the plots.
            The mapper can refer to only one cell type and it will work, leaving all the other ones as they were before (default ""). 
        n_jobs_inner : int, optional
            Internal parallelization for pairwise comparisons. Does not really work (default 10).
        parallel_pair : bool, optional
            Enables pairwise Wilcoxon parallel execution if True/ NOT RECOMMENDED, MUST COPY THE WHOLE ANNDATA (default False).
        no_adata : bool, optional
            If True, does not load or integrate with an AnnData object (default False).
        pred_name : str, optional
            Suffix of the prediction file to use for filtering (default None).
        parallel_summary : bool, optional
            Enables parallel execution for summary statistics (default False).
        gene_mapping : Union[str, dict], optional
            obs column name or dictionary for gene name mapping (default None).
        gene_feature : str, optional
            Feature name to assign to adata.var.index (default None).
    Returns
    -------
        None
            All results are saved to disk and/or integrated into the provided AnnData
            object according to the function parameters.
    Notes
    -----
        - Raises an exception if no tumor cell groups are found.
        - Incorporates GSEA computations (Hallmark gene sets) if data is available.
        - Logs various progress updates and may commit results to a remote storage if
            not running locally.
        - Supports reading and writing intermediate data to disk, enabling reusability
            of pairwise DE comparisons, summary statistics, and region-level analyses.
        - Adjusts the main data log transform based on the `log_layer` parameter,
            either using in-memory sc.pp.log1p or an existing log-transformed layer.
    """

    if no_adata: 
        if not load_pair:
            raise Exception("If no adata is provided, the pairwise comparison must be loaded or not used")
        if isinstance(gene_mapping, str):
            raise Exception("If no adata is provided, the gene mapping must be a dictionary or None")
    
    if not local:
        print("This code is running on a remote worker!")
    print(f'Marker genes for {name} at {time} are being computed...')
    
    pred_name = pred_name if pred_name else name
    id_ = ext_name
    time_suffix = 'early' if 'I-II' in time else 'late'
    
    all_path = f'{w_folder}/{time}_{id_}'
    key_pair = "rank_genes_groups_tumor"
    regions = ['tumorall']
    region = 'tumorall' # This we hardocde beacuse the beginning is not coded for using multiple regions

    preds = pd.read_csv(f'{backup_dir}/{pred_name}_predicted_leiden_{time_suffix}.csv', index_col=0)
    if pred_name == 'Atlas':
        preds = preds[preds.batch == id_]

    if 'I-II' in time:
        stages = ['IA1', 'IB', 'IA2', 'IA3', 'IIB', 'II', 'I']
    if 'III-IV' in time:
        stages = ['IIIA', 'IIIB','III', 'III or IV', 'IV']
    stages = None if skip_stages else stages
    
    if not no_adata:
        adata = ad.read_h5ad(f'{backup_dir}/filtered_{ext_name}.h5ad')
        
        try:
            adata = adata[preds.index].copy()
            adata.obs.loc[preds.index, cell_key] = preds.loc[preds.index, cell_key]
        except:
            Exception("The index of the predictions does not match the index of the adata")
        adata.obs['type_tissue'] = adata.obs[cell_key]
        # print(adata)

        adata = adata[adata.obs[stage_key].isin(stages)].copy() if stages else adata

        preds = preds.loc[adata.obs.index]

        if log_layer == 'do_log1p':
            sc.pp.log1p(adata)
        elif log_layer:
            adata.X = adata.layers[log_layer]
        print("CHECKING THE DATA IS LOGARITHMED")
        print(adata[:10,10:20].to_df())
        if gene_feature:
            adata.var.index = adata.var[gene_feature]

        if isinstance(gene_mapping, str):
            gene_mapping = adata.var.loc[:, gene_mapping].to_dict()

        import gc
        gc.collect()
    else:
        import h5py
        from anndata.experimental import read_elem

        with h5py.File(f'{backup_dir}/filtered_{ext_name}.h5ad', 'r') as f:
            # Check if 'obs' group exists (common for scRNA-seq data)
            if 'obs' in f:
                obs_matrix = read_elem(f['obs'])
            else:
                # Handle the case where 'obs' is not present
                raise("obs matrix not found in the h5ad file.")

        preds[stage_key] = obs_matrix[stage_key]
        preds = preds[preds[stage_key].isin(stages)].copy() if stages else preds

        del obs_matrix

    # TODO This should determine regions not only tumor groups
    valid_types = list(preds[cell_key].value_counts().loc[lambda x: x > 2].index)
    types = preds[cell_key].unique()
    if not tumor_is_int:
        tumor_types = [g for g in valid_types if 
            any(x in g for x in ['Tumor','Ciliated','AT2', 'AT1', 'Club'])]
    else:
        tumor_types = [g for g in valid_types if g.isdigit()]

    if len(tumor_types) == 0:
        warnings.warn("No tumor groups found. COMPARISON WILL BE DONE FOR ALL TYPES")

    if not load_summary:
        print("Loading Pairwise")
        if load_pair:
            DE_pair = np.load(all_path + '_tumorpair.npy', allow_pickle='TRUE').item()
            if not no_adata:
                adata.uns[key_pair] = DE_pair
        else:
            # Adress this: is sensitive to the population composition, which introduces an element of unpredictability to the marker sets due to variation in cell type abundances
            # SOLVED by using pairwise wilcoxon
            DE_pair = rank_genes_groups_pairwise(adata, 'type_tissue', method='wilcoxon', use_raw=False,
                            groups=valid_types, subgroups=tumor_types, parallel=parallel_pair,
                            n_jobs=num_processes//n_jobs_inner, n_jobs_inner=n_jobs_inner)
            if not no_adata:
                adata.uns[key_pair] = DE_pair
            np.save(all_path + '_tumorpair.npy', DE_pair)
            if not local:
                vol.commit()
        
        DE_pair = np.load(all_path + '_tumorpair.npy', allow_pickle='TRUE').item()
        if not no_adata:
            adata.uns[key_pair] = DE_pair

    if len(tumor_types) == 0:
        raise Exception("No tumor groups found THE SUMMARIES WOULD BE EMPTY")
    DE_summary = {group: {} for group in tumor_types}
    
    print("Loading Summary")
    if not load_summary:
        if parallel_summary:
            with multiprocessing.Pool(num_processes) as pool:
                for group_scores in pool.starmap(process_gene,
                                                    [(group1, DE_pair, valid_types) for group1 in tumor_types]):
                    DE_summary[group_scores[0]] = group_scores[1]
        else:
            for group1 in tumor_types:
                group_scores = process_gene(group1, DE_pair, valid_types)
                DE_summary[group_scores[0]] = group_scores[1]

        if not no_adata:
            adata.uns[f'rank_genes_groups_summary_{region}'] = DE_summary

        np.save(all_path + f'_summary_{region}.npy', DE_summary)
        if not local:
            vol.commit()

    elif not load_regions:
        DE_summary = np.load(
            all_path + f'_summary_{region}.npy', allow_pickle='TRUE').item()
        
        if not no_adata:
            adata.uns[f'rank_genes_groups_summary_{region}'] = DE_summary
        
        if zuani_symbols_summary:
            gene_name_map = pd.read_csv(
                f'{backup_dir}/zuani_ensembl.csv')\
                .iloc[:,:1]
            gene_name_map = gene_name_map.to_dict()['Unnamed: 0']
            gene_name_map = {str(k): v for k, v in gene_name_map.items()}
            
            print('THE GENE MAP')
            print(str(gene_name_map)[:100])
            for cell_type, genes_dict in DE_summary.items():
                updated_genes_dict = {}
                for gene, scores in genes_dict.items():
                    new_gene_name = gene_name_map.get(gene, gene)
                    updated_genes_dict[new_gene_name] = scores
                DE_summary[cell_type] = updated_genes_dict

            if not no_adata:
                adata.uns[f'rank_genes_groups_summary_{region}'] = DE_summary


    print("Loading Regions")
    for region in regions:
        if load_regions:
            DE_region = np.load(
                all_path + f'_{region}.npy', allow_pickle=True).item()
            if not no_adata:
                adata.uns[f'rank_genes_groups_{region}'] = DE_region
            continue

        regioner = {cell_type: {gene: np.mean(scores) for gene, scores
                                 in genes_dict.items()}
                        for cell_type, genes_dict in
                          DE_summary.items()}
        regioner = {cell_type: sorted(genes.items(), key=lambda k: k[1], reverse=True)
                     for cell_type, genes in regioner.items()}

        types_num = len(regioner)
        genes_num = len(next(iter(DE_summary.values())).keys())
        scores = [[None] * genes_num for _ in range(types_num)]
        names = [[None] * genes_num for _ in range(types_num)]
        
        for i, cell_type in enumerate(regioner.keys()):
            j = 0
            for gene, score in regioner[cell_type]:
                scores[i][j] = score
                names[i][j] = gene
                j += 1

        names_zip = list(zip(*names))
        dtypes = [(typer, float) for typer in regioner.keys()]
        dtypes_names = [(typer, 'O') for typer in regioner.keys()]

        i = 0
        for n_scores in zip(*scores):
            if i==0:
                rank_scores = np.rec.array([n_scores], dtype=dtypes)
                rank_names = np.rec.array([names_zip[i]], dtype=dtypes_names)
            else:
                temp_scores = np.rec.array([n_scores], dtype=dtypes)
                rank_scores = np.rec.array(np.concatenate((rank_scores, temp_scores)))
        
                temp_names = np.rec.array([names_zip[i]], dtype=dtypes_names)
                rank_names = np.rec.array(np.concatenate((rank_names, temp_names)))
            i += 1

        DE_region: dict[str,Union[dict, np.recarray]] = {'params': {'groupby': 'type_tissue', 'reference': region,
                                'method': 'wilcoxon', 'use_raw': False,
                                'layer': None, 'corr_method': 'benjamini-hochberg'}}
        DE_region['names'] = rank_names
        DE_region['scores'] = rank_scores

        assert isinstance(DE_region['names'], np.recarray) and isinstance(DE_region['scores'], np.recarray)

        if not no_adata:
            adata.uns[f'rank_genes_groups_{region}'] = DE_region
        np.save( f'{all_path}_{region}.npy',
                 DE_region, allow_pickle=True) # type: ignore
        if not local:
            vol.commit()

    print('Plotting Marker genes')
    

    if region_mapping:
        mapper = json.load(open(region_mapping))

        valid_types = [mapper.get(t, t) for t in valid_types]

        types = DE_region['scores'].dtype.names
        DE_region['scores'].dtype.names = tuple(mapper.get(name, name) for name in types )
        DE_region['names'].dtype.names = tuple(mapper.get(name, name) for name in types )

        if not no_adata:
            adata.uns[f'rank_genes_groups_{region}'] = DE_region
        np.save( f'{all_path}_{region}.npy',
                 DE_region)
        if not local:
            vol.commit()

    types = DE_region['scores'].dtype.names
    num_types = len(types)


    fig, axs = plt.subplots((num_types + 1) // 2, 2, figsize=(16, 4.5 * ((num_types + 1) // 2)))
    n_genes = 20
    for i, type in enumerate(types):
        titlesize = 9
        fontsize = 6
        row = i // 2
        col = i % 2
        cond_plot(DE_region, [type], valid_types, n_genes=n_genes,
                  ax=axs[row, col], sharey=False, key=f'rank_genes_groups_{region}', show=False,
                  fontsize=fontsize, titlesize=titlesize, gene_mapping=gene_mapping)
    # Hide any unused subplots
    if num_types % 2 != 0:
        fig.delaxes(axs[-1, -1])
    plt.savefig(f'{w_folder}/markergenes_{name}_{region}_{time}.png', bbox_inches='tight')
    if not local:
        vol.commit()

    print("Plotting GSEA")
    if not os.path.exists('h.all.v2023.2.Hs.symbols.gmt'):
        import subprocess
        subprocess.run(["wget", "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt"])
    
    gsea_folder = f'{w_folder}/../gseapy_gsea' if local else f'{w_folder}/gseapy_gsea'
    if not os.path.exists(gsea_folder):
        os.makedirs(gsea_folder)
    combined_dfs = {}

    for region in regions:
        gsea_path = f'{gsea_folder}/heatmap_{name}_{region}_{time}.csv'
        if load_gsea_heatmap:
            combined_dfs[region] = pd.read_csv(gsea_path, index_col=0)
        else:
            types = DE_region['scores'].dtype.names

            combined_dfs[region] = get_gseas_df(DE_region, valid_types, types,
                id_, load_gsea=load_gsea,
                gsea_folder=gsea_folder, gene_mapping=gene_mapping)
            
            combined_dfs[region].to_csv(gsea_path)
            if not local:
                vol.commit()
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(combined_dfs[region], cmap='viridis')
        plt.title(f'Hallmarks Scores by Cell Type for {region}')
        plt.xlabel('Hallmarks')
        plt.ylabel('Cell Types')
        plt.savefig(f'{gsea_folder}/heatmap_{name}_{region}_{time}.png', bbox_inches='tight')
        if not local:
            vol.commit()

    return None

if __name__ == '__main__':
    print('Running function locally')

    # get_de(ext_name="Deng_Liu_LUAD_2024", name='Deng', time='III-IV', cell_key='cell_type_adjusted', 
    #                        no_adata=True, skip_stages=True, stage_key='stage', log_layer='do_log1p',
    #                        load_pair = True, load_summary = True, load_regions = True,
    #                         load_gsea = False, load_gsea_heatmap = False,
    #                         tumor_is_int=False, region_mapping='cell_map_late.json')
    
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

        with read_elem(h5py.File(f'{backup_dir}/filtered_{ext_name}.h5ad', 'r')) as f:
            adata = f

    dss = dss[5:6]
    try:
        del adata
        del subset
        import gc
        gc.collect()
    except:
        pass
    

    def _run_get_de(params):
        get_de(**params)



    futures = [None] * len(dss)
    with multiprocessing.Pool(len(dss)) as pool:

        for i, ds in enumerate(dss):
            futures[i] = pool.apply_async(_run_get_de, [{
                "ext_name": ds, "name": '-'.join(ds.split('_')[0:4:3]), "pred_name": 'Atlas', "time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "uicc_stage",
                "log_layer": "do_log1p", "load_pair": True, "load_summary": False, "load_regions": False,
                "load_gsea": False, "load_gsea_heatmap": False, "tumor_is_int": False, "region_mapping": False,
                "n_jobs_inner": 10, "num_processes": 10, "parallel_pair": False, "parallel_summary": False,
                "gene_feature": "feature_name", "no_adata": True
            }])

        for future in futures:
            future.get()


        # future1 = pool.apply_async(_run_get_de, [{
        #     "ext_name": "Hu_Zhang_2023_NSCLC", "name": "Hu", "time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "Clinical Stage",
        #     "log_layer": "do_log1p", "load_pair": True, "load_summary": True, "load_regions": True,
        #     "load_gsea": False, "load_gsea_heatmap": False, "tumor_is_int": True, "region_mapping": False
        # }])

        # future2 = pool.apply_async(_run_get_de, [{
        #     "ext_name": "Deng_Liu_LUAD_2024", "name": "Deng","time": "I-II", "cell_key": "cell_type_adjusted", "stage_key": "Pathological stage",
        #     "log_layer": "data", "load_pair": True, "load_summary": True, "load_regions": True,
        #     "load_gsea": False, "load_gsea_heatmap": False, "tumor_is_int": True, "region_mapping": True
        # }])

        # future3 = pool.apply_async(_run_get_de, [{
        #     "ext_name": "Deng_Liu_LUAD_2024", "name": "Deng","time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "Pathological stage",
        #     "log_layer": "data", "load_pair": True, "load_summary": True, "load_regions": True,
        #     "load_gsea": False, "load_gsea_heatmap": False, "tumor_is_int": True, "region_mapping": False
        # }])

        # future4 = pool.apply_async(_run_get_de, [{
        #     "ext_name": "Zuani_2024_NSCLC", "name": "Zuani","time": "I-II", "cell_key": "cell_type_adjusted", "stage_key": "stage",
        #     "log_layer": "do_log1p", "load_pair": True, "load_summary": True, "load_regions": True,
        #     "load_gsea": False, "load_gsea_heatmap": False, "tumor_is_int": True, "region_mapping": False
        # }])

        # future5 = pool.apply_async(_run_get_de, [{
        #     "ext_name": "Zuani_2024_NSCLC", "name": "Zuani","time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "stage",
        #     "log_layer": "do_log1p", "load_pair": True, "load_summary": True, "load_regions": True,
        #     "load_gsea": False, "load_gsea_heatmap": False, "tumor_is_int": True, "region_mapping": False
        # }])

        # future1.get()
        # future2.get()
        # future3.get()
        # future4.get()
        # future5.get()

# Main entry point
@app.local_entrypoint()
def main():
    # Run the remote functions concurrently
    # We do this per dataset to avoid batch effects
    print("Starting differential expression analysis on the remote worker...")
    
    # Start both tasks
    # future1 = get_de.spawn(ext_name="Zuani_2024_NSCLC", name='Zuani', time='I-II',
    #                        cell_key='cell_type_adjusted', stage_key='stage', log_layer='do_log1p',
    #                        load_pair = False, load_summary = False, load_regions = False,
    #                         load_gsea = False, load_gsea_heatmap = False,
    #                         tumor_is_int=True)
    
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
    #                         tumor_is_int=True)
    
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
    # future1.get()
    # future2.get()
    # future3.get()
    # future4.get()
    # future5.get()
    # future6.get()



