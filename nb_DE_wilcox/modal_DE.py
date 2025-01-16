import os
from typing import Iterable, List, Literal, Optional, Union
import warnings
import multiprocessing
import json

import numpy as np
import anndata as ad
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

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

def get_gseas_df(adata: ad.AnnData, valid_types: List[str],
                 types: List[str], id_: str, load_gsea: bool = False,
                 key:str = 'rank_gene_groups', gsea_folder:str = 'gseapy_gsea') -> pd.DataFrame:
    """
    Generates a combined DataFrame from GSEA results for different cell types.

    Args:
    adata (ad.AnnData): An AnnData object containing the dataset.
    valid_types (List[str]): A list of valid cell types to filter the cell types.
    types (List[str]): A list of cell types to be considered for analysis.
    id_ (str): Identifier used in generating the GSEA path.
    load_gsea (bool, optional): Flag to load existing GSEA results. Defaults to False.

    Returns:
    pd.DataFrame: A combined DataFrame with GSEA results pivoted by cell type and hallmark.

    The function uses the provided 'types', intersects them with 'valid_types',
    and computes GSEA for each type. The final DataFrame is a pivot table with cell types as rows
    and GSEA hallmarks as columns.
    """
    # Intersect types with valid types
    types = set(types).intersection(set(valid_types))
    dfs = []

    for type in types:
        ranks = adata.uns[key]
        type_path = type.replace('/','_')
        gsea_path = f'{gsea_folder}/{type_path}_{id_}'

        # Generate gene scores
        gene_scores = {gene: ranks['scores'][type][i] for i, gene
                        in enumerate(ranks['names'][type])}
        gene_scores = pd.Series(gene_scores)

        # Run GSEA
        if load_gsea:
            gseas = np.load(gsea_path, allow_pickle='TRUE').item()
        else:
            pre_res = gseapy.prerank(rnk=gene_scores,
                                     gene_sets='h.all.v2023.2.Hs.symbols.gmt',
                                     processes=20, permutation_num=100,
                                     seed=6, no_plot=True)
            gseas = pre_res.res2d
            # We remove this because it clutters the folders and  though it has more info it is not used
            # np.save(gsea_path, gseas)

        data1 = {'hallmark': list(gseas['Term'].str.slice(9)), 'score': list(gseas['NES'])}
        df_celltype1 = pd.DataFrame(data1)
        df_celltype1['cell_type'] = type
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
    adata: ad.AnnData,
    group_name: str,
    n_genes: int = 20,
    gene_symbols: Optional[str] = None,
    gene_names: Optional[List[str]] = None,
    key: str = 'rank_genes_groups',
    fontsize: int = 8,
    titlesize: int = 10,
    show: Optional[bool] = None,
    ax: Optional[plt.Axes] = None,
    **kwds,
) -> plt.Axes:
    """
    Visualizes the ranking of genes for a specified group from an AnnData object.

    Parameters:
        adata: ad.AnnData
            The AnnData object containing the dataset and analysis results.
        group_name: str
            The name of the group for which to rank genes.
        n_genes: int, optional
            The number of top genes to display (default is 20).
        gene_symbols: Optional[str], optional
            The key in adata.var where alternative gene symbols are stored.
        gene_names: Optional[List[str]], optional
            Explicit list of gene names to use for plotting.
        key: str, optional
            The key in adata.uns where the ranking is stored (default is 'rank_genes_groups').
        fontsize: int, optional
            Font size for gene names (default is 8).
        titlesize: int, optional
            Font size for the title (default is 10).
        show: Optional[bool], optional
            If True, show the plot immediately.
        ax: Optional[plt.Axes], optional
            A matplotlib axes object to plot on. If None, a new figure is created.
        **kwds:
            Additional keyword arguments to pass to plotting functions.

    Returns:
        plt.Axes:
            The matplotlib axes with the plot.

    Raises:
        ValueError:
            If n_genes is less than 1.
    """
    if n_genes < 1:
        raise ValueError(f"n_genes must be positive; received n_genes={n_genes}.")

    reference = str(adata.uns[key]['params']['reference'])

    try:
        gene_names = gene_names if gene_names is not None else adata.uns[key]['names'][group_name][:n_genes]
        gene_mask = np.isin(adata.uns[key]['names'][group_name], gene_names)
        scores = adata.uns[key]['scores'][group_name][gene_mask]
    except Exception as e:
        scores = adata.uns[key]['scores'][group_name][:n_genes]

    ymin = np.min(scores)
    ymax = np.max(scores)
    ymax += 0.3 * (ymax - ymin)

    ax = ax if ax is not None else plt.subplot(111)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.9, n_genes - 0.1)

    if gene_symbols:
        gene_names = adata.var[gene_symbols].loc[gene_names].tolist()

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


scvi_image = modal.Image.from_registry(
    "ghcr.io/scverse/scvi-tools:py3.11-cu12-1.2.x-runtime")\
    .pip_install('gprofiler-official==1.0.0', 'gseapy==1.1.1', 'GEOparse==2.0.4')\
.pip_install('scanpy','matplotlib', 'seaborn')

# To delete mutiple files: modal volume ls --json DE-vol | jq -r '.[] | select(.Filename | test("^Tumor")) | .Filename' | xargs -I {} sh -c 'echo Deleting: {}; modal volume rm DE-vol "/{}"'
vol = modal.Volume.from_name("DE-vol", create_if_missing=True)

# TODO does notwork
def upload_files_to_volume():

    file_dict = [{'local':f'{local_dir}/{name}_predicted_leiden.csv',
                 'remote':f'{backup_dir}/{name}_predicted_leiden.csv'},
                 {'local':f'{local_dir}/{name}/filtered_{ext_name}.h5ad',
                 'remote':f'{backup_dir}/filtered_{ext_name}.h5ad'},
                 {'local':f'wilcoxon_DE/{time}_{id_}_tumorpair.npy',
                 'remote':f'{backup_dir}/{time}_{id_}_tumorpair.npy'},]
    
    for file in file_dict:
        with vol.mount() as mount_path:
            file_path = os.path.join(mount_path, file['remote'])
            try:
                with open(file_path, 'rb'):
                    continue
            except FileNotFoundError:
                pass
            with open(file['local'], 'rb') as f_local:
                with open(file_path, 'wb') as f_remote:
                    f_remote.write(f_local.read())


# upload_files_to_volume()

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
        num_processes = 40, load_pair = True, load_summary = True, load_regions = True,
        load_gsea = True, load_gsea_heatmap = True, zuani_symbols_summary = False,
        tumor_is_int = False, region_mapping=False, n_jobs_inner=10, parallel_pair = False,
    ):
    
    if not local:
        print("This code is running on a remote worker!")
    print(f'Marker genes for {name} at {time} are being computed...')
    
    id_ = ext_name
    time_suffix = 'early' if 'I-II' in time else 'late'
    
    all_path = f'{w_folder}/{time}_{id_}'
    key_pair = "rank_genes_groups_tumor"
    regions = ['tumorall']
    region = 'tumorall' # This we hardocde beacuse the beginning is not coded for using multiple regions

    adata = ad.read_h5ad(f'{backup_dir}/filtered_{ext_name}.h5ad')
    if not region_mapping:
        preds = pd.read_csv(f'{backup_dir}/{name}_predicted_leiden_{time_suffix}.csv', index_col=0)
    else:
        preds = pd.read_csv(f'{w_folder}/{name}_predicted_leiden_{time_suffix}.csv', index_col=0)
    
    preds.index = adata.obs.index

    adata.obs[cell_key] = preds[cell_key]
    adata.obs['type_tissue'] = adata.obs[cell_key]

    print(adata)

    stages = None
    if 'I-II' in time:
        stages = ['IA1', 'IB', 'IA2', 'IA3', 'IIB', 'II']
    if 'III-IV' in time:
        stages = ['IIIA', 'IIIB','III', 'III or IV', 'IV']
    stages = None if skip_stages else stages
    adata = adata[adata.obs[stage_key].isin(stages)].copy() if stages else adata

    if log_layer == 'do_log1p':
        sc.pp.log1p(adata)
    elif log_layer:
        adata.X = adata.layers[log_layer]
    print("CHECKING THE DATA IS LOGARITHMED") 
    print(adata[:10,10:20].to_df())

    import gc
    gc.collect()

    # TODO This should determine regions not only tumor groups
    valid_types = list(adata.obs['type_tissue'].value_counts().loc[lambda x: x > 2].index)
    types = adata.obs.type_tissue.unique()
    if not tumor_is_int:
        tumor_types = [g for g in valid_types if 'Tumor' in g]
    else:
        tumor_types = [g for g in valid_types if g.isdigit()]

    if len(tumor_types) == 0:
        warnings.warn("No tumor groups found. COMPARISON WILL BE DONE FOR ALL TYPES")

    print("Loading Pairwise")
    if load_pair:
        adata.uns[key_pair] = np.load(all_path + '_tumorpair.npy', allow_pickle='TRUE').item()
    else:
        # Adress this: is sensitive to the population composition, which introduces an element of unpredictability to the marker sets due to variation in cell type abundances
        # Solved by using pairwise wilcoxon
        results = rank_genes_groups_pairwise(adata, 'type_tissue', method='wilcoxon', use_raw=False,
                        groups=valid_types, subgroups=tumor_types, parallel=parallel_pair,
                        n_jobs=num_processes//n_jobs_inner, n_jobs_inner=n_jobs_inner)
        adata.uns[key_pair] = results
        np.save(all_path + '_tumorpair.npy', adata.uns[key_pair])
        if not local:
            vol.commit()
    
    adata.uns[key_pair] = np.load(all_path + '_tumorpair.npy', allow_pickle='TRUE').item()
    results = adata.uns[key_pair]

    if not tumor_is_int:
        groups = [ group for group in valid_types if 'Tumor' in group]
    else:
        groups = [g for g in valid_types if g.isdigit()]

    if len(groups) == 0:
        raise Exception("No tumor groups found THE SUMMARIES WOULD BE EMPTY")
    scores_dict = {group: {} for group in groups}
    
    print("Loading Summary")
    if not load_summary:
        with multiprocessing.Pool(num_processes) as pool:
            for group_scores in pool.starmap(process_gene,
                                                [(group1, results, valid_types) for group1 in groups]):
                scores_dict[group_scores[0]] = group_scores[1]

        adata.uns[f'rank_genes_groups_summary_{region}'] = scores_dict
        np.save(all_path + f'_summary_{region}.npy', scores_dict)
        if not local:
            vol.commit()
    elif not load_regions:
        adata.uns[f'rank_genes_groups_summary_{region}'] = np.load(
            all_path + f'_summary_{region}.npy', allow_pickle='TRUE').item()
        
        if zuani_symbols_summary:
            gene_name_map = pd.read_csv(
                f'{backup_dir}/zuani_ensembl.csv')\
                .iloc[:,:1]
            gene_name_map = gene_name_map.to_dict()['Unnamed: 0']
            gene_name_map = {str(k): v for k, v in gene_name_map.items()}
            
            print('THE GENE MAP')
            print(str(gene_name_map)[:100])
            for cell_type, genes_dict in adata.uns[f'rank_genes_groups_summary_{region}'].items():
                updated_genes_dict = {}
                for gene, scores in genes_dict.items():
                    new_gene_name = gene_name_map.get(gene, gene)
                    updated_genes_dict[new_gene_name] = scores
                adata.uns[f'rank_genes_groups_summary_{region}'][cell_type] = updated_genes_dict

    # regions = ['normalall']

    print("Loading Regions")
    for region in regions:
        if load_regions:
            adata.uns[f'rank_genes_groups_{region}'] = np.load(
                all_path + f'_{region}.npy', allow_pickle=True).item()
            continue

        regioner = {cell_type: {gene: np.mean(scores) for gene, scores
                                 in genes_dict.items()}
                        for cell_type, genes_dict in
                          adata.uns[f'rank_genes_groups_summary_{region}'].items()}
        regioner = {cell_type: sorted(genes.items(), key=lambda k: k[1], reverse=True)
                     for cell_type, genes in regioner.items()}

        types_num = len(regioner)
        genes_num = len(next(iter(results.values()))['names'])
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

        rec_region = {'params': {'groupby': 'type_tissue', 'reference': region,
                                'method': 'wilcoxon', 'use_raw': False,
                                'layer': None, 'corr_method': 'benjamini-hochberg'}}
        rec_region['names'] = rank_names
        rec_region['scores'] = rank_scores

        adata.uns[f'rank_genes_groups_{region}'] = rec_region
        np.save( f'{all_path}_{region}.npy',
                 adata.uns[f'rank_genes_groups_{region}'])
        if not local:
            vol.commit()

    print('Plotting Marker genes')
    def cond_plot(adata, cond_types, valid_types, ax=None,
                key='wilcoxon', fontsize=9, titlesize=14, **kwds):
        if set(cond_types).issubset(valid_types):
            rank_genes_group(adata, cond_types[0], n_genes=n_genes,
                                ax=ax, sharey=False, key=key, show=False,
                                fontsize=fontsize, titlesize=titlesize)
        else:
            # pass
            # Draw an empty plot with a message
            ax.text(0.5, 0.5, f'Missing cells: {cond_types}', color='red',
                        ha='center', va='center', transform=ax.transAxes) 
            ax.axis('off')

    region = 'tumorall'

    if region_mapping:
        mapper = json.load(open(f'{subcluster_dir}/mapping_{time_suffix}_leiden.json'))

        valid_types = [mapper.get(t, t) for t in valid_types]

        types = adata.uns[f'rank_genes_groups_{region}']['scores'].dtype.names
        adata.uns[f'rank_genes_groups_{region}']['scores'].dtype.names = tuple(mapper[name] for name in types )
        adata.uns[f'rank_genes_groups_{region}']['names'].dtype.names = tuple(mapper[name] for name in types )


        np.save( f'{all_path}_{region}.npy',
                 adata.uns[f'rank_genes_groups_{region}'])
        if not local:
            vol.commit()

    types = adata.uns[f'rank_genes_groups_{region}']['scores'].dtype.names
    num_types = len(types)
    fig, axs = plt.subplots((num_types + 1) // 2, 2, figsize=(16, 4.5 * ((num_types + 1) // 2)))
    n_genes = 20
    for i, type in enumerate(types):
        titlesize = 9
        fontsize = 6
        row = i // 2
        col = i % 2
        cond_plot(adata, [type], valid_types, n_genes=n_genes,
                  ax=axs[row, col], sharey=False, key=f'rank_genes_groups_{region}', show=False,
                  fontsize=fontsize, titlesize=titlesize)
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
            types = adata.uns[f'rank_genes_groups_{region}']['scores'].dtype.names

            combined_dfs[region] = get_gseas_df(adata, valid_types, types,
                id_, load_gsea=load_gsea, key=f'rank_genes_groups_{region}',
                gsea_folder=gsea_folder)
            
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

    get_de(ext_name="Zuani_2024_NSCLC", name='Zuani', time='I-II',
                           cell_key='cell_type_adjusted', skip_stages=False, stage_key='stage', log_layer='do_log1p',
                           load_pair = True, load_summary = True, load_regions = True,
                            load_gsea = False, load_gsea_heatmap = False,
                            tumor_is_int=False, region_mapping=False)

    def _run_get_de(params):
        get_de(**params)

    # with multiprocessing.Pool(2) as pool:
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


