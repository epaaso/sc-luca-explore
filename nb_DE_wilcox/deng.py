import modal
import numpy as np
from typing import List
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

app = modal.App("DE - wilcox")

# Specify your local data directory
local_dir = '/root/datos/maestria/netopaas/luca_explore/surgeries/'

# Define the remote path where the data will be available in the remote function
backup_dir = "/data"

# Paths within the mounted directory
ext_name = "Deng_Liu_LUAD_2024"
name = 'Deng'
id_ = ext_name

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
    types = set(types).intersection(valid_types)
    dfs = []

    for type in types:
        ranks = adata.uns[key]
        type_path = type.replace('/','_')
        gsea_path = f'{gsea_folder}/{type_path}_{id_}.npz'

        # Generate gene scores
        gene_scores = {gene: ranks['scores'][type][i] for i, gene in enumerate(ranks['names'][type])}
        gene_scores = pd.Series(gene_scores)

        # Run GSEA
        if load_gsea:
            gseas = np.load(gsea_path, allow_pickle='TRUE').item()
        else:
            pre_res = gseapy.prerank(rnk=gene_scores, gene_sets='h.all.v2023.2.Hs.symbols.gmt',
                                     processes=20, permutation_num=100, seed=6, no_plot=True)
            gseas = pre_res.res2d
            np.save(gsea_path, gseas)

        data1 = {'hallmark': list(gseas['Term'].str.slice(9)), 'score': list(gseas['NES'])}
        df_celltype1 = pd.DataFrame(data1)
        df_celltype1['cell_type'] = type
        dfs.append(df_celltype1)

    # Combine all DataFrames
    combined_df = pd.concat(dfs).reset_index(drop=True).pivot(index='cell_type', columns='hallmark', values='score')
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


scvi_image = modal.Image.from_registry(
    "ghcr.io/scverse/scvi-tools:py3.11-cu12-1.2.x-runtime")\
    .pip_install('gprofiler-official==1.0.0', 'gseapy==1.1.1', 'GEOparse==2.0.4')\
    .pip_install('matplotlib==3.4.3', 'seaborn==0.11.2')

vol = modal.Volume.from_name("DE-vol", create_if_missing=True)


# Remote function with GPU and mounted local directory
@app.function(
    image=scvi_image,
    # gpu='any',
    timeout=10000,
    cpu=20,
    volumes={"/data": vol}
#    mounts=[
#        modal.Mount.from_local_dir(
#            local_path=local_data_dir,
#            remote_path=remote_data_dir
#        )
#    ]
)
def get_de():
    print("This code is running on a remote worker!")
    cell_key = 'cell_type_adjusted'
    load_pair = True
    load_summary = False
    load_regions = False
    load_gsea = False
    load_gsea_heatmap = False
    num_processes = 20
    time = 'I-IV'
    
    import multiprocessing

    adata = ad.read_h5ad(f'{backup_dir}/filtered_{ext_name}.h5ad')

    preds = pd.read_csv(f'{backup_dir}/{name}_predicted_leiden.csv')
    preds.index = adata.obs.index
    adata.obs[cell_key] = preds[cell_key]
    adata.obs['type_tissue'] = adata.obs[cell_key]

    valid_types = list(adata.obs['type_tissue'].value_counts().loc[lambda x: x > 2].index)
    types = adata.obs.type_tissue.unique()
    w_folder = backup_dir

    # Load

    all_path = f'{w_folder}/{time}_{id_}'
    key_pair = "rank_genes_groups_tumor"

    tumor_types = [g for g in valid_types if 'Tumor' in g]

    adata.uns[key_pair] = np.load(all_path + '_tumorpair.npy', allow_pickle='TRUE').item()
    results = adata.uns[key_pair]

    groups = [ group for group in valid_types if 'Tumor' in group]
    scores_dict = {group: {} for group in groups}
    
    print("Loading Summary")
    if not load_summary:
        with multiprocessing.Pool(num_processes) as pool:
            for group_scores in pool.starmap(process_gene,
                                                [(group1, results, valid_types) for group1 in groups]):
                scores_dict[group_scores[0]] = group_scores[1]

        adata.uns['rank_genes_groups_summary_tumorall'] = scores_dict
    else:
        adata.uns['rank_genes_groups_summary_tumorall'] = np.load(
            all_path + '_tumorall.npy', allow_pickle='TRUE').item()

    regions = ['tumorall']
    # regions = ['normalall']

    print("Loading Regions")
    for region in regions:
        if load_regions:
            adata.uns[f'rank_genes_groups_{region}'] = np.load(all_path + f'_{region}.npy', allow_pickle=True).item()
            continue

        regioner = {cell_type: {gene: np.mean(scores) for gene, scores in genes_dict.items()}
                        for cell_type, genes_dict in adata.uns[f'rank_genes_groups_summary_{region}'].items()}
        regioner = {cell_type: sorted(genes.items(), key=lambda k: k[1], reverse=True) for cell_type, genes in regioner.items()}

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

        rec_region = {'params': {'groupby': 'type_tissue', 'reference': region, 'method': 'wilcoxon', 'use_raw': False, 'layer': None, 'corr_method': 'benjamini-hochberg'}}
        rec_region['names'] = rank_names
        rec_region['scores'] = rank_scores

        adata.uns[f'rank_genes_groups_{region}'] = rec_region
        np.save(backup_dir + f'{name}_{region}.npy', adata.uns[f'rank_genes_groups_{region}'])
            
    from gprofiler import GProfiler
    import gseapy
    import os
    gp = GProfiler(return_dataframe=True, user_agent='INMEGEN')

    if not os.path.exists('h.all.v2023.2.Hs.symbols.gmt'):
        import subprocess
        subprocess.run(["wget", "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt"])
    
    gsea_folder = '/data'
    combined_dfs = {}

    for region in regions:
        gsea_path = f'{gsea_folder}/heatmap_{region}_{time}.csv'
        if load_gsea_heatmap:
            combined_dfs[region] = pd.read_csv(gsea_path, index_col=0)
        else:
            types = adata.uns[f'rank_genes_groups_{region}']['scores'].dtype.names
            combined_dfs[region] = get_gseas_df(adata, valid_types, types, id_, load_gsea=False, key=f'rank_genes_groups_{region}')
            combined_dfs[region].to_csv(gsea_path)
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(combined_dfs[region], cmap='viridis')
        plt.title(f'Hallmarks Scores by Cell Type for {region}')
        plt.xlabel('Hallmarks')
        plt.ylabel('Cell Types')
        plt.savefig(f'{gsea_folder}/heatmap_{region}_{time}.png')

    return None

# Main entry point
@app.local_entrypoint()
def main():
    # Run the remote function
    print("Starting differential expression analysis on the remote worker...")
    result = get_de.remote()
