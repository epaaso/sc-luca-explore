import os
from typing import List, Optional
import logging

from matplotlib.axes import Axes
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import gseapy

logging.basicConfig(level=logging.INFO)


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
                            in enumerate(ranks['names'][typer]) if ranks['scores'][typer][i] != 0.5}
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


def load_de_region(region_file):
    """Loads a de_region object from file, or returns None if not found."""
    if os.path.exists(region_file):
        logging.info(f"Loading {region_file}")
        return np.load(region_file, allow_pickle=True).item()
    else:
        logging.warning(f"File not found: {region_file}")
        return None


def average_de_regions(region_files):
    """
    Loads de_region objects from the list of file paths and computes the 
    average gene score per cell type across all datasets.
    
    This function does not assume that all datasets have the same set of cell types 
    or that gene order is identical. For each cell type, it creates the union of genes 
    encountered, and for each gene computes the mean of the scores from the datasets
    that contain that gene.
    
    Returns a new de_region object with:
       - "scores": a recarray where each field is the average score array for a cell type.
       - "names": a recarray where each field is the corresponding gene names array.
       - "params": taken from the first loaded file.
    """
    loaded = [load_de_region(f) for f in region_files]
    de_regions_list = [d for d in loaded if d is not None]
    if not de_regions_list:
        raise ValueError("No valid de_region files loaded.")
        
    # Create a union of all cell types encountered
    all_cell_types = set()
    for d in de_regions_list:
        # Each file stores scores as a recarray; extend with its field names.
        all_cell_types.update(d["scores"].dtype.names)
    all_cell_types = sorted(all_cell_types)  # sort for consistency

    averaged_scores = {}
    averaged_names = {}
    union_genes = set()
    dataset_gene_maps = {}
    for ct in all_cell_types:
        # Filter datasets that include this cell type.
        candidate_datasets = [d for d in de_regions_list if ct in d["scores"].dtype.names]
        # Build the union of genes and per-dataset mapping.
        dataset_gene_maps[ct] = []
        for d in candidate_datasets:
            # Convert names and scores to numpy arrays.
            names_arr = np.array(d["names"][ct])
            scores_arr = np.array(d["scores"][ct])
            # Build a mapping: gene -> score.
            gene_map = {str(gene): score for gene, score in zip(names_arr, scores_arr)}
            union_genes = union_genes.union(set(gene_map.keys()))
            dataset_gene_maps[ct].append(gene_map)
        
    union_genes = union_genes - {''}  # remove empty gene names    
    union_genes = sorted(union_genes)

    for ct in all_cell_types:
        # For each gene in the union, get its score from each dataset (default=0.5).
        avg_scores = []
        for gene in union_genes:
            gene_scores = [gene_map[gene] for gene_map in dataset_gene_maps[ct] if gene in gene_map]
            if len(gene_scores) < 1:
                gene_scores = [0.5]
            avg_scores.append(np.mean(gene_scores))

        order = np.argsort(-np.array(avg_scores))
        avg_scores = np.array(avg_scores)[order]
        sorted_genes = np.array(union_genes)[order]

        averaged_names[ct] = np.array(sorted_genes)
        averaged_scores[ct] = np.array(avg_scores)

    # Convert averaged_scores and averaged_names dictionaries to recarrays.
    # Use a one-element recarray with each field as type object ('O').
    dtype_scores = [(ct, float) for ct in all_cell_types]
    dtype_names  = [(ct, 'O') for ct in all_cell_types]

    arr_scores, arr_names = [], []
    for ct in all_cell_types:
        arr_names.append(averaged_names[ct])
        arr_scores.append(averaged_scores[ct])

    scores_rec = np.core.records.fromarrays(arr_scores,
                                              dtype=dtype_scores)
    names_rec = np.core.records.fromarrays(arr_names,
                                             dtype=dtype_names)
    
    
    # Use the params from the first loaded de_region (adjust if needed)
    params = de_regions_list[0]["params"]
    
    averaged_de_region = {
        "scores": scores_rec,
        "names": names_rec,
        "params": params
    }
    return averaged_de_region


def plot_marker_genes(de_region, out_file, n_genes=20):
    """
    Plots marker genes based on the de_region object.
    
    For each cell type, it selects the top n_genes (sorted by score) and plots a bar plot.
    """
    logging.info("Plotting marker genes")
    cell_types = list(de_region["scores"].dtype.names)
    num_types = len(cell_types)
    
    fig, axs = plt.subplots(
        (num_types + 1) // 2, 2, figsize=(16, 4.5 * ((num_types + 1) // 2))
    )
    axs = axs.ravel()
    n_genes = 20
    for i, cell_type in enumerate(cell_types):
        cond_plot(
        de_region, [cell_type], cell_types, n_genes=n_genes,
        ax=axs[i], sharey=False, key="rank_genes_groups_tumorall",
        show=False, fontsize=6, titlesize=9, gene_mapping=None
        )
    # Remove any extra axes
    if len(axs) > num_types:
        for j in range(num_types, len(axs)):
            fig.delaxes(axs[j])
    
    plt.savefig(out_file, bbox_inches="tight")
    logging.info(f"Marker genes plot saved to {out_file}")
    plt.close()


def plot_gsea_heatmap(de_region):
    """
    Creates a heatmap using the de_region gene scores.
    
    Here we build a DataFrame with cell types as rows and a common set of genes as columns.
    """
    logging.info("Plotting GSEA heatmap")
    cell_types = list(de_region["scores"].dtype.names)
    
    gsea_folder = os.path.join(wilcox_path,'..', "gseapy_gsea")
    os.makedirs(gsea_folder, exist_ok=True)

    region = "tumorall"
    gsea_path = os.path.join(
        gsea_folder, f"heatmap_averaged_{region}_{time}.csv"
    )

    types = de_region["scores"].dtype.names
    combined_dfs = {region: get_gseas_df(
        de_region, cell_types, types,
        "averaged", load_gsea=False,
        gsea_folder=gsea_folder, gene_mapping=None, num_threads=10
    )}
    combined_dfs[region].to_csv(gsea_path)

    
    plt.figure(figsize=(15, 10))
    sns.heatmap(combined_dfs[region], cmap="viridis")
    plt.title(f"Hallmarks Scores by Cell Type for {region}")
    plt.xlabel("Hallmarks")
    plt.ylabel("Cell Types")
    out_gsea = os.path.join(
        gsea_folder, f"heatmap_averaged_{region}_{time}.png"
    )
    plt.savefig(out_gsea, bbox_inches="tight")
    logging.info(f"GSEA plot saved to {out_gsea}")


def pair_to_auc_summary(time, ext_name, wilcox_path, load_summary=False):
    os.chdir('/root/host_home/luca')
    import sys
    sys.path.append('/root/host_home/luca')
    from itertools import takewhile
    from nb_DE_wilcox.modal_DE import DEProcessor, DEConfig

    config = DEConfig()
    config.processor.regions_AUC = True
    config.processor.parallel_summary = True
    config.processor.num_processes = 8
    config.common.ext_name = ext_name
    config.common.time = time
    config.__post_init__()  # Ensure defaults are applied.
    processor = DEProcessor(config.common, config.processor, config.dataloader)
    
    if not load_summary:
        print('Loading de_pair')
        de_pair = np.load(f'{wilcox_path}/{time}_{ext_name}_pair.npy', allow_pickle=True).item()
        print('Loaded de_pair')

        tumor_types = tumor_types = [
                g for g in de_pair.keys()
                if not any(x in g for x in ['Tumor', 'Ciliated', 'Alveolar', 'epithelial', 'Club'])
            ]
        prefix = tumor_types[0][:5]
        tumor_types = list(takewhile(lambda t: t[:5] == prefix, tumor_types))
        tumor_types = [t.split('_vs_')[1] for t in tumor_types]

        de_summary = processor.compute_summary(adata=None, de_pair=de_pair, tumor_types=tumor_types, valid_types=tumor_types)
    else:
        de_summary = np.load(f'{wilcox_path}/{time}_{ext_name}_summary_normalall.npy', allow_pickle=True).item()

    # Filter the de_summary to include only the 'B cell' type FOR TESTING
    # de_summary = {key: de_summary[key] for key in ['B cell']}    

    de_regions = processor.compute_regions(adata=None, de_summary=de_summary)
    return de_regions


def aggregate_count_matrix(region_files, nan_gene_frac=0.5):
    all_genes = set()
    row_data = {}   # row key -> { gene -> score }


    for i, region_file in enumerate(region_files):
        if not os.path.exists(region_file):
            continue

        # Infer dataset name from filename or path
        dataset_id = '_'.join(dss[i].split('_')[:2])

        # Load the de_region object
        de_region = np.load(region_file, allow_pickle=True).item()
        scores_rec = de_region["scores"]
        names_rec = de_region["names"]

        # For each cell_type in the recarray fields
        for cell_type in scores_rec.dtype.names:
            # Combine cell_type and dataset for row label
            row_label = f"{cell_type}_{dataset_id}"

            # Scores and gene names for this cell_type
            cell_scores = scores_rec[cell_type]
            cell_genes = names_rec[cell_type]

            # Convert to dictionary: gene -> score
            gene_map = dict(zip(cell_genes, cell_scores))
            row_data[row_label] = gene_map
            all_genes.update(cell_genes)

    # Build a DataFrame with rows = row_data keys, columns = sorted list of all_genes
    all_genes = sorted(all_genes)
    df = pd.DataFrame(index=row_data.keys(), columns=all_genes, dtype=float)

    # Fill in the DataFrame
    for row_label, gene_map in row_data.items():
        for gene, score in gene_map.items():
            df.at[row_label, gene] = score

    df = df.loc[:, df.isna().mean() < nan_gene_frac]
    return df

if __name__ == "__main__":

    import h5py
    from anndata.experimental import read_elem

    wilcox_path = "/root/host_home/luca/nb_DE_wilcox/wilcoxon_DE"
    time = 'III-IV'
    average_now = False
    output_average = os.path.join(wilcox_path, f"{time}_averaged_tumorall.npy")

    if average_now:
        # Get atlas dsets
        file_obj = h5py.File('/root/datos/maestria/netopaas/luca/data/atlas/extended_tumor_hvg.h5ad', 'r')
        obs_matrix = read_elem(file_obj['obs'])
        dss = list(obs_matrix['dataset'].unique())
        file_obj.close()

        # List of de_region files from various datasets (adjust these paths as needed)
        dss.extend(['Trinks_Bishoff_2021_NSCLC', 'Deng_Liu_LUAD_2024', 'Zuani_2024_NSCLC', 'Hu_Zhang_2023_NSCLC',
                    'extended'
                    ])
        region_files = [ os.path.join(wilcox_path,
                        f"{time}_{ds}_tumorall.npy")
            for ds in dss
        ]

        ### Convert some de_pair to AUC summary #######
        de_regions = pair_to_auc_summary(f'{time}', 'extended', wilcox_path, load_summary=True)

        ##### Aggregate all summary files to a count matrix for cellphonedb ###
        count_matrix = aggregate_count_matrix(region_files)

        count_matrix.to_csv(os.path.join(wilcox_path, f"auc_count_cellphonedb_{time}.csv"))
        
        # # Compute the averaged de_region object
        # averaged_de_region = average_de_regions(region_files)
        # np.save(output_average, averaged_de_region, allow_pickle=True)
        # logging.info(f"Averaged de_region saved to {output_average}")
    else:
        averaged_de_region = np.load(output_average, allow_pickle=True).item()
        logging.info(f"Averaged de_region loaded from {output_average}")
    
    # Plot marker genes using the new averaged de_region
    # marker_plot_file = os.path.join(os.path.dirname(output_average), f"{time}_averaged_markergenes.png")
    # plot_marker_genes(averaged_de_region, marker_plot_file)
    
    # Plot a GSEA heatmap based on the averaged de_region scores
    # plot_gsea_heatmap(averaged_de_region)