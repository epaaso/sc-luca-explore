import warnings

from typing import Optional, List, Union, Literal, Tuple, Dict, Iterable

import urllib.request
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import multiprocessing

import numpy as np

import scipy as sp
from scipy import stats
from scipy.stats import linregress

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import GEOparse

import networkx as nx

import anndata as ad
from scanpy.tl import rank_genes_groups
from scanpy.pl import rank_genes_groups_violin


def remove_repeated_var_inds(adata_tmp):
    '''
    Remove the cols with repeated indices of an AnnData object
    '''
    adata_tmp.var['i'] = adata_tmp.var.index
    adata_tmp.var.drop_duplicates(subset=['i'], inplace=True)
    del adata_tmp.var['i']
    
    return adata_tmp


def join_map_mart(adata_tmp, annot, gene_annot='external_gene_name', how='left'):
    '''
    Takes the index of the first arg AnnData and reduces the df
    in the right to those var indices
    '''
    maps_gene_names = pd.merge(
        pd.DataFrame({gene_annot:adata_tmp.var.index}),
        annot,
        how=how, on=gene_annot, suffixes=('_xxx','_yyy'))
    maps_gene_names.rename(columns={'external_gene_name':'gene_symbol'}, inplace=True)
    maps_gene_names.drop_duplicates(subset=['gene_symbol'], inplace=True)
    maps_gene_names.set_index('gene_symbol', inplace=True)
    return maps_gene_names


def search_str_index2(s_list:list, search_str:str) -> list:
    """
    Get the indexes for in a list that contain a str
    
    Parameters:
        s_list (list): the list to search in
        search_str (str): the str to search for, it only must be contained

    Returns:
        matched_indexes (list): A list of the indexes where it was found
    """
    
    matched_indexes = []
    i = 0
    length = len(s_list)

    while i < length:
        if type(s_list[i]) is not str:
            i += 1
            continue
        if search_str in s_list[i]:
            matched_indexes.append(i)
        i += 1
        
    return matched_indexes


def search_str_index(s_list:list, regex_pattern:str) -> list:
    """
    Get the indexes in a list that match a `regex_pattern`
    
    Parameters:
        s_list (list): the list to search in
        search_str (str): the str to search for, it only must be contained

    Returns:
        matched_indexes (list): A list of the indexes where it was found
    """
    
    matched_indexes = []
    i = 0
    length = len(s_list)

    while i < length:
        if type(s_list[i]) is not str:
            i += 1
            continue
        if re.search(regex_pattern, s_list[i]):
            matched_indexes.append(i)
        i += 1
        
    return matched_indexes

def get_geo_exprs(gse_str='', data_dir='/root/datos/maestria/netopaas/lung_scRNA'):
    """
    Builds a metada matrix from geo SOFT file for scRNA-seq data that
    doesn't have the expression matrix attached. A csv is saved in
    f'{data_dir}/{gse_str}/{gse_str}_metadata.csv'
    
    Parameters:
        gse_str (str): The string of the GSE to be gotten
        data_dir (str): Where the CSV is to be saved

    Returns:
        metadata (dict): A dict with first level the features, second the sample
    
    """
    gse = GEOparse.get_GEO(geo=gse_str, destdir=f"{data_dir}/{gse_str}/", silent=True)

    # Get expression data and metadata matrices
    exprs = []
    gsmNames = []
    metadata = {}
    sup_dict = {}
    for gsm_name, gsm in gse.gsms.items():
        if gsm.metadata['type'][0]=='SRA':
             # Expression data
            # print(gsm.__dict__)
            if len(gsm.table)>0:
                # TODO will there really be a table here anytime?
                # if so run code here
                pass
            else:
                # Get the supplementary files with their type because no table is attached
                sup_file_url = gsm.metadata['supplementary_file_1'][0]
                
                # TODO it is in this array but no standard index, search for it later
                l1 = gsm.metadata['data_processing']
                s = r'upplementary'
                matched_indexes = search_str_index(l1, s)
                if matched_indexes:
                    sup_file_type = l1[matched_indexes[0]].split(':')[1]
                else:
                    warnings.warn('')
                    sup_file_type = 'Missing'
                
                s = r'[Gg]enome[_ ][Bb]uild'
                matched_indexes = search_str_index(l1, s)
                genome_build = l1[matched_indexes[0]].split(':')[1]
                
                if not 'sup_file_url' in sup_dict:
                    sup_dict['sup_file_url'] = {}
                sup_dict['sup_file_url'][gsm_name] = sup_file_url
                
                if not 'sup_file_type' in sup_dict:
                    sup_dict['sup_file_type'] = {}
                sup_dict['sup_file_type'][gsm_name] = sup_file_type
                
                if not 'genome_build' in sup_dict:
                    sup_dict['genome_build'] = {}
                sup_dict['genome_build'][gsm_name] = genome_build
                
                
                # print('No expression table, saving supplementary file url'
                #              f'{sup_file_url} with type: {sup_file_type}')
            if hasattr(gsm, 'SRA'):
                warnings.warn("There is an SRArun access table, consider using "
                              "your snakemake workflow to parallely download them")
                
    # Metadata
            for key,value in gsm.metadata.items():
                if (key=='characteristics_ch1' or key=='characteristics_ch2') and (len([i for i in value if i!=''])>1 or value[0].find(': ')!=-1):
                    tmpVal = 0
                    for tmp in value:
                        splitUp = [i.strip() for i in tmp.split(':')]
                        if len(splitUp)==2:
                            if not splitUp[0] in metadata:
                                metadata[splitUp[0]] = {}
                            metadata[splitUp[0]][gsm_name] = splitUp[1]
                        else:
                            if not key in metadata:
                                metadata[key] = {}
                            metadata[key][gsm_name] = splitUp[0]
                else:
                    if not key in metadata:
                        metadata[key] = {}
                    if len(value)==1:
                        metadata[key][gsm_name] = ' '.join([j.replace(',',' ') for j in value])
            ftp_name = sup_dict['sup_file_url'][gsm_name].split('/')[-1]
            
            key = 'sup_type'
            if not key in metadata:
                metadata[key] = {}
            metadata[key][gsm_name] = sup_dict['sup_file_type'][gsm_name]
            
            key = 'local_path'
            if not key in metadata:
                metadata[key] = {}
            metadata[key][gsm_name] = f'{data_dir}/{gse_str}/{ftp_name}'
            
            key = 'genome_build'
            if not key in metadata:
                metadata[key] = {}
            metadata[key][gsm_name] = sup_dict['genome_build'][gsm_name]
            
            metadata['local_path'][gsm_name] = f'{data_dir}/{gse_str}/{ftp_name}'
    pd.DataFrame(metadata).to_csv(f'{data_dir}/{gse_str}/{gse_str}_metadata.csv')

    return metadata

def download_url(args, data_dir='/root/datos/maestria/netopaas/lung_scRNA'):
    t0 = time.time()
    #Extract url and file path from args
    url = args[0]
    path = args[1]
    try:
        # For getting ftp that does not need 
        gsm_path, response = urllib.request.urlretrieve(url,
                                      path)
        return(url, time.time() - t0)
    except Exception as e:
        print('Exception in download_url():', e)


def download_parallel(args):
    """
    Downloads the zipped array of urls in 1st pos with paths in 2nd pos
    
    """
    cpus = int(cpu_count()/3)
    print("CPUS: ", cpus)
    
    with ThreadPool(cpus -1 ) as pool:
        for result in pool.imap_unordered(download_url, args):
            print('url:', result[0], 'time (s):', result[1])


def compare_groups(adata: ad.AnnData, groupby: str, group1: str, group2: str,
                   method:str='wilcoxon', use_raw:bool=False, parallel:bool=False):

    key = f'{group1}_vs_{group2}'
    if parallel:
        print(f'Started copying {key}')
    else:
        print(f'Comparing {key}')        
    adata_temp = adata.copy() if parallel else adata # Make a copy to avoid modifying the shared adata
    if parallel:
        print(f'Ended copying {key}')
        
    rank_genes_groups(adata_temp, groupby=groupby, groups=[group1], reference=group2,
                      method=method, use_raw=use_raw, key_added=key)

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
                               use_raw: Optional[bool] = None,
                               method: Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']] = 'wilcoxon',
                               parallel: bool = False,
                               n_jobs: int = 2):
    """
    Perform pairwise comparison of marker genes between specified groups. Expects log data.

    Parameters:
    - adata: AnnData object containing the data.
    - groupby: The key for the observations grouping to consider.
    - groups: List of groups to include in pairwise comparisons.
    - method: The statistical method to use for the test ('t-test', 'wilcoxon', etc.).
    - n_jobs: Number of jobs to run in parallel. -1 means using all processors.

    Returns:
    - Returns a dict for all pairwise groups and its statistics
    """

    pairwise_results = {}
    summary_stats = {}
    results = []

    comparisons = [(group1, group2) for group1 in groups for group2 in groups if group1 != group2]
    print(comparisons)

    if parallel:
        with multiprocessing.Pool(n_jobs) as pool:
            results = pool.starmap(compare_groups, [(adata, groupby, group1, group2, method, use_raw) for group1, group2 in comparisons])
    else:
        for comparison in comparisons:
            group1, group2 = comparison
            results.append(compare_groups(adata, groupby, group1, group2, method, use_raw, parallel))

    for key, result in results:
        pairwise_results[key] = result

    return pairwise_results


# def 

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

def violin_genes_pval(adata: ad.AnnData, cluster: Union[str, List[str]], genes: List[str],
                      key: str = 'rank_genes_groups', p_threshold: float = 0.05,
                      gene_symbols: str = 'feature_name') -> None:
    """
    Plots a violin plot for the specified genes and annotates the plots with p-values
    indicating the statistical significance of the difference in gene expression
    across the specified cluster groups. The function requires that
    `sc.tl.rank_genes_groups` has already been run on the AnnData object with the
    specified key.

    Parameters:
    adata (anndata.AnnData): The annotated data matrix.
    cluster (str or List[str]): Cluster or list of clusters for which to plot the violin plots.
    genes (List[str]): List of gene names for which to plot the violin plots.
    key (str, optional): Key under which to look for the rank_genes_groups results in adata.uns.
    p_threshold (float, optional): Threshold for statistical significance.
    gene_symbols (str, optional): Column in adata.var DataFrame that stores gene symbols.

    Returns:
    None: This function does not return any value.
    """
    
    # Get the gene indices that match the gene symbols
    ens_genes = adata.var.index[adata.var[gene_symbols].isin(genes)]

    # Create the violin plot
    ax = rank_genes_groups_violin(adata, split=False, use_raw=False, show=False,
                                        groups=cluster, gene_symbols=gene_symbols,
                                        gene_names=genes, key=key)

    # Build a dictionary of gene names to their p-values
    significance_dict = {gene: pval for gene, pval in 
                         zip(adata.uns[key]['names'][cluster],
                             adata.uns[key]['pvals'][cluster])
                         if gene in ens_genes}
    
    # Annotate the violin plot with p-values
    i = 0
    for gene, pval in significance_dict.items():
        color = 'red' if pval < p_threshold else 'black'
        pval = round(pval,3)
        ylim = ax[0].get_ylim()[1]
        yrange = ylim - ax[0].get_ylim()[0]
        
        # Annotate the plot with the p-value for each gene
        ax[0].text(i, ylim - yrange*0.05, f'{pval}', ha='center', va='bottom', color=color)
        i += 1
    
    # Show the plot
    plt.show()


def sparse_column_quantiles(csr, col_index, quantiles=[0.25, 0.5, 0.75]):
    """
    Calculate quantiles for a column in a CSR matrix more memory efficiently,
    considering the distribution of zeros without concatenating a large array of zeros.
    
    :param csr: CSR matrix
    :param col_index: Index of the column for which to calculate quantiles
    :param quantiles: List of quantiles to calculate (values between 0 and 1)
    :return: Quantile values for the specified column
    """
    # Extract non-zero values from the column and sort them
    non_zero_values = np.sort(csr[:, col_index].data)
    
    # Total number of elements (including zeros)
    total_elements = csr.shape[0]
    
    # Number of zero values
    n_zeros = total_elements - len(non_zero_values)
    
    # Placeholder for quantile results
    quantile_results = []
    
    for q in quantiles:
        # Calculate the position of the quantile in the sorted array
        quantile_pos = q * (total_elements - 1)
        
        # Determine the value of the quantile
        if quantile_pos < n_zeros:
            # Quantile falls within the range of zeros
            quantile_value = 0
        else:
            # Adjust the position for the array of non-zeros and find the corresponding quantile value
            adjusted_pos = int(quantile_pos - n_zeros)
            quantile_value = non_zero_values[min(adjusted_pos, len(non_zero_values)-1)]
        
        quantile_results.append(quantile_value)
    
    return quantile_results

def centers_errs_from_grouped(adata_filtered: ad.AnnData, 
                              group_key: str = 'group_compare', 
                              center_stat: Literal['median', 'mean'] = 'mean') -> Tuple[Dict[str, List[float]], np.ndarray]:
    """
    Calculate the central tendency (mean or median) and interquartile range errors for each feature across groups in an AnnData object.
    We focus on non-zero values, because of sparsity.

    Parameters:
    ----------
    adata_filtered : AnnData
        The filtered AnnData object containing the dataset to analyze.
    group_key : str, optional
        The key in `adata_filtered.obs` to use for grouping the data, by default 'group_compare'.
    center_stat : {'median', 'mean'}, optional
        The statistic to use for calculating the center of the data. Can be either 'mean' or 'median', by default 'mean'.

    Returns:
    -------
    Tuple[Dict[str, List[float]], np.ndarray]
        A tuple containing:
        - A dictionary mapping each group to a list of center statistics (mean or median) for each feature.
        - An array of shape (#groups, 2, #features) containing the lower and upper interquartile range errors for each feature in each group.
    """
    # Check if the data matrix is sparse
    sparse = isinstance(adata_filtered.X, sp.sparse.spmatrix)
    
    # Group the data
    grouped = adata_filtered.obs.groupby(group_key)
    
    center_data = {}
    yerrs = []
    for group, indices in grouped.indices.items():
        group_data = adata_filtered[indices].X
        
        centers = []
        lowers = []
        uppers = []
        for i in range(group_data.shape[1]):  # Iterate over features
            x = group_data[:, i]
            if sparse:
                x = x.data if x.data.size > 0 else np.array([0])
            else:
                x = x[x != 0]  # Focus on non-zero values for dense arrays
            
            # Calculate center statistic
            center = np.mean(x) if center_stat == 'mean' else np.median(x)
            centers.append(center)
            
            # Calculate interquartile range
            quantile_values = np.quantile(x, [0.25, 0.75])
            lower = quantile_values[0]
            upper = quantile_values[1]
            lowers.append(max(center - lower, 0))
            uppers.append(max(upper - center, 0))
        
        center_data[group] = centers
        group_err = np.stack([np.array(lowers), np.array(uppers)])
        yerrs.append(group_err)
    
    yerrs = np.stack(yerrs)
    return center_data, yerrs


def rank_genes_groups_bar(adata: ad.AnnData, group_key: str = None,
                          group_names: List[str] = [], gene_list: List[str] = [],
                          rest: bool = True, show: bool = True,
                          figsize: tuple=(10, 6), center_stat: Literal['median', 'mean']= 'mean',
                          gene_symbols: str = None) -> plt.Axes:
    """
    Plots a bar chart comparing the mean gene expression levels across specified groups with error bars representing interquartile ranges.

    Parameters:
    - adata: AnnData object containing the dataset. The counts can be in sparse mode
    - group_key: Key in `adata.obs` that corresponds to the grouping (e.g., 'cell_type').
    - group_names: List of group names to include in the comparison. If empty, compares all groups found under `group_key`.
    - gene_list: List of genes to include in the plot. If empty, includes all genes.
    - rest: If True, all other groups not specified in `group_names` are combined into a single "Rest" group.
    - show: If True, displays the plot. Otherwise, the plot is not displayed but returned for further customization.
    - gene_symbols: str of the column to get the gene symbols from. If None index is used

    Returns:
    - A matplotlib Axes object containing the plot.
    """

    sparse = True if sp.sparse.issparse(adata.X) else False  
    # Step 1: Filter the AnnData object to only include the genes of interest
    if gene_list:
        adata_filtered = adata[:, adata.var_names.isin(gene_list)]
        li1 = gene_list
        li2 = adata_filtered.var.index
        s = set(li2)
        temp3 = [x for x in li1 if x not in s]
        if temp3:
            raise ValueError(f'{temp3} not in adata features')
            
    else:
        adata_filtered = adata

    

    adata_filtered.obs['group_compare'] = 'Rest' if rest else None
    if group_names:
        # Build a grouping key for just the two groups we want
        for group_name in group_names:
            adata_filtered.obs.loc[adata_filtered.obs[group_key] == group_name, 'group_compare'] = group_name
        
    adata_filtered.obs['group_compare'] = adata_filtered.obs['group_compare'].astype('category')
    
    center_data, yerrs = centers_errs_from_grouped(adata_filtered, group_key='group_compare')
    
    # Step 2: Prepare the data for plotting
    # Convert mean and IQR data into DataFrames for easier plotting
    if gene_symbols:
        index = adata_filtered.var[gene_symbols]
    else:
        index = adata_filtered.var_names
    mean_df = pd.DataFrame(center_data, index=index)
    
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=figsize)
    
    # Step 3: Plot the stacked bar plot with IQR as error bars
    mean_df.plot(kind='bar', yerr=yerrs, capsize=4, ax=ax)
    ax.set_title('Non-Zero Gene Expression by Group with IQR')
    ax.set_ylabel(f'{center_stat.capitalize()} of Non-zero Expression Values')
    ax.set_xlabel('Gene')
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend(title=group_key)

    if show:
        plt.show()

    return ax

def bar_genes_pval(adata: ad.AnnData, group_names: List[str]=None,
                   gene_list: List[str]=None, group_key:str=None,
                   key: str = 'rank_genes_groups', p_threshold: float = 0.05,
                   rest: bool=True, gene_symbols: str = None, show:bool=True) -> None:
    """
    Plots a bar plot for the specified genes and annotates the plots with p-values
    indicating the statistical significance of the difference in gene expression
    across the specified groups. The function requires that
    `sc.tl.rank_genes_groups` has already been run on the AnnData object with the
    specified key.

    Parameters:
    - adata (anndata.AnnData): The annotated data matrix.
    - group_names (List[str]): List of groups for which to plot the violin plots.
    - gene_list (List[str]): List of gene names for which to plot the violin plots.
    - key (str, optional): Key under which to look for the rank_genes_groups results in adata.uns.
    - p_threshold (float, optional): Threshold for statistical significance.
    - rest (boolean, optional): Compare against the rest?
    - gene_symbols: str of the column to get the gene symbols from. If None index is used
    - show (boolean, optional): Wether to show the plot or save it for later

    Returns:
    - A matplotlib Axes object containing the plot.
    """
    try:
        adata.uns[key]
    except KeyError:
        raise(f"The function requires that `sc.tl.rank_genes_groups` has already been run in {key} key")
    
    # Create the violin plot
    ax = rank_genes_groups_bar(adata, group_key, group_names=group_names, gene_list=gene_list,
                               rest=rest, gene_symbols=gene_symbols, show=False)

    dupl_genes = []
    for gene in gene_list:
        if (adata.var.index == gene).sum() > 1:
            dupl_genes.append(gene)
    if dupl_genes:
        raise ValueError(f"The genes {dupl_genes} are repeated the plot won't display correctly. Please handle duplicates")

    # Build a dictionary of gene names to their p-values
    # TODO for multiple groups
    significance_dict = {gene: pval for gene, pval in 
                          zip(adata.uns[key]['names'][group_names[0]],
                             adata.uns[key]['pvals'][group_names[0]])
                         if gene in gene_list}
    
    # Annotate the violin plot with p-values
    i = 0
    # Variables to help adjust the bracket and text position
    y_bracket_height = 0.2  # Height of the bracket above the highest bar
    arm_length_fraction = 0.02  # Fraction of the distance between bars to use as arm length
    
    for gene, pval in significance_dict.items():
        color = 'red' if pval < p_threshold else 'black'
        significance_text = "*" if pval < p_threshold else "ns"
        
        # Get the x-axis positions of the left and right bars or error bars for the current gene comparison
        xlimLeft = ax.containers[0].lines[1][0].get_data()[0][i]
        xlimRight = ax.containers[2].lines[1][0].get_data()[0][i]
        
        
        # Calculate the dynamic arm length based on the x-axis limits
        arm_length = (ax.get_ylim()[1]) * arm_length_fraction
        y_text_offset = arm_length/2
        
        # Calculate the maximum y-limit for the bracket based on the bars or error bars
        ylim1 = ax.containers[0].lines[1][1].get_data()[1][i]
        ylim2 = ax.containers[2].lines[1][1].get_data()[1][i]
        ylim = max(ylim1, ylim2) + y_text_offset + arm_length  # Adjust for bracket height
        
        # Coordinates for the main bracket line
        bracket_coords = [[xlimLeft, ylim], [xlimRight, ylim]]
        
        # Coordinates for the arms at each end of the bracket
        left_arm = [[xlimLeft, ylim - arm_length], [xlimLeft, ylim ]]
        right_arm = [[xlimRight, ylim - arm_length], [xlimRight, ylim]]
        
        # Draw the main bracket line
        ax.plot(*zip(*bracket_coords), lw=1, c='black')
        
        # Draw the arms at each end of the bracket
        ax.plot(*zip(*left_arm), lw=1, c='black')
        ax.plot(*zip(*right_arm), lw=1, c='black')
        
        # Annotate the plot with the significance indicator above the bracket
        ax.text((xlimLeft + xlimRight) / 2, ylim + y_text_offset, significance_text, ha='center', va='bottom', color=color)
        
        i += 1  # Increment the position for the next gene
    
    # Show the plot
    if show:
        plt.show()

    return ax


def functional_network_plot(G: nx.Graph, node_pie_data: Dict[str, Dict[str, float]],
                            num_cells: Dict[str, int] = {}, weight_key:str = 'MI', max_width: float = 8,
                            max_radius: float = 0.05, label_fontsize: float = 6,
                            legend_fontsize: float = 7, spring_force: float = 1,
                            scale: float = 1,
                            figsize: Tuple[int, int] = (12, 8), show: bool = True) -> plt.Axes:
    """
    Draws a network graph with nodes represented by pie charts indicating various functions,
    and edges thickness based on a specified metric (e.g., Mutual Information, MI).

    Parameters:
    - G (nx.Graph): A NetworkX graph object.
    - node_pie_data (Dict[str, Dict[str, float]]): A dictionary mapping node identifiers to 
      another dictionary of function names and their corresponding values.
    - num_cells (Dict[str, int], optional): A dictionary mapping function names to cell counts.
    - weight_key (str, optional): The name of the key for the edge weights
    - max_width (float, optional): Maximum width for graph edges.
    - max_radius (float, optional): Maximum radius for node pie charts.
    - label_fontsize (float, optional): Font size for node labels.
    - legend_fontsize (float, optional): Font size for the legend.
    - spring_force (float, optional): The spring force parameter for the network layout.
    - scale (float, optional): Scale factor for positions. If scale is None, no rescaling is performed.
    - figsize (Tuple[int, int], optional): Figure size for the plot.
    - show (bool, optional): If True, display the plot. Otherwise, the plot is not shown.

    Returns:
    matplotlib.axes.Axes: The Axes object containing the plot.
    """
    # Assuming you have a dynamic number of functions, determine the colormap
    cmap = plt.cm.tab20  # Choose a colormap
    cmap2 = plt.cm.tab20b
    # cmap.extend(plt.cm.tab20b)
    all_functions = set(func for data in node_pie_data.values() for func in data.keys())
    num_functions = len(all_functions)
    
    # Generate a color for each function by evenly spacing across the colormap
    colors = cmap(np.linspace(0, 1, num_functions//2))
    add = num_functions % 2
    colors2 = cmap2(np.linspace(0, 1, num_functions//2 + add))
    colors = np.vstack([colors, colors2])
    
    # Check if we have no expressed values and put it in the end of the functs if they
    no_expr = [x for x in all_functions if x in ['NO UNDEREXPRESSION','NO OVEREXPRESSION']]
    no_expr  = None if len(no_expr) == 0 else no_expr[0]
    if no_expr:
        all_functions.remove(no_expr)
        sorted_functions = sorted(all_functions).append(no_expr)
    else:
        sorted_functions = sorted(all_functions)
        
    
    # Create a dictionary to map each function to its color        
    function_to_color = {func: color for func, color in zip(sorted(all_functions), colors)}
    if no_expr:
        function_to_color[no_expr] = plt.cm.gray(0.0)

    # Normalize the cells nums to a respectable radius
    function_to_radius = {}
    if len(num_cells) > 0:
        max_cells = max(num_cells.values())
        function_to_radius = {func: max_radius*num/max_cells 
                              for func, num in num_cells.items()}
        
    
    fig, ax = plt.subplots(figsize=figsize)  # Example size, adjust as needed
    # Draw the network
    pos = nx.spring_layout(G, k=spring_force, scale=scale)
    # Get edge weights and scale them as desired for visualization
    edge_weights = [max_width*float(G[u][v][weight_key]) for u, v in G.edges()]

    # Draw edges with thickness based on the 'weight' attribute
    nx.draw_networkx_edges(G, pos, width=edge_weights, ax=ax)
    
    # For each node, draw the pie chart with colors mapped by function
    for node, data in node_pie_data.items():
        try:
            pos[node]
        except KeyError:
            continue
        sizes = [ abs(size) for size in data.values() ]
        labels = list(data.keys())
        pie_colors = [function_to_color[label] for label in labels]
        pie_radius = function_to_radius[node]
        
        x, y = pos[node]
        
        ax.pie(sizes, colors=pie_colors, radius=pie_radius, center=(x, y),
                wedgeprops=dict(edgecolor='w'))
    
    # Add labels for each node next to them
    for node, (x, y) in pos.items():
        ax.text(x, y + 0.07, s=node, horizontalalignment='center',
                verticalalignment='center', fontsize=label_fontsize,
                color='red', fontweight='bold')
    
    # Legend
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=func,
                                 markerfacecolor=color, markersize=5)
                      for func, color in function_to_color.items()]
    
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1),
               loc='upper left', title="Functions", fontsize = legend_fontsize)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if show:
        plt.show()

    return fig, ax


def plot_scatter_2genes(adata, gene_a: str, gene_b: str, symbol_feature:str = 'feature_name',
                        color_feature:str = None, scale:str=None, title:str = None, 
                        cutoff_a: int = 0, cutoff_b:int = 0) -> plt.Axes:
    
    
    if not symbol_feature:
        gene_a_ens = gene_a
        gene_b_ens = gene_b
    else:
        gene_a_ens = adata.var.index[adata.var[symbol_feature] == gene_a]
        gene_b_ens = adata.var.index[adata.var[symbol_feature] == gene_b]
    
    # Extract expression values for GeneA and GeneB
    gene_a_expression = adata[:, gene_a_ens].X.toarray().ravel()
    gene_b_expression = adata[:, gene_b_ens].X.toarray().ravel()
    
    mask_cutoff = np.logical_and(gene_a_expression > cutoff_a, gene_b_expression > cutoff_b)
    gene_a_expression = gene_a_expression[mask_cutoff]
    gene_b_expression = gene_b_expression[mask_cutoff]
    adata = adata[mask_cutoff]
    
    
    if scale == 'log':
        # If not shifted by 1 to many 0s and no linear regresssion is possible
        gene_a_expression = np.log(gene_a_expression)
        gene_b_expression = np.log(gene_b_expression)
        
    try:
        all_feats = adata.obs[color_feature].unique()
        cmap = plt.cm.tab10
        colors = cmap(np.linspace(0, 1, len(all_feats)))
        function_to_color = {func: color for func, color in zip(sorted(all_feats), colors)}
        point_colors = [function_to_color[label] for label in adata.obs[color_feature]]
    except KeyError:
        colors = 'blue'
        
    # Fit a linear trend line
    try:
        slope, intercept, r_value, p_value, std_err = linregress(gene_a_expression, gene_b_expression)
        line = slope * gene_a_expression + intercept
    except Exception as e:
        print(f'Will not plot {title} because {e}')
        return None

    # Create a figure and ax object
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a scatter plot on the ax
    if isinstance(colors, str):
        # Plot all data points in blue if there's no specific color feature
        ax.scatter(gene_a_expression, gene_b_expression, color='blue')
    else:
        # If there are specific colors for different categories
        for i, color_val in enumerate(adata.obs[color_feature].unique()):
            # Create mask for current category
            mask = adata.obs[color_feature] == color_val
            # Extract relevant expressions for current category
            gene_a_expression_cat = gene_a_expression[mask]
            gene_b_expression_cat = gene_b_expression[mask]

            # Scatter plot for current category
            ax.scatter(gene_a_expression_cat, gene_b_expression_cat, alpha=0.5, color=point_colors[i], label=color_val)

    # scatter = ax.scatter(gene_a_expression, gene_b_expression, alpha=0.5, color=point_colors)

    # Plot the linear trend line on the ax
    ax.plot(gene_a_expression, line, 'r', label=f'fitted line (p={p_value:.2e}, $r^2$={r_value**2:.5f}) slope={slope:.2f}')
    # Displaying the legend outside the plot
    
    # ax.set_yscale('log')
    # ax.set_xscale('log')

    # Label the axes
    ax.set_xlabel(f'{gene_a} mRNA level', fontsize=27)
    ax.set_ylabel(f'{gene_b} mRNA level', fontsize=27)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.set_title(f'{gene_a} vs {gene_b} expression with linear trend in {title}:{len(gene_a_expression)} samples', fontsize=19)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=18)

    # Return the ax object for further modification
    return ax


def plot_scatter_3genes(adata, gene_a: str, gene_b: str, gene_c: str, symbol_feature:str = 'feature_name',
                        color_feature:str = None, scale:str=None, title:str = None, 
                        cutoff_a: int = 0, cutoff_b:int = 0) -> plt.Axes:
    if not symbol_feature:
        gene_a_ens = gene_a
        gene_b_ens = gene_b
        gene_c_ens = gene_c
    else:
        gene_a_ens = adata.var.index[adata.var[symbol_feature] == gene_a]
        gene_b_ens = adata.var.index[adata.var[symbol_feature] == gene_b]
        gene_c_ens = adata.var.index[adata.var[symbol_feature] == gene_c]
    
    # Extract expression values for GeneA and GeneB
    gene_a_expression = adata[:, gene_a_ens].X.toarray().ravel()
    gene_b_expression = adata[:, gene_b_ens].X.toarray().ravel()
    gene_c_expression = adata[:, gene_c_ens].X.toarray().ravel()
    
    mask_cutoff = np.logical_and(gene_a_expression > cutoff_a, gene_b_expression > cutoff_b)
    mask_cutoff = np.logical_and(mask_cutoff, gene_c_expression > 0)
    gene_a_expression = gene_a_expression[mask_cutoff]
    gene_b_expression = gene_b_expression[mask_cutoff]
    gene_c_expression = gene_c_expression[mask_cutoff]
    adata = adata[mask_cutoff]
    
    
    if scale == 'log':
        # If not shifted by 1 to many 0s and no linear regresssion is possible
        gene_a_expression = np.log(gene_a_expression)
        gene_b_expression = np.log(gene_b_expression)
        gene_c_expression = np.log(gene_c_expression)
    try:
        colors = adata.obs[color_feature].unique()
        cmap = plt.cm.tab10
        colors = cmap(np.linspace(0, 1, len(colors)))
    except KeyError:
        colors = 'blue'


    x = gene_a_expression
    y = gene_b_expression
    z = gene_c_expression

    # Fit a linear model (plane) to the data
    # We need to add a column of ones to x and y to include an intercept in the model
    XYZ = np.column_stack((x, y, np.ones(x.shape)))
    coeffs, res, _, _ = np.linalg.lstsq(XYZ, z, rcond=None)  # Solves for Ax = b, where A = XYZ, and b = z
    res = res[0]
    total_var = np.sum((np.mean(z) - z)**2)
    r2 = 1 - (res/ total_var)

    # Calculate residual variance (sigma squared)
    residuals = z - XYZ @ coeffs
    dof = len(z) - len(coeffs)  # degrees of freedom
    residual_variance = residuals.T @ residuals / dof

    # Calculate standard errors of coefficients
    XtX_inv = np.linalg.inv(XYZ.T @ XYZ)
    standard_errors = np.sqrt(np.diag(XtX_inv * residual_variance))
    
    # Calculate t-statistics
    t_stats = coeffs / standard_errors
    
    # Calculate p-values
    p_values = [2 * (1 - stats.t.cdf(np.abs(t), dof)) for t in t_stats]
    print(p_values)

    # Create a meshgrid of x and y values to plot the plane
    xx, yy = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(y), max(y), 10))
    zz = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]

    # Plot
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the data points
    ax.scatter(x, y, z, color='b', label='Data points')

    # Plot the plane
    ax.plot_surface(xx, yy, zz, color='r', alpha=0.5, rstride=100, cstride=100, label='Trend plane')

    ax.set_xlabel(f'{gene_a} mRNA level', fontsize=14)
    ax.set_ylabel(f'{gene_b} mRNA level', fontsize=14)
    ax.set_zlabel(f'{gene_c} mRNA level', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=8)
    # ax.set_title(f'Scatter plot of {gene_a} and {gene_b} vs {gene_c} expression with linear trend in {title}:{len(gene_a_expression)} samples')

    # Since plt.legend does not support 3D plots directly, workaround to create a legend for the plane
    plane_patch = plt.Line2D([0], [0], linestyle="none", c='r', alpha=0.5, marker = 'o')
    gene_a = 'KMT2D'
    ax.legend([plane_patch], [f'$m_{{{gene_a}}}$={coeffs[0]:.2f}, $m_{{{gene_b}}}$={coeffs[1]:.2f}, $R^2$={r2:.3f}'], numpoints = 1, fontsize=10)
#Intercept={coeffs[2]:.3f},
    ax.view_init(elev=15, azim=-13)  # Example angles: elev is elevation, azim is azimuth
    # fig.tight_layout()
    return ax, fig

    
def line_regress_genes(adata, gene_a: str, gene_b: str, symbol_feature:str = 'feature_name',
                        scale:str=None, title:str = None, 
                        cutoff_a: int = 0, cutoff_b:int = 0) -> plt.Axes:
    
    
    if not symbol_feature:
        gene_a_ens = gene_a
        gene_b_ens = gene_b
    else:
        gene_a_ens = adata.var.index[adata.var[symbol_feature] == gene_a]
        gene_b_ens = adata.var.index[adata.var[symbol_feature] == gene_b]
    
    # Extract expression values for GeneA and GeneB
    gene_a_expression = adata[:, gene_a_ens].X.toarray().ravel()
    gene_b_expression = adata[:, gene_b_ens].X.toarray().ravel()
    
    mask_cutoff = np.logical_and(gene_a_expression > cutoff_a, gene_b_expression > cutoff_b)
    gene_a_expression = gene_a_expression[mask_cutoff]
    gene_b_expression = gene_b_expression[mask_cutoff]
    adata = adata[mask_cutoff]
    
    
    if scale == 'log':
        # If not shifted by 1 to many 0s and no linear regresssion is possible
        gene_a_expression = np.log(gene_a_expression)
        gene_b_expression = np.log(gene_b_expression)
        
    # Fit a linear trend line
    try:
        slope, intercept, r_value, p_value, std_err = linregress(gene_a_expression, gene_b_expression)
        line = slope * gene_a_expression + intercept
        print(f'Scatter plot of {gene_a} vs {gene_b} expression with linear trend in {title}:{len(gene_a_expression)} cells')
        return slope, intercept, r_value, p_value
        
    except Exception as e:
        print(f'Will not plot {title} because {e}')