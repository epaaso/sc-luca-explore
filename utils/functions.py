import warnings

import urllib.request
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

import GEOparse

from typing import Optional, List, Union, Literal, Tuple, Dict
import anndata as ad
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


def rank_genes_group(
    adata: ad.AnnData,
    group_name: str,
    n_genes: int = 20,
    gene_symbols: Optional[str] = None,
    gene_names: List[str] = None,
    key: Optional[str] = 'rank_genes_groups',
    fontsize: int = 8,
    titlesize: int = 10,
    show: Optional[bool] = None,
    ax: Optional[plt.Axes] = None,
    **kwds,
):
    if n_genes < 1:
        raise ValueError(
            "Specifying a negative number for n_genes has not been implemented for "
            f"this plot. Received n_genes={n_genes}."
        )

    reference = str(adata.uns[key]['params']['reference'])
    # group_names = adata.uns[key]['names'].dtype.names if groups is None else groups

    ymin = np.Inf
    ymax = -np.Inf
    try:
        gene_names[0]
        gene_mask = np.isin(adata.uns[key]['names'][group_name], gene_names)
        scores = adata.uns[key]['scores'][group_name][gene_mask]
    except Exception as e:
        gene_names = adata.uns[key]['names'][group_name][:n_genes]
        scores = adata.uns[key]['scores'][group_name][:n_genes]
    
        
    
    ymin = np.min(scores)
    ymax = np.max(scores)
    ymax += 0.3 * (ymax - ymin)

    ax = ax if ax else plt.subplot(111)
    ax.set_ylim(ymin, ymax)

    ax.set_xlim(-0.9, n_genes - 0.1)

    # Mapping to gene_symbols
    if gene_symbols is not None:
        gene_names = adata.var[gene_symbols][gene_names]

    # Making labels
    for ig, gene_name in enumerate(gene_names):
        ax.text(
            ig,
            scores[ig],
            gene_name,
            rotation='vertical',
            verticalalignment='bottom',
            horizontalalignment='center',
            fontsize=fontsize
        )

    ax.set_title('{} vs. {}'.format(group_name, reference),
                fontsize=titlesize)
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