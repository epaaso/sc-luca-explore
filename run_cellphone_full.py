import os
import re
import pandas as pd
import scanpy as sc
import numpy as np
import networkx as nx
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
from cellphonedb.utils import search_utils
from cellphonedb.utils import db_utils
import datetime

base_dir = '/home/epaaso/REPOS/sc-luca-explore'
backup_dir = '/datos/migccl/neto_maestria/luca_explore'
wilcox_dir = f'{base_dir}/nb_DE_wilcox/wilcoxon_DE'
graph_dir = f'{base_dir}/outputARACNE'
cpdb_version = 'v5.0.0'
cpdb_target_dir = os.path.join(backup_dir, 'cellphoneDB/db/', cpdb_version)
cpdb_file_path = f'{cpdb_target_dir}/cellphonedb.zip'

if not os.path.exists(cpdb_file_path):
    print(f"Downloading CellphoneDB version {cpdb_version} to {cpdb_target_dir}...")
    db_utils.download_database(cpdb_target_dir, cpdb_version)

times = ['III-IV']

for time in times:
    print(f"\n================ Processing time: {time} ================")
    if time == 'I-II':
        out_path = f'{backup_dir}/cellphoneDB/method2'
        time_folder = 'early'
    else:
        out_path = f'{backup_dir}/cellphoneDB/method2_{time}'
        time_folder = 'late'
        
    os.makedirs(out_path, exist_ok=True)
    
    meta_file_path = f'{backup_dir}/cellphoneDB/metadata_{time}.tsv' if time == 'III-IV' else f'{backup_dir}/cellphoneDB/metadata.tsv'
    counts_file_path = f'{backup_dir}/cellphoneDB/normalised_log_counts_{time}.h5ad' if time == 'III-IV' else f'{backup_dir}/cellphoneDB/normalised_log_counts.h5ad'
    
    # 1. Prepare data
    print("Preparing metadata and counts...")
    csv_path = f'{wilcox_dir}/auc_count_cellphonedb_{time}_funcnames.csv'
    if not os.path.exists(csv_path):
        print(f"Skipping {time} because {csv_path} does not exist.")
        continue
    count_matrix = pd.read_csv(csv_path, index_col=0)
    count_matrix['cell_type'] = None

    for i, name in enumerate(count_matrix.index):
        if 'extended' in name or 'UKIM' in name:
            count_matrix.at[name, 'cell_type'] = '_'.join(name.split('_')[:-1])
        else:
            count_matrix.at[name, 'cell_type'] = '_'.join(name.split('_')[:-2])

    count_matrix['barcode_sample'] = count_matrix.index
    count_matrix = count_matrix.drop(columns=['barcode_sample']).groupby(by=["cell_type"]).mean()
    count_matrix['barcode_sample'] = count_matrix.index
    count_matrix['cell_type'] = count_matrix.index
    count_matrix.index.name = 'CellType'

    count_matrix[['cell_type', 'barcode_sample']].to_csv(meta_file_path, sep='\t')     
    metadata = pd.read_csv(meta_file_path, sep = '\t', index_col=0)

    count_matrix.index = count_matrix['barcode_sample']
    count_matrix_ = count_matrix.drop(columns=['cell_type', 'barcode_sample'])
    
    if count_matrix_.empty:
        continue

    adata = sc.AnnData(count_matrix_)
    adata.obs = metadata
    
    if time == 'III-IV':
        adata.X = np.nan_to_num(adata.X, nan=0.5)

    adata.write_h5ad(counts_file_path)
    
    # 2. Run CellphoneDB
    print("Running CellphoneDB statistical analysis...")
    cpdb_results = cpdb_statistical_analysis_method.call(
        cpdb_file_path = cpdb_file_path,
        meta_file_path = meta_file_path,
        counts_file_path = counts_file_path,
        counts_data = 'hgnc_symbol',
        score_interactions = True,
        iterations = 1000,
        threshold = 0.001 if time == 'I-II' else 0.1,
        threads = 7,
        debug_seed = 42,
        result_precision = 3,
        pvalue = 0.1 if time == 'I-II' else 0.05,
        subsampling = False,
        separator = '|',
        debug = False,
        output_path = out_path,
        output_suffix = None
    )
    
    # 3. Extract LR Pairs for Every Ecotype
    print("Extracting LR pairs for ecotypes...")
    cluster_root = os.path.join(graph_dir, 'cluster_runs', time_folder)
    cluster_pattern = re.compile(r'cluster[_-]?(\d+)')
    
    if not os.path.exists(cluster_root):
        print(f"No cluster runs found for {time_folder} at {cluster_root}")
        continue

    for cluster_name in sorted(os.listdir(cluster_root)):
        cluster_path = os.path.join(cluster_root, cluster_name)
        if not os.path.isdir(cluster_path):
            continue
            
        folder_match = cluster_pattern.search(cluster_name)
        if not folder_match:
            continue
        cluster_id = int(folder_match.group(1))
        
        graph_found = False
        G = None
        for fname in os.listdir(cluster_path):
            if not fname.endswith(".txt"):
                continue
            if f"net_{time}_leidenwu_cluster" not in fname or not 'pearson' in fname:
                continue
            
            graph_path = os.path.join(cluster_path, fname)
            G = nx.read_edgelist(
                graph_path,
                delimiter='\t',
                data=[('MI', float), ('pvalue', float), ('pearson', float), ('pvals', float), ('sign', str)],
                create_using=nx.Graph(),
                nodetype=str,
            )
            graph_found = True
            break
            
        if not graph_found or G is None:
            continue
            
        nodes = list(G.nodes())
        print(f"Cluster {cluster_id} ({time}): {len(nodes)} cell types")
        
        all_interactions = []
        for node in nodes:
            try:
                search_results = search_utils.search_analysis_results(
                    query_cell_types_1 = [node],
                    query_cell_types_2 = nodes,
                    query_interactions = list(cpdb_results['significant_means'].interacting_pair),
                    significant_means = cpdb_results['significant_means'],
                    deconvoluted = cpdb_results['deconvoluted'],
                    separator = '|',
                    long_format = True,
                )
                
                sig_ints = search_results[search_results['significant_mean'] > 0]
                if not sig_ints.empty:
                    all_interactions.append(sig_ints)
            except Exception as e:
                pass
                
        if all_interactions:
            cluster_interactions = pd.concat(all_interactions).drop_duplicates()
            out_file = f"{cluster_path}/extracted_lr_pairs_all.csv"
            cluster_interactions.to_csv(out_file, index=False)
            print(f"Saved {len(cluster_interactions)} LR pairs for cluster {cluster_id} to {out_file}")
        else:
            print(f"No significant interactions found for cluster {cluster_id}")
