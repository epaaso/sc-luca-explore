# Cell 2
import os
import pandas as pd

base_dir = '/root/host_home/sc-luca-explore' 
# os.chdir('/root/host_home/luca')
os.chdir(base_dir)

from utils import abundance

from IPython.display import HTML, display
from cellphonedb.utils import db_releases_utils
import ktplotspy as kpy

display(HTML(db_releases_utils.get_remote_database_versions_html()['db_releases_html_table']))

# Cell 3
backup_dir = '/root/host_home/datos/'
# backup_dir = '/root/datos/maestria/netopaas/luca_explore'
wilcox_dir = f'{base_dir}/nb_DE_wilcox/wilcoxon_DE'
graph_dir = f'{base_dir}/outputARACNE'

# -- Version of the databse
cpdb_version = 'v5.0.0'
# os.mkdir(f'{backup_dir}/cellphoneDB')
cpdb_target_dir = os.path.join(backup_dir, 'cellphoneDB/db/', cpdb_version)
# Cell 5
from cellphonedb.utils import db_utils
db_utils.download_database(cpdb_target_dir, cpdb_version)
# Cell 9
backup_dir = '/root/host_home/datos/'
# backup_dir = '/root/datos/maestria/netopaas/luca_explore'
wilcox_dir = f'{base_dir}/nb_DE_wilcox/wilcoxon_DE'
graph_dir = f'{base_dir}/outputARACNE/'

# -- Version of the databse
cpdb_version = 'v5.0.0'
# os.mkdir(f'{backup_dir}/cellphoneDB')
cpdb_target_dir = os.path.join(backup_dir, 'cellphoneDB/db/', cpdb_version)
# Cell 10
time = 'I-II'
cpdb_file_path = f'{cpdb_target_dir}/cellphonedb.zip'
meta_file_path = f'{backup_dir}/cellphoneDB/metadata.tsv'
counts_file_path = f'{backup_dir}/cellphoneDB/normalised_log_counts.h5ad'
microenvs_file_path = f'{backup_dir}/cellphoneDB/microenvironment.tsv'
out_path = f'{backup_dir}/cellphoneDB/method2'
# Cell 13
count_matrix = pd.read_csv(f'{wilcox_dir}/auc_count_cellphonedb_{time}_funcnames.csv', index_col=0)
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
# count_matrix.index = range(count_matrix.shape[0])

count_matrix[['cell_type', 'barcode_sample']].to_csv(meta_file_path, sep='\t')     

metadata = pd.read_csv(meta_file_path, sep = '\t', index_col=0)
metadata.head()
# Cell 15
import scanpy as sc
count_matrix.index = count_matrix['barcode_sample']
count_matrix_ = count_matrix.drop(columns=['cell_type', 'barcode_sample'])

adata = sc.AnnData(count_matrix_)
adata.obs = metadata
adata.write_h5ad(counts_file_path)
adata
# Cell 16
adata = sc.read_h5ad(counts_file_path)
adata.X
# Cell 17
list(adata.obs.index).sort() == list(metadata['barcode_sample']).sort()
# Cell 18
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.boxplot(count_matrix.drop(columns=['cell_type', 'barcode_sample']).values.T, labels=count_matrix.index)
plt.xlabel('Row (Cell Type)')
plt.ylabel('Abundance')
plt.title('Boxplot of Abundance per Row (Across Columns)')
plt.xticks(rotation=90)
plt.show()
# Cell 20
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
cpdb_results = cpdb_statistical_analysis_method.call(
    cpdb_file_path = cpdb_file_path,                 # mandatory: CellphoneDB database zip file.
    meta_file_path = meta_file_path,                 # mandatory: tsv file defining barcodes to cell label.
    counts_file_path = adata,             # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
    counts_data = 'hgnc_symbol',                     # defines the gene annotation in counts matrix.
    # active_tfs_file_path = active_tf_path,           # optional: defines cell types and their active TFs.
    # microenvs_file_path = microenvs_file_path,       # optional (default: None): defines cells per microenvironment.
    score_interactions = True,                       # optional: whether to score interactions or not. 
    iterations = 1000,                               # denotes the number of shufflings performed in the analysis.
    threshold = 0.001,                                 # defines the min % of cells expressing a gene for this to be employed in the analysis.
    threads = 7,                                     # number of threads to use in the analysis.
    debug_seed = 42,                                 # debug randome seed. To disable >=0.
    result_precision = 3,                            # Sets the rounding for the mean values in significan_means.
    pvalue = 0.1,                                   # P-value threshold to employ for significance.
    subsampling = False,                             # To enable subsampling the data (geometri sketching).
    # subsampling_log = False,                         # (mandatory) enable subsampling log1p for non log-transformed data inputs.
    # subsampling_num_pc = 100,                        # Number of componets to subsample via geometric skectching (dafault: 100).
    # subsampling_num_cells = adata.shape[0]/3,                    # Number of cells to subsample (integer) (default: 1/3 of the dataset).
    separator = '|',                                 # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
    debug = False,                                   # Saves all intermediate tables employed during the analysis in pkl format.
    output_path = out_path,                          # Path to save results.
    output_suffix = None                             # Replaces the timestamp in the output files by a user defined string in the  (default: None).
    )

# Cell 21
date = "11_22_2025_190903"
# 04_03_2025_051244.txt
means = pd.read_csv(f"{out_path}/statistical_analysis_means_{date}.txt", sep='\t')
pvals = pd.read_csv(f"{out_path}/statistical_analysis_pvalues_{date}.txt", sep='\t')
decon = pd.read_csv(f"{out_path}/statistical_analysis_deconvoluted_{date}.txt", sep='\t')
significant_means = pd.read_csv(f"{out_path}/statistical_analysis_significant_means_{date}.txt", sep='\t')
intercation_scores = pd.read_csv(f"{out_path}/statistical_analysis_interaction_scores_{date}.txt", sep='\t')

cpdb_results_E = {
    "means": means,
    "pvalues": pvals,
    "deconvoluted": decon,
    "significant_means": significant_means,
    "interaction_scores": intercation_scores
}
# Cell 22
cpdb_results = cpdb_results_E
# Cell 23
import os
import anndata as ad
import pandas as pd
import ktplotspy as kpy
import matplotlib.pyplot as plt
%matplotlib inline

# Cell 25
import os
import re
import networkx as nx

Gs_E = {}
cluster_pattern = re.compile(r'cluster[_-]?(\d+)')

time_folder = {'I-II': 'early', 'III-IV': 'late'}.get(time, time.lower())
cluster_root = os.path.join(graph_dir, 'cluster_runs', time_folder)

for cluster_name in sorted(os.listdir(cluster_root)):
    cluster_path = os.path.join(cluster_root, cluster_name)
    if not os.path.isdir(cluster_path):
        continue

    folder_match = cluster_pattern.search(cluster_name)
    if not folder_match:
        continue
    cluster_id = int(folder_match.group(1))

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

        # Normalize attribute names and provide a canonical 'weight'
        for _, _, d in G.edges(data=True):
            # Canonical weight: use MI if present, else absolute Pearson
            d['weight'] = d.get('MI', abs(d.get('pearson', 1.0)))
            if 'pvalue' in d:
                d['p_value'] = d['pvalue']

        G.graph['source_path'] = str(graph_path)
        Gs_E[cluster_id] = G
        break
# Cell 26
Gs = Gs_E
# Cell 27
for G in Gs.values():
    nx.set_node_attributes(G, 0.0, 'degree_centrality')
    degree_centrality = nx.degree_centrality(G)
    nx.set_node_attributes(G, degree_centrality, 'degree_centrality')

    nx.set_node_attributes(G, 0.0, 'betweenness')
    betweenness = nx.betweenness_centrality(G, normalized=True)
    nx.set_node_attributes(G, betweenness, 'betweenness')
# Cell 28
cell_cats = list(abundance.color_map.keys())
cell_cats.remove('epithelial')

cmap = abundance.cell_category_mapping

top2_nodes = {}
for cluster_id, G in Gs.items():
    # Sort vertices by degree in descending order and take the top two
    top2_nodes[cluster_id] = []
    for cat in cell_cats:
        top2 = sorted([(name, G.nodes[name]['betweenness']) for name in G.nodes() if cmap(name)==cat], key=lambda x: x[1], reverse=True)[:2]
        selected_nodes = [name for name, _ in top2]
        top2_nodes[cluster_id].extend(selected_nodes)

top2_nodes
# Cell 29
from cellphonedb.utils import search_utils

top_interactions = {}

for cluster_id, nodes in top2_nodes.items():
    top_interactions[cluster_id] = []
    for node in nodes:
        search_results = search_utils.search_analysis_results(
            query_cell_types_1 = [node],  # List of cells 1, will be paired to cells 2 (list or 'All').
            query_cell_types_2 = nodes,     # List of cells 2, will be paired to cells 1 (list or 'All').
            # query_genes = ['TGFBR1'],                                       # filter interactions based on the genes participating (list).
            query_interactions = list(cpdb_results['significant_means'].interacting_pair),                            # filter intereactions based on their name (list).
            significant_means = cpdb_results['significant_means'],          # significant_means file generated by CellphoneDB.
            deconvoluted = cpdb_results['deconvoluted'],                    # devonvoluted file generated by CellphoneDB.
            # interaction_scores = cpdb_results['interaction_scores'],        # interaction score generated by CellphoneDB.
            # query_minimum_score = 0,                                       # minimum score that an interaction must have to be filtered.
            separator = '|',                                                # separator (default: |) employed to split cells (cellA|cellB).
            long_format = True,                                             # converts the output into a wide table, removing non-significant interactions
            # query_classifications = ['Signaling by Transforming growth factor']
        )

        top_ints = search_results.sort_values(by='significant_mean', ascending=False)['interacting_pair'][:3]
        top_interactions[cluster_id].extend(list(top_ints))
    top_interactions[cluster_id] = list(set(top_interactions[cluster_id]))  # Remove duplicates
top_interactions[0]
# Cell 31
cpdb_tumors= [x for x in cpdb_results['deconvoluted'].columns if 'Tumor' in x]
g_tumors = [node for node in list(Gs[0].nodes) if  'Tumor' in node]

# check intersection 
intersection = set(cpdb_tumors).intersection(set(g_tumors))
intersection
# Cell 32
groups = pd.read_csv(f'metadata/groups_{time}_leidenwu_funcnames.csv')
pred_tumors = [x for x in groups['cell_type_adjusted'] if 'Tumor' in x]
intersection = set(cpdb_tumors).intersection(set(pred_tumors))
intersection
# Cell 34
from plotnine import facet_wrap

ps = []
clusters = list(top_interactions.keys())
# clusters.remove('0')
for cluster_id in clusters:
    p = kpy.plot_cpdb(
        adata = adata,
        cell_type1='|'.join(top2_nodes[cluster_id]),
        cell_type2='|'.join(top2_nodes[cluster_id]),
        means = cpdb_results['means'],
        pvals = cpdb_results['pvalues'],
        # interaction_scores= cpdb_results['interaction_scores'],
        celltype_key = "cell_type",
        # genes = ["TGFB2", "CSF1R", "COL1A1"],
        interacting_pairs=[x.replace('_', '-') for x in top_interactions[cluster_id]],
        figsize = (25, 25),
        title = f"Highest interactions in cluster {cluster_id} for {time}",
        max_size = 6,
        highlight_size = 0.75,
        degs_analysis = False,
        standard_scale = False,
        keep_significant_only=False,
        # return_classif=True
    )
    (p + facet_wrap('~ classification', ncol=2)).show()
    ps.append(p)
# Cell 35
clusters = list(top_interactions.keys())
for cluster_id in clusters:
    fig = kpy.plot_cpdb_chord(
        adata = adata,
        cell_type1='|'.join(top2_nodes[cluster_id]),
        cell_type2='|'.join(top2_nodes[cluster_id]),
        means = cpdb_results['means'],
        pvals = cpdb_results['pvalues'],
        deconvoluted = cpdb_results['deconvoluted'],
        interaction_scores= cpdb_results['interaction_scores'],
        min_interaction_score=50,
        celltype_key = "cell_type",
        # genes = ["TGFB2", "CSF1R", "COL1A1"],
        interacting_pairs=[x.replace('_', '-') for x in top_interactions[cluster_id]],
        figsize = (10, 10),
        title = f"Highest interactions in cluster {cluster_id} for {time}",
        legend_kwargs = { "loc": "center","bbox_to_anchor": (1, 0.9),"fontsize": 6 },
        max_size = 6,
        highlight_size = 0.75,
        degs_analysis = False,
        standard_scale = False,
        keep_significant_only=True,
        # return_classif=False
    )
    fig.savefig(f"chord_{time}_{cluster_id}.png", dpi=300)
# Cell 37
time = 'III-IV'
cpdb_file_path = f'{cpdb_target_dir}/cellphonedb.zip'
meta_file_path = f'{backup_dir}/cellphoneDB/metadata_{time}.tsv'
counts_file_path = f'{backup_dir}/cellphoneDB/normalised_log_counts_{time}.h5ad'
microenvs_file_path = f'{backup_dir}/cellphoneDB/microenvironment_{time}.tsv'
out_path = f'{backup_dir}/cellphoneDB/method2_{time}'
# Cell 40
count_matrix = pd.read_csv(f'{wilcox_dir}/auc_count_cellphonedb_{time}_funcnames.csv', index_col=0)
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
# count_matrix.index = range(count_matrix.shape[0])

count_matrix[['cell_type', 'barcode_sample']].to_csv(meta_file_path, sep='\t')     

metadata = pd.read_csv(meta_file_path, sep = '\t', index_col=0)
metadata
# Cell 42
import scanpy as sc
count_matrix.index = count_matrix['barcode_sample']
count_matrix_ = count_matrix.drop(columns=['cell_type', 'barcode_sample'])

adata = sc.AnnData(count_matrix_)
adata.obs = metadata
adata.write_h5ad(counts_file_path)
adata
# Cell 43
import numpy as np

adata = sc.read_h5ad(counts_file_path)
adata.X = np.nan_to_num(adata.X, nan=0.5)
adata.X
# Cell 44
list(adata.obs.index).sort() == list(metadata['barcode_sample']).sort()
# Cell 46
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
cpdb_results_L = cpdb_statistical_analysis_method.call(
    cpdb_file_path = cpdb_file_path,                 # mandatory: CellphoneDB database zip file.
    meta_file_path = meta_file_path,                 # mandatory: tsv file defining barcodes to cell label.
    counts_file_path = adata,             # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
    counts_data = 'hgnc_symbol',                     # defines the gene annotation in counts matrix.
    # active_tfs_file_path = active_tf_path,           # optional: defines cell types and their active TFs.
    # microenvs_file_path = microenvs_file_path,       # optional (default: None): defines cells per microenvironment.
    score_interactions = True,                       # optional: whether to score interactions or not. 
    iterations = 1000,                               # denotes the number of shufflings performed in the analysis.
    threshold = 0.1,                                 # defines the min % of cells expressing a gene for this to be employed in the analysis.
    threads = 6,                                     # number of threads to use in the analysis.
    debug_seed = 42,                                 # debug randome seed. To disable >=0.
    result_precision = 3,                            # Sets the rounding for the mean values in significan_means.
    pvalue = 0.05,                                   # P-value threshold to employ for significance.
    subsampling = False,                             # To enable subsampling the data (geometri sketching).
    subsampling_log = False,                         # (mandatory) enable subsampling log1p for non log-transformed data inputs.
    subsampling_num_pc = 100,                        # Number of componets to subsample via geometric skectching (dafault: 100).
    subsampling_num_cells = adata.shape[0]/3,                    # Number of cells to subsample (integer) (default: 1/3 of the dataset).
    separator = '|',                                 # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
    debug = False,                                   # Saves all intermediate tables employed during the analysis in pkl format.
    output_path = out_path,                          # Path to save results.
    output_suffix = None                             # Replaces the timestamp in the output files by a user defined string in the  (default: None).
    )

# Cell 47
date = '11_23_2025_000201'
means = pd.read_csv(f"{out_path}/statistical_analysis_means_{date}.txt", sep='\t')
pvals = pd.read_csv(f"{out_path}/statistical_analysis_pvalues_{date}.txt", sep='\t')
decon = pd.read_csv(f"{out_path}/statistical_analysis_deconvoluted_{date}.txt", sep='\t')
significant_means = pd.read_csv(f"{out_path}/statistical_analysis_significant_means_{date}.txt", sep='\t')
intercation_scores = pd.read_csv(f"{out_path}/statistical_analysis_interaction_scores_{date}.txt", sep='\t')

cpdb_results_L = {
    "means": means,
    "pvalues": pvals,
    "deconvoluted": decon,
    "significant_means": significant_means,
    "interaction_scores": intercation_scores
}
# Cell 48
# cpdb_results_R = cpdb_results
cpdb_results = cpdb_results_L
# Cell 49
import pandas as pd
import ktplotspy as kpy
import matplotlib.pyplot as plt
%matplotlib inline
# Cell 51
import os
import re
import networkx as nx

Gs_L = {}
cluster_pattern = re.compile(r'cluster[_-]?(\d+)')

time_folder = {'I-II': 'early', 'III-IV': 'late'}.get(time, time.lower())
cluster_root = os.path.join(graph_dir, 'cluster_runs', time_folder)

for cluster_name in sorted(os.listdir(cluster_root)):
    cluster_path = os.path.join(cluster_root, cluster_name)
    if not os.path.isdir(cluster_path):
        continue

    folder_match = cluster_pattern.search(cluster_name)
    if not folder_match:
        continue
    cluster_id = int(folder_match.group(1))

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

        # Normalize attribute names and provide a canonical 'weight'
        for _, _, d in G.edges(data=True):
            # Canonical weight: use MI if present, else absolute Pearson
            d['weight'] = d.get('MI', abs(d.get('pearson', 1.0)))
            if 'pvalue' in d:
                d['p_value'] = d['pvalue']

        G.graph['source_path'] = str(graph_path)
        Gs_L[cluster_id] = G
        break
# Cell 52
Gs = Gs_L
# Cell 53
for G in Gs.values():
    nx.set_node_attributes(G, 0.0, 'degree_centrality')
    degree_centrality = nx.degree_centrality(G)
    nx.set_node_attributes(G, degree_centrality, 'degree_centrality')

    nx.set_node_attributes(G, 0.0, 'betweenness')
    betweenness = nx.betweenness_centrality(G, normalized=True)
    nx.set_node_attributes(G, betweenness, 'betweenness')
# Cell 54
cell_cats = list(abundance.color_map.keys())
cell_cats.remove('epithelial')

cmap = abundance.cell_category_mapping

top2_nodes = {}
for cluster_id, G in Gs.items():
    # Sort vertices by degree in descending order and take the top two
    top2_nodes[cluster_id] = []
    for cat in cell_cats:
        top2 = sorted([(name, G.nodes[name]['betweenness']) for name in G.nodes() if cmap(name)==cat], key=lambda x: x[1], reverse=True)[:2]
        selected_nodes = [name for name, _ in top2]
        top2_nodes[cluster_id].extend(selected_nodes)

top2_nodes
# Cell 55
from cellphonedb.utils import search_utils

top_interactions = {}

for cluster_id, nodes in top2_nodes.items():
    top_interactions[cluster_id] = []
    for node in nodes:
        search_results = search_utils.search_analysis_results(
            query_cell_types_1 = [node],  # List of cells 1, will be paired to cells 2 (list or 'All').
            query_cell_types_2 = nodes,     # List of cells 2, will be paired to cells 1 (list or 'All').
            # query_genes = ['TGFBR1'],                                       # filter interactions based on the genes participating (list).
            query_interactions = list(cpdb_results['significant_means'].interacting_pair),                            # filter intereactions based on their name (list).
            significant_means = cpdb_results['significant_means'],          # significant_means file generated by CellphoneDB.
            deconvoluted = cpdb_results['deconvoluted'],                    # devonvoluted file generated by CellphoneDB.
            # interaction_scores = cpdb_results['interaction_scores'],        # interaction score generated by CellphoneDB.
            # query_minimum_score = 0,                                       # minimum score that an interaction must have to be filtered.
            separator = '|',                                                # separator (default: |) employed to split cells (cellA|cellB).
            long_format = True,                                             # converts the output into a wide table, removing non-significant interactions
            # query_classifications = ['Signaling by Transforming growth factor']
        )

        top_ints = search_results.sort_values(by='significant_mean', ascending=False)['interacting_pair'][:3]
        top_interactions[cluster_id].extend(list(top_ints))
    top_interactions[cluster_id] = list(set(top_interactions[cluster_id]))  # Remove duplicates
top_interactions[2]
# Cell 57
cpdb_tumors= [x for x in cpdb_results['deconvoluted'].columns if 'Tumor' in x]
g_tumors = [node for node in list(Gs[0].nodes) if  'Tumor' in node]

# check intersection 
intersection = set(cpdb_tumors).intersection(set(g_tumors))
intersection
# Cell 58
groups = pd.read_csv(f'metadata/groups_{time}_leidenwu_funcnames.csv')
pred_tumors = [x for x in groups['cell_type_adjusted'] if 'Tumor' in x]
intersection = set(cpdb_tumors).intersection(set(pred_tumors))
intersection
# Cell 60
from plotnine import facet_wrap

ps = []
clusters = list(top_interactions.keys())
# clusters.remove('0')
for cluster_id in clusters:
    p = kpy.plot_cpdb(
        adata = adata,
        cell_type1='|'.join(top2_nodes[cluster_id]),
        cell_type2='|'.join(top2_nodes[cluster_id]),
        means = cpdb_results['means'],
        pvals = cpdb_results['pvalues'],
        # interaction_scores= cpdb_results['interaction_scores'],
        celltype_key = "cell_type",
        # genes = ["TGFB2", "CSF1R", "COL1A1"],
        interacting_pairs=[x.replace('_', '-') for x in top_interactions[cluster_id]],
        figsize = (25, 25),
        title = f"Highest interactions in cluster {cluster_id} for {time}",
        max_size = 6,
        highlight_size = 0.75,
        degs_analysis = False,
        standard_scale = False,
        keep_significant_only=False,
        # return_classif=True
    )
    (p + facet_wrap('~ classification', ncol=2)).show()
    ps.append(p)
# Cell 61
clusters = list(top_interactions.keys())
for cluster_id in clusters:
    fig = kpy.plot_cpdb_chord(
        adata = adata,
        cell_type1='|'.join(top2_nodes[cluster_id]),
        cell_type2='|'.join(top2_nodes[cluster_id]),
        means = cpdb_results['means'],
        pvals = cpdb_results['pvalues'],
        deconvoluted = cpdb_results['deconvoluted'],
        interaction_scores= cpdb_results['interaction_scores'],
        min_interaction_score=50,
        celltype_key = "cell_type",
        # genes = ["TGFB2", "CSF1R", "COL1A1"],
        interacting_pairs=[x.replace('_', '-') for x in top_interactions[cluster_id]],
        figsize = (10, 10),
        title = f"Highest interactions in cluster {cluster_id} for {time}",
        legend_kwargs = { "loc": "center","bbox_to_anchor": (1, 0.9),"fontsize": 6 },
        max_size = 6,
        highlight_size = 0.75,
        degs_analysis = False,
        standard_scale = False,
        keep_significant_only=True,
        # return_classif=False
    )
    fig.savefig(f"chord_{time}_{cluster_id}.png", dpi=300)
# Cell 62

