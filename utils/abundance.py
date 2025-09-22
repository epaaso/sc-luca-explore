import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import pearsonr
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

import holoviews as hv
from collections import Counter
from bokeh.models import GlyphRenderer
from bokeh.models.glyphs import Text
# activate your preferred backend
if not hv.Store.current_backend:
    hv.extension('matplotlib')

# Annotate samples with therapy, Wu Zhou has 7 that are not annotated
samples_therapy = {
    'immune01': 'chemo', 'immune02': 'immuno+chemo', 'immune03': 'immuno+chemo', 'immune04': 'immuno+chemo', 'immune05': 'chemo',
    'immune06': 'immuno+chemo', 'immune07': 'immuno+chemo', 'immune08': 'chemo', 'immune09': 'immuno+chemo', 'immune10': 'immuno+chemo',
    'immune11': 'immuno+chemo', 'immune12': 'immuno+chemo', 'immune13': 'immuno+chemo', 'immune14': 'immuno+chemo', 'immune15': 'immuno+chemo',
    'p2t1': 'XRT+chemo', '2019_p2t2': 'XRT+chemo',
    'S01': 'TKI', 'S11': 'TKI', 'S56': 'TKI', 'S58': 'TKI',
    'S71': 'TKI', 'S79': 'TKI', 'S82': 'TKI',
    'GSM3516670': 'chemo'
}

# Define colors for different cell categories
color_map = {
    'immune adaptive': 'lightgreen',
    'immune innate': 'blue',
    'immune both': 'purple',
    'stromal': '#D2B48C',
    'epithelial': 'orange',
    'tumoral': 'red'
}

# Define cell categories
cell_categories = {
    'immune adaptive': ['T cell CD8 activated', 'T cell CD4', 'B cell', 'B cell dividing', 'T cell regulatory', 'T cell CD8 effector memory',
                         'T cell CD8 activated', 'cDC2', 'T cell CD8 terminally exhausted', 'T cell CD4 dividing', 'T cell CD8 dividing', 'T cell CD8 naive'],
    'immune both': ['Mast cell', 'myeloid dividing', 'pDC', 'DC mature', 'T cell NK-like', "Plasma cell", "Plasma cell dividing"],
    'immune innate': ['Monocyte classical', 'Monocyte non-classical', 'NK cell', 'Macrophage', 'Macrophage alveolar', 'NK cell dividing', 'Neutrophils', 'cDC1'],
    'stromal': ['Fibroblast peribronchial', 'Fibroblast alveolar', 'Endothelial cell venous','Endothelial cell arterial', 'Endothelial cell lymphatic',
                 'Endothelial cell capillary', 'Smooth muscle cell', 'Pericyte', 'Fibroblast adventitial', 'stromal dividing'],
    'epithelial': ['Alveolar cell type 1', 'Alveolar cell type 2', 'Ciliated', 'Club', 'transitional club/AT2', 'ROS1+ healthy epithelial',
                    'Mesothelial','AT1', 'AT2', 'AT2_ROS1+', 'Club_AT2_even_1','Club_AT2_even_2', 'Club_Ciliated', 'Club_AT2_high', 'Club_AT2_low',
                    'Club/AT2','Club_AT2/1_high' ],
}

cell_type_to_category = {}
for category, cell_types in cell_categories.items():
    for cell_type in cell_types:
        cell_type_to_category[cell_type] = category

def cell_category_mapping(cell_type):
    if 'Tumor' in cell_type:
        return 'tumoral'
    else:
        return cell_type_to_category[cell_type]


def draw_graph(G, ax, title, scale1= 30, scale2=4, k=3, font_size=8):
    

    # Assign colors based on cell category
    node_colors = []
    for node in G.nodes():
        category = cell_category_mapping(node)
        node_colors.append(color_map.get(category, 'gray'))

    # Perform Newman clustering
    newman_communities = list(greedy_modularity_communities(G))

    # Map nodes to their community index
    node_community_newman = {}
    for idx, community in enumerate(newman_communities):
        for node in community:
            node_community_newman[node] = idx

    # Assign colors to nodes based on community
    node_colors_newman = [node_community_newman[node] for node in G.nodes()]

    # Compute positions for the node clusters
    superpos = nx.spring_layout(G,
    scale=scale1,  # SCAAAAAAAAAAAALE 1
    seed=1)
    centers = list(superpos.values())
    pos = {}
    for center, comm in zip(centers, newman_communities):
        subgraph = G.subgraph(comm)
        subpos = nx.spring_layout(subgraph, center=center, seed=1,
                                scale=scale2, k=k) # SCAAAAAAAAAAAALE 2
        pos.update(subpos)

    # Avoid label overlap (optional adjustment)
    # for k, v in pos.items():
    #     for k2, v2 in pos.items():
    #         if k != k2 and abs(v[0] - v2[0]) < 700 and abs(v[1] - v2[1]) < 20:
    #             pos[k2] = (v2[0], v2[1] + 50)

    nx.draw(
        G, pos=pos, node_color=node_colors, node_size=300,
        edge_color='grey', with_labels=True, font_size=font_size,
        font_color="black", ax=ax
    )

    legend_handles = [mpatches.Patch(color=color, label=category) for category, color in color_map.items()]
    ax.legend(handles=legend_handles, loc='best', fontsize=8)
    # plt.show()

    ax.set_title(title)
    ax.axis('off')


def plot_abundance_heatmap(corr_:pd.DataFrame, show_plot=True, cluster_samples=False,
                            xtick_params:dict = {}, ylabel_size:int = 8):
    """
    Plot a heatmap of normalized counts for given samples. Scales with a log function and the zeros->infs
    are converted to nans.

    Parameters
    ----------
    corr_ : pandas.DataFrame
        Contains sample-level normalized counts and a 'dataset' column.

    cluster_samples : bool
        If True, cluster samples using hierarchical clustering

    Returns
    -------
    None
        Displays a heatmap with annotated brackets indicating datasets
        and cell type categories.
    """
    corr = corr_.copy()

    # Existing processing code...
    if 'sample' in corr.columns:
        corr.set_index('sample', inplace=True)
    corr.index = [sample.split('_')[-1] for sample in corr.index]
    corr.sort_values('dataset', inplace=True)
    dataset = corr['dataset'][::-1]

    # Create figure with adjusted gridspec to include colorbar space
    fig = plt.figure(figsize=(22, 20))
    gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[1, 25, 0.3], height_ratios=[10, 1])

    ax0 = fig.add_subplot(gs[0, 0])  # Left brackets
    ax1 = fig.add_subplot(gs[0, 1])  # Heatmap
    ax_cbar = fig.add_subplot(gs[0, 2])  # Colorbar
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom brackets

    # Plot heatmap with colorbar in the dedicated axis
    data = np.log(corr.drop(columns='dataset'))
    data = data.replace(-np.inf, np.nan).replace(np.inf, np.nan)
    # Reorder columns as before...
    ordered_cell_types = []
    for cat, cat_cell_types in cell_categories.items():
        ordered_cell_types.extend([ct for ct in cat_cell_types if ct in data.columns])
    other_cols = [col for col in data.columns if col not in ordered_cell_types]
    data = data[ordered_cell_types + other_cols]
    cell_types = list(data.columns)
    
    # Plot heatmap and align axes
    if cluster_samples:
        data_link = data.replace(np.nan, -20)
        Z = linkage(data_link.values, method='average', metric='euclidean')
        row_order = leaves_list(Z)
        data = data.iloc[row_order]

    sns.heatmap(
        data,
        cmap='viridis',
        ax=ax1,
        cbar_ax=ax_cbar,
        cbar_kws={'label': 'Ln(Normalized Counts)'},
        yticklabels=data.index
    )
    ax1.set_yticks(np.arange(len(data.index)) + 0.5)
    ax1.set_yticklabels(data.index, rotation=0, fontsize=ylabel_size)
    xtick_params_default = {'rotation': 45, 'ha': 'right', 'rotation_mode': 'anchor', 'size': 12}
    xtick_params = {**xtick_params_default, **xtick_params}
    ax1.set_xticklabels(ax1.get_xticklabels(), **xtick_params)
    for label in ax1.get_yticklabels():
        if label.get_text() in samples_therapy:
            label.set_color("red")



    ############################ DATASET BRACKETS ###############################3
    ax0.axis('off')

    # Set the limits to match the heatmap
    ax0.set_ylim(ax1.get_ylim())

    # Find positions where the dataset changes
    positions = []
    start = 0
    current_dataset = dataset.iloc[0]
    for i, dataset_ in enumerate(dataset):
        if dataset_ != current_dataset:
            positions.append((start, i - 1, current_dataset))
            start = i
            current_dataset = dataset_
    positions.append((start, len(dataset) - 1, current_dataset))

    # Create a mapping of dataset names to colors
    unique_datasets = dataset.unique()
    colors = plt.get_cmap('tab20', len(unique_datasets))
    color_mapping = {dataset: colors(i) for i, dataset in enumerate(unique_datasets)}

    # Draw brackets on ax0
    for (start, end, dataset) in positions:
        y_start = len(corr.index) - start
        y_end = len(corr.index) - end - 1
        ax0.plot([0, 0], [y_start, y_end], color=color_mapping[dataset], linewidth=2)
        ax0.text(0, (y_start + y_end) / 2, dataset[-18:], va='center', ha='right', fontsize=8)


    ############################ CELL TYPE BRACKETS ###############################
    ax3.set_xlim(ax1.get_xlim())  # Explicitly align x-axis limits

    # Clean up ax3's appearance
    ax3.axis('off')
    ax3.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for spine in ax3.spines.values():
        spine.set_visible(False)

    # Draw dataset brackets on ax0 (existing code)...
    # Draw category brackets on ax3...
    def draw_bracket(ax, start, end, text, y=-0.5, bracket_height=0.1, color='black'):
        ax.plot([start, start], [y, y - bracket_height], lw=1.5, color=color)
        ax.plot([end, end], [y, y - bracket_height], lw=1.5, color=color)
        ax.plot([start, end], [y, y], lw=1.5, color=color)
        ax.text((start + end)/2, y - bracket_height, text, ha='center', va='top', fontsize=14)

    y_level = -0.8
    end = 0
    for cat_name, cat_celltypes in cell_categories.items():
        indices = [cell_types.index(ct) for ct in cat_celltypes if ct in cell_types]
        if not indices:
            continue
        start, end = min(indices), max(indices)
        draw_bracket(ax3, start, end+0.5, cat_name, y=y_level, color=color_map[cat_name])
    draw_bracket(ax3, end+1, len(cell_types), 'tumoral', y=y_level, color=color_map['tumoral'])

    ax3.set_ylim(y_level - 0.5, None)  # Adjust ylim for bracket space

    plt.tight_layout()
    pos = ax_cbar.get_position()
    pos = [pos.x0, pos.y0, pos.width, pos.height]  # Shift the colorbar 0.05 units to the left
    ax_cbar.set_position(pos)
    
    if show_plot:
        plt.show()

    return fig


def draw_pearson_graphs(corr_types_, title, corr_threshold:float=0.25, p_threshold:float=0.05, draw:bool=True):
    """
    Draw three Pearson correlation graphs (overall, negative, and positive).

    Parameters
    ----------
    corr_types_ : pd.DataFrame
        DataFrame containing columns to compute pairwise correlations on. membership and dataset columns must be that last cols.
    title : str
        Title used for labeling the graphs.
    corr_threshold : float
        Minimum absolute correlation value to keep an edge.
    p_threshold : float
        Maximum p-value threshold to keep an edge.

    Returns
    -------
    networkx.Graph
        A graph containing nodes and edges based on correlation and p-value constraints.
        Also prints out the neg cand pos correlation graphs.
    """

    data = corr_types_.drop(columns=['dataset']).to_numpy().T
    col_names = corr_types_.drop(columns=['dataset']).columns
    # Initialize matrices for correlation coefficients and p-values
    n_variables = data.shape[0]  # Number of variables, eg. cell types``
    pearson_matrix = np.zeros((n_variables, n_variables))
    pearson_p_values = np.zeros((n_variables, n_variables))

    for i in range(n_variables):
        for j in range(n_variables):
            if i == j:
                # The correlation of a variable with itself is always 1, and the p-value is 0
                pearson_matrix[i, j] = 1
                pearson_p_values[i, j] = 0
            elif i < j:
                # Compute correlation and p-value for unique pairs only
                correlation, p_value = pearsonr(data[i], data[j])
                if np.isnan(correlation):
                    # print(f'Correlation between {col_names[i]} and {col_names[j]} is NaN')
                    correlation = 0
                    p_value = 1
                pearson_matrix[i, j] = pearson_matrix[j, i] = correlation
                pearson_p_values[i, j] = pearson_p_values[j, i] = p_value
                
    pearson_df = pd.DataFrame(pearson_matrix, columns=col_names, index=col_names)
    pearson_p_values_df = pd.DataFrame(pearson_p_values, columns=col_names, index=col_names)

    G = nx.Graph()

    # Add edges between all nodes with correlation as edge attribute
    for col1 in pearson_df.columns:
        for col2 in pearson_df.index:
            if col1 != col2:
                # Add an edge between col1 and col2 with the correlation as an attribute
                G.add_edge(col1, col2, weight=(round(pearson_df.loc[col1, col2],2)))

    # Remove edges with very low correlation and high p-value
    threshold = corr_threshold
    threshold_p = p_threshold
    for (u, v, d) in list(G.edges(data=True)):        
        if abs(d['weight']) < threshold or abs(pearson_p_values_df.loc[u, v]) > threshold_p:
            G.remove_edge(u, v)

    # Remove all nodes with no edges
    nodes_with_no_edges = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(nodes_with_no_edges)

    # Save the corr_threshold as an attribute of the graph G
    G.graph['corr_threshold'] = corr_threshold

    # Subset to only edges with negative weight
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    G_negative = G.edge_subgraph(negative_edges).copy()
    G_positive = G.edge_subgraph(positive_edges).copy()

    if draw:
        _, axes = plt.subplots(1, 3, figsize=(22, 9))  # Create a figure with 2 subplots
        draw_graph(G, axes[0], f'Pearson Corr {title}')

        draw_graph(G_negative, axes[1], f'Negative Corr {title}')

        draw_graph(G_positive, axes[2], f'Positive Corr {title}')

    return G_positive


def plot_degree_centrality(
    G: nx.Graph,
    betweenness=False,
    time='III-IV',
    ax=None
):
    """
    Plots a bar chart of degree or betweenness centralities for all nodes in G, grouped by category,
    sorted within each category, and colored according to color_map.
    """
    # 1) Compute degree or betweenness centrality
    if not betweenness:
        centrality = nx.degree_centrality(G)
        metric = "Degree"
    else:
        centrality = nx.betweenness_centrality(G)
        metric = "Betweenness"
    
    # Store centrality values in the graph nodes
    centrality_attr = f"{metric.lower()}_centrality"
    for node, val in centrality.items():
        G.nodes[node][centrality_attr] = val
    
    # 2) Group nodes by their category and sort within each category
    from collections import defaultdict
    category_groups = defaultdict(list)
    for node, cent in centrality.items():
        category = cell_category_mapping(node)
        category_groups[category].append((node, cent))
    
    # Sort each category's nodes by centrality in descending order
    for category in category_groups:
        category_groups[category].sort(key=lambda x: x[1], reverse=True)
    
    # 3) Sort categories alphabetically and concatenate nodes
    sorted_categories = sorted(category_groups.keys())
    sorted_centrality = []
    for category in sorted_categories:
        sorted_centrality.extend(category_groups[category])
    
    # 4) Extract nodes, values, and prepare colors
    if sorted_centrality:
        nodes, values = zip(*sorted_centrality)
    else:
        nodes, values = [], []
    colors = [color_map[cell_category_mapping(node)] for node in nodes] if nodes else []
    
    # 5) Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    
    ax.bar(range(len(values)), values, color=colors, edgecolor="black")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(nodes, rotation=45, ha='right', rotation_mode='anchor', size=8)
    ax.set_xlabel("Nodes")
    ax.set_ylabel(f"{metric} Centrality")
    ax.set_title(f"{metric} Centrality ({time})")
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    return G


def cluster_samples_leiden(corr_types: pd.DataFrame, k: int = 10, res:float = 0.4, output_prefix: str = "samples"):
    """
    Cluster samples (rows) using a kNN graph + Leiden.

    Parameters
    ----------
    corr_types : pd.DataFrame
        Rows = samples. Non‑numeric columns (e.g. 'dataset','membership') will be ignored.
    k : int
        Number of nearest neighbours per sample.
    res : float
        Leiden resolution parameter.
    output_prefix : str
        (Currently unused) prefix for potential output artifacts.

    Returns
    -------
    igraph.Graph
        Graph with vertex attribute 'membership'.
    """
    import igraph
    import leidenalg
    from scipy.spatial.distance import pdist, squareform

    # --- 1. Copy & isolate numeric feature matrix ---------------------------
    corr_types = corr_types.copy()

    # Keep index (sample names)
    samples = corr_types.index.astype(str)

    # Select numeric columns only
    numeric_df = corr_types.select_dtypes(include=[np.number]).copy()

    dropped = set(corr_types.columns) - set(numeric_df.columns)
    if dropped:
        # Optional: inform user (silently skipping categorical/object columns)
        print(f"[cluster_samples_leiden] Ignoring non-numeric columns: {sorted(dropped)}")

    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric feature columns available for clustering.")

    # Replace ±inf with NaN then impute simple column means
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    if numeric_df.isna().all(axis=1).any():
        # Drop rows entirely NaN
        all_nan = numeric_df.isna().all(axis=1)
        print(f"[cluster_samples_leiden] Dropping {all_nan.sum()} samples with all-NaN features: {list(samples[all_nan])}")
        numeric_df = numeric_df.loc[~all_nan]
        samples = numeric_df.index.astype(str)

    # Impute remaining NaNs with column means
    if numeric_df.isna().any().any():
        col_means = numeric_df.mean(axis=0)
        numeric_df = numeric_df.fillna(col_means)

    # --- 2. Validate sample count / k ---------------------------------------
    n_samples = numeric_df.shape[0]
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples after cleaning; got {n_samples}.")
    if k >= n_samples:
        new_k = max(1, n_samples - 1)
        print(f"[cluster_samples_leiden] Reducing k from {k} to {new_k} (n_samples={n_samples}).")
        k = new_k

    # --- 3. Distance matrix -------------------------------------------------
    try:
        dist_mat = squareform(pdist(numeric_df.values, metric="euclidean"))
    except Exception as e:
        raise RuntimeError(
            "[cluster_samples_leiden] Failed computing pairwise distances. "
            f"Numeric dtypes: {numeric_df.dtypes.to_dict()}"
        ) from e

    # --- 4. Build kNN graph -------------------------------------------------
    g = igraph.Graph()
    g.add_vertices(list(samples))  # vertex names

    for i, si in enumerate(samples):
        # argsort distances; first is self (0 distance)
        order = np.argsort(dist_mat[i])
        neighbors = order[1:k+1]
        for j in neighbors:
            sj = samples[j]
            if not g.are_adjacent(si, sj):
                g.add_edge(si, sj)

    # --- 5. Leiden clustering -----------------------------------------------
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=res
    )
    g.vs["membership"] = part.membership

    # --- 6. 2D layout + plot (optional quick viz) ---------------------------
    layout_2d = g.layout_fruchterman_reingold()
    coords = np.array(layout_2d.coords)

    plt.figure(figsize=(6, 5))
    memberships = np.array(part.membership)
    unique_clusters = sorted(set(memberships))
    cmap = plt.get_cmap('tab20', len(unique_clusters))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=memberships, cmap=cmap, s=50)
    plt.title("Leiden Clusters")
    plt.colorbar(scatter, label="Cluster")

    for i, name in enumerate(samples):
        plt.text(coords[i, 0], coords[i, 1], str(name)[-6:], fontsize=7)

    plt.tight_layout()
    plt.show()

    return g
    # plt.savefig(f"{output_prefix}_leiden_clusters.png", bbox_inches="tight")
    # plt.close()

    # print("Done. Wrote plot to:", f"{output_prefix}_leiden_clusters.png")


def plot_degree_distribution_power_law(G):
    """
    Plots the degree distribution of graph G on two subplots:
      (1) normal scale histogram
      (2) log-log scatter + linear regression fit for power law
    
    Parameters
    ----------
    G : networkx.Graph
        The graph for which to plot the degree distribution.
    """
    # 1) Collect the degrees of all nodes.
    degrees = [deg for _, deg in G.degree()]
    # Build a histogram over integer bins.
    min_d, max_d = min(degrees), max(degrees)
    bins = np.arange(min_d, max_d + 2) - 0.5  # +1, but shift 0.5 for better alignment
    counts, bin_edges = np.histogram(degrees, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Midpoints for plotting

    # 2) Create two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Normal scale histogram
    ax1.bar(bin_centers, counts, width=1.0, color="skyblue", edgecolor="black")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Count")
    ax1.set_title("Degree Distribution (linear scale)")

    # (b) Log-log scatter plus power-law fit line
    # Filter out bins with zero counts to avoid log issues.
    nonzero_mask = counts > 0
    x_nonzero = bin_centers[nonzero_mask]
    y_nonzero = counts[nonzero_mask]

    ax2.scatter(x_nonzero, y_nonzero, color="blue", marker="o", s=25)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Degree (log scale)")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Degree Distribution (log-log)")

    # 3) Fit a line in log-log space to estimate power-law exponent.
    log_x = np.log10(x_nonzero)
    log_y = np.log10(y_nonzero)
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
    # line: log10(y) = intercept + slope * log10(x)
    # => y = 10^(intercept) * x^(slope)

    # Make a smooth line for plotting the fit
    line_logx = np.linspace(log_x.min(), log_x.max(), 100)
    line_logy = intercept + slope * line_logx

    ax2.plot(10**line_logx, 10**line_logy, color="red",
             label=f"Fit: y ~ x^({slope:.2f})")

    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_celltype_boxplot(
    corr_types: pd.DataFrame, 
    title: str=None,
    category_map = None, 
    cat_colors: dict = None,
    ax: plt.Axes = None
):
    """
    Creates a boxplot from the corr_types DataFrame. Each column is treated as a distinct cell type.
    Optionally color by a 'category_map' that maps cell types to categories, and color tick labels
    by a 'cat_colors' dict that maps categories to color values.
    """
    # Melt the DataFrame so each row is (sample, cell_type, abundance)
    melted = corr_types.melt(var_name='cell_type', value_name='abundance', ignore_index=False)
    melted.reset_index(inplace=True)
    melted.rename(columns={'index': 'sample'}, inplace=True)
    melted['abundance'] = np.log(melted['abundance'])

    # If we have categories for each cell type, add a 'cell_category' column
    if category_map:
        melted['cell_category'] = melted['cell_type'].map(category_map)
    else:
        melted['cell_category'] = 'Uncategorized'

    melted.sort_values(['cell_category', 'cell_type'], inplace=True)
    # Create the boxplot
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=melted, 
        x='cell_type', 
        y='abundance', 
        hue='cell_category',
        palette=cat_colors,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(f"Ln(Relative Abundance) {title}")
    ax.legend(title='Cell Category', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    

    # If a category-to-color mapping is provided, color the x‐tick labels
    if cat_colors and category_map:
        for tick_label in ax.get_xticklabels():
            cell_type = tick_label.get_text()
            category = category_map(cell_type)
            color = cat_colors.get(category, 'black')
            tick_label.set_color(color)

    if not ax:
        plt.show()


def fix_node_colors(plot, element):
    r = plot.state.node_renderer
    if 'color' in r.data_source.data:
        r.glyph.fill_color = 'color'  # Bokeh field name

# ── 1  tiny hook: rotate / nudge the two label columns ────────────────
def label_hook(offset=15, ang=(45, -45), scale=1.6):
    """Rotate, nudge, and scale (font size) the two label columns in a Sankey plot.

    Parameters
    ----------
    offset : int
        Horizontal shift applied to left vs right side labels.
    ang : tuple(int,int)
        (left_angle, right_angle) rotations in degrees.
    scale : float
        Factor to multiply current font size (e.g. 1.5 makes labels 50% larger).
    """
    return lambda p, _ : (
        lambda lbls:
            (lambda xs, mid:
                [
                    lbls[i].set_rotation(ang[x < mid]) or
                    lbls[i].set_position((x + (-1.6*offset, 1*offset)[x >= mid], y)) or
                    lbls[i].set_ha(('right','left')[x >= mid]) or
                    lbls[i].set_fontsize(lbls[i].get_fontsize()*scale)
                    for i, (x, y) in enumerate(map(lambda t: t.get_position(), lbls))
                ]
            )(xs := [t.get_position()[0] for t in lbls], (max(xs)+min(xs))/2)
    )(p.handles.get('labels', []))

# ── 2  minimal Sankey builder (no colour conversion) ──────────────────
def sankey(g, category, palette, *, title='', w=700, h=700,
           include_intra_self=True):
    """Bipartite Sankey: immune categories on left, (stromal|epithelial|tumoral) on right.

    Adds undirected edges only once (immune -> other). Optionally also
    represents intra-partition connectivity (immune-immune, other-other)
    as a self edge from a category to itself (category -> category),
    with cross-category intra edges split 0.5 to each side to avoid
    double counting.

    Parameters
    ----------
    g : networkx.Graph (undirected)
    category : callable(node)->str
      Maps raw node name to high-level category.
    palette : dict
      category -> color
    include_intra_self : bool
      If True, aggregate intra-partition edges into category self-links.
    """
    from collections import Counter
    from holoviews import Dataset

    def is_immune(cat: str) -> bool:
        return cat.startswith('immune')

    # Aggregate inter-partition flows and intra-partition edges
    inter_flows = Counter()
    intra_self = Counter()
    immune_cats, other_cats = set(), set()

    for u, v in g.edges():
        c1, c2 = category(u), category(v)
        i1, i2 = is_immune(c1), is_immune(c2)

        # Inter (immune vs other)
        if i1 and not i2:
            immune_cats.add(c1); other_cats.add(c2)
            inter_flows[(c1, c2)] += 1
        elif i2 and not i1:
            immune_cats.add(c2); other_cats.add(c1)
            inter_flows[(c2, c1)] += 1
        else:
            # Intra-partition (both immune or both other)
            # Distribute weight to category self counts (split if different cats)
            if c1 == c2:
                intra_self[c1] += 1
            else:
                intra_self[c1] += 0.5
                intra_self[c2] += 0.5
            if i1:  # both immune
                immune_cats.update([c1, c2])
            else:   # both other
                other_cats.update([c1, c2])

    if not inter_flows and not (include_intra_self and intra_self):
        raise ValueError("No inter-partition edges (immune ↔ other) to display.")

    # Build link records
    link_records = [
        {
            'source': s,
            'target': t,
            'value' : v,
            'color' : palette.get(s, 'gray')
        }
        for (s, t), v in inter_flows.items()
    ]

    if include_intra_self:
        for cat, v in intra_self.items():
            if v > 0:
                link_records.append({
                    'source': cat,
                    'target': cat,
                    'value' : v,
                    'color' : palette.get(cat, 'gray')
                })

    links_df = pd.DataFrame(link_records)

    # Node ordering: immune (left) then other (right)
    immune_list = sorted(immune_cats)
    other_list = sorted(other_cats)
    all_labels = immune_list + other_list

    nodes_df = pd.DataFrame({
        'label': all_labels,
        'color': [palette.get(c, 'gray') for c in all_labels],
        'side' : (['immune'] * len(immune_list)) + (['other'] * len(other_list))
    }, index=all_labels)
    nodes_df.index.name = 'index'

    nodes_ds = Dataset(nodes_df, kdims=['label'], vdims=['color', 'side'])

    # Ensure 'color' present in links dataframe for both backends
    if 'color' not in links_df.columns:
        links_df['color'] = links_df['source'].map({c: palette.get(c, 'gray') for c in all_labels})

    sk = hv.Sankey(
        (links_df[['source', 'target', 'value', 'color']], nodes_ds),
        kdims=['source', 'target'],
        vdims=['value', 'color']
    )

    if hv.Store.current_backend == 'bokeh':
        sk = sk.opts(
            title=title,
            labels='label',
            edge_color='color',
            node_fill_color='color',
            width=w, height=h,
            hooks=[label_hook()],
            bgcolor='white'
        )
    else:
        sk = sk.opts(
            title=title,
            labels='label',
            edge_color='color',
            node_color='color',
            fig_inches=w/150,
            aspect=h/float(w),
            hooks=[label_hook()],
            bgcolor='white'
        )
    return sk