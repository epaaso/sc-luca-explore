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

import plotly.graph_objects as go
from collections import Counter

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


def plot_abundance_heatmap(corr_:pd.DataFrame, show_plot=True, cluster_samples=False):
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
    ax1.set_yticklabels(data.index, rotation=0, fontsize=6)
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
        ax.text((start + end)/2, y - bracket_height, text, ha='center', va='top', fontsize=10)

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


def cluster_samples_leiden(corr_types: pd.DataFrame, k: int = 10, output_prefix: str = "samples"):
    """
    Reads a 'corr_types' CSV (samples x features),
    builds a kNN graph, runs the Leiden algorithm,
    and plots a 2D layout of the graph colored by cluster.

    Args:
        corr_types_csv: Path to CSV. Rows = samples, columns = features.
        k: Number of nearest neighbors to connect per sample (default=10).
        output_prefix: Prefix for output plots/files.
    """
    import igraph
    import leidenalg
    corr_types = corr_types.copy()
    
    # 2) Compute distance matrix (Euclidean).
    from scipy.spatial.distance import pdist, squareform
    dist_mat = squareform(pdist(corr_types.values, metric="euclidean"))

    # 3) Build a kNN graph in igraph.
    g = igraph.Graph()
    samples = corr_types.index.tolist()
    g.add_vertices(samples)  # add one vertex per sample

    for i, sample_name in enumerate(samples):
        # Sort all samples by ascending distance to the i-th sample
        sorted_idxs = np.argsort(dist_mat[i])
        # Skip self-distance (index i), take the next k neighbors
        neighbors = sorted_idxs[1 : k + 1]
        for j in neighbors:
            # Add undirected edge (if not already existing)
            if not g.are_adjacent(samples[i], samples[j]):
                g.add_edge(samples[i], samples[j])

    # 4) Run Leiden with default parameters
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=0.6)
    # Membership vector: cluster assignment for each vertex
    membership = part.membership
    g.vs["membership"] = membership

    # 5) Create a 2D layout (e.g. Fruchterman–Reingold)
    layout_2d = g.layout_fruchterman_reingold()
    coords = np.array(layout_2d.coords)

    # 6) Plot with Matplotlib, coloring points by cluster
    plt.figure(figsize=(6, 5))
    
    unique_clusters = sorted(set(membership))
    cmap = plt.get_cmap('tab20', len(unique_clusters))
    scatter = plt.scatter(coords[:,0], coords[:,1], c=membership, cmap=cmap, s=50)
    plt.title("Leiden Clusters")
    plt.colorbar(scatter, label="Cluster")

    # Optionally label each sample
    for i, sample_name in enumerate(samples):
        plt.text(coords[i,0], coords[i,1], sample_name[-6:], fontsize=7)

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

def make_sankey_plot(g, category_map, color_map, title=None):
    """
    Creates a two-column Sankey diagram where the nodes are categories (A/B)
    and the flows are colored according to the source node's color.

    :param g: A graph with edges between nodes
    :param category_map: A function mapping each node to its category
    :param color_map: A dict mapping category name to a color code (e.g., "#FF0000")
    """
    # Build a list of unique categories
    categories = sorted({category_map(node) for node in g.nodes()})

    # Create repeated labels for two columns
    labels = [f"{cat} (A)" for cat in categories] + [f"{cat} (B)" for cat in categories]
    cat_index = {cat: i for i, cat in enumerate(categories)}

    # Count edges between categories
    edge_counts = Counter()
    for u, v in g.edges():
        source_cat = category_map(u)
        target_cat = category_map(v)
        edge_counts[(source_cat, target_cat)] += 1

    # Build the source, target, and value lists
    source = []
    target = []
    value = []
    offset = len(categories)
    for (src_cat, tgt_cat), count in edge_counts.items():
        source.append(cat_index[src_cat])           # left column
        target.append(cat_index[tgt_cat] + offset)  # right column
        value.append(count)

    # Assign colors to each 'A' and 'B' node
    node_colors = []
    for cat in categories:
        node_colors.append(color_map[cat])  # left-column color
    for cat in categories:
        node_colors.append(color_map[cat])  # right-column color

    # Color each flow based on its source node's color
    link_colors = [node_colors[s] for s in source]

    # Create the Sankey figure
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            x=[0]*len(categories) + [1]*len(categories),  # two columns
            y=[(i+0.5)/len(categories) for i in range(len(categories))] * 2
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])
    fig.update_layout(title_text=title, font_size=10)

    return fig
