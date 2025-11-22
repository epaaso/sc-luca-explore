import numpy as np
import pandas as pd
from collections import Counter
import json
from pathlib import Path

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import pearsonr
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
import seaborn as sns

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

import holoviews as hv

_CELL_METADATA_PATH = Path(__file__).resolve().parents[1] / "metadata" / "cell_mappings.json"

with _CELL_METADATA_PATH.open() as f:
    _CELL_METADATA = json.load(f)

# Annotate samples with therapy, Wu Zhou has 7 that are not annotated
samples_therapy = _CELL_METADATA["samples_therapy"]

# Define colors for different cell categories
color_map = _CELL_METADATA["color_map"]

# Define cell categories
cell_categories = _CELL_METADATA["cell_categories"]

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


def plot_abundance_heatmap(
    corr_: pd.DataFrame,
    show_plot: bool = True,
    cluster_samples: bool = False,
    xtick_params: dict = {},
    ylabel_size: int = 8,
    figsize: tuple = (22, 20),
    group_by_membership: bool = False,
):
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
    grouping_column = 'membership' if group_by_membership and 'membership' in corr.columns else 'dataset'
    if grouping_column not in corr.columns:
        raise ValueError(
            f"Grouping column '{grouping_column}' not found in corr dataframe columns: {corr.columns.tolist()}"
        )

    sort_columns = [grouping_column]
    if grouping_column != 'dataset' and 'dataset' in corr.columns:
        sort_columns.append('dataset')
    corr.sort_values(sort_columns, inplace=True)

    group_series = corr[grouping_column][::-1]

    # Create figure with adjusted gridspec to include colorbar space
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[1, 25, 0.3], height_ratios=[10, 1])

    ax0 = fig.add_subplot(gs[0, 0])  # Left brackets
    ax1 = fig.add_subplot(gs[0, 1])  # Heatmap
    ax_cbar = fig.add_subplot(gs[0, 2])  # Colorbar
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom brackets

    # Plot heatmap with colorbar in the dedicated axis
    columns_to_drop = {grouping_column}
    if 'dataset' in corr.columns:
        columns_to_drop.add('dataset')
    if 'membership' in corr.columns:
        columns_to_drop.add('membership')
    data = np.log(corr.drop(columns=list(columns_to_drop)))
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
    current_group = group_series.iloc[0]
    for i, group_value in enumerate(group_series):
        if group_value != current_group:
            positions.append((start, i - 1, current_group))
            start = i
            current_group = group_value
    positions.append((start, len(group_series) - 1, current_group))

    # Create a mapping of dataset names to colors
    unique_groups = group_series.unique()
    colors = plt.get_cmap('tab20', len(unique_groups))
    color_mapping = {group: colors(i) for i, group in enumerate(unique_groups)}

    # Draw brackets on ax0
    for (start, end, group_value) in positions:
        y_start = len(corr.index) - start
        y_end = len(corr.index) - end - 1
        ax0.plot([0, 0], [y_start, y_end], color=color_mapping[group_value], linewidth=2)
        label_text = str(group_value)
        if grouping_column == 'dataset':
            label_text = label_text[-18:]
        ax0.text(
            0,
            (y_start + y_end) / 2,
            label_text,
            va='center',
            ha='right',
            fontsize=16,
            rotation=60,
            rotation_mode='anchor'
        )


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
        ax.text((start + end)/2, y - bracket_height, text, ha='center', va='top', fontsize=18)

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
    pos = [pos.x0 -0.05, pos.y0, pos.width, pos.height]  # Shift the colorbar 0.05 units to the left
    ax_cbar.set_position(pos)
    ax_cbar.yaxis.label.set_size(20)
    ax_cbar.tick_params(axis='both', labelsize=18)
    
    
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
    xlabel_color:str = None,
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
    if xlabel_color:
        ax.yaxis.set_tick_params( labelcolor=xlabel_color)
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
def label_hook(offset:int=15, ang:tuple[int,int]=(45, -45),
                scale:float=1.6, intra_flows:dict[str,int] = {}):
    """Rotate, nudge, and scale (font size) the two label columns in a Sankey plot.

    Parameters
    ----------
    offset : int
        Horizontal shift applied to left vs right side labels.
    ang : tuple(int,int)
        (left_angle, right_angle) rotations in degrees.
    scale : float
        Factor to multiply current font size (e.g. 1.5 makes labels 50% larger).
    intra_flows: dict(str,int)
        Intra-category connections between nodes, if exists appends number of intra-cat 
        nodes to node label

    """
    return lambda p, _ : (
        lambda lbls:
            (lambda xs, mid, ls:
                [
                    lbls[i].set_rotation(ang[x < mid]) or
                    lbls[i].set_position((x + (-1.6*offset, 1*offset)[x >= mid], y)) or
                    lbls[i].set_ha(('right','left')[x >= mid]) or
                    lbls[i].set_fontsize(lbls[i].get_fontsize()*scale) or
                    lbls[i].set_text(
                        (ls[i] + '-' + str(intra_flows.get(ls[i],'0')),
                         lbls[i].get_text()
                        )[intra_flows=={}]
                         )

                    for i, (x, y) in enumerate(map(lambda t: t.get_position(), lbls))
                ]
            )(xs := [t.get_position()[0] for t in lbls], (max(xs)+min(xs))/2,
               ls := [t.get_text().split(' - ')[0] for t in lbls] )
    )(p.handles.get('labels', []))


def find_undirected_duplicates(graph):
    # Treat edges as undirected by normalizing (u,v) -> (min,max)
    if graph.is_multigraph():
        pairs = [(min(u, v), max(u, v)) for u, v, _ in graph.edges(keys=True)]
    else:
        pairs = [(min(u, v), max(u, v)) for u, v in graph.edges()]
    counts = Counter(pairs)
    return {e: c for e, c in counts.items() if c > 1}


# --- helper: draw clipped left→right gradient for each ribbon path ---
def _paint_gradients_on_ax(ax, paths, palette):
    if not paths:
        return

    # Gather label positions to determine which ribbon belongs to which nodes
    labels = []
    for txt in ax.texts:
        label = txt.get_text().strip()
        if not label:
            continue
        if label == ax.get_title():
            continue
        x, y = txt.get_position()
        base = label.split('-', 1)[0].strip()
        labels.append({'text': label, 'base': base, 'x': x, 'y': y})

    if not labels:
        return

    xs = np.array([lab['x'] for lab in labels])
    split_x = 0.5 * (xs.min() + xs.max())
    left_labels = [lab for lab in labels if lab['x'] <= split_x]
    right_labels = [lab for lab in labels if lab['x'] > split_x]

    if not right_labels:
        right_labels = [lab for lab in labels if lab['x'] >= split_x]

    if not left_labels or not right_labels:
        return

    def _lookup_color(base_label: str) -> np.ndarray:
        candidates = (
            base_label,
            base_label.strip(),
            base_label.lower(),
            base_label.title(),
            base_label.upper()
        )
        for cand in candidates:
            if cand in palette:
                return np.array(mcolors.to_rgba(palette[cand]))
        return np.array(mcolors.to_rgba('gray'))

    for path in paths:
        verts = getattr(path, 'vertices', None)
        if verts is None or len(verts) == 0:
            continue

        xs_path = verts[:, 0]
        ys_path = verts[:, 1]

        xmin = np.min(xs_path)
        xmax = np.max(xs_path)

        left_mask = np.isclose(xs_path, xmin, atol=1e-6)
        right_mask = np.isclose(xs_path, xmax, atol=1e-6)

        if not left_mask.any() or not right_mask.any():
            continue

        y_left = np.mean(ys_path[left_mask])
        y_right = np.mean(ys_path[right_mask])

        left_label = min(left_labels, key=lambda lab: abs(lab['y'] - y_left))
        right_label = min(right_labels, key=lambda lab: abs(lab['y'] - y_right))

        c1 = _lookup_color(left_label['base'])
        c2 = _lookup_color(right_label['base'])

        # small RGBA gradient image
        W, H = 256, 4
        t = np.linspace(0, 1, W)[None, :]
        grad = (1 - t)[..., None]*c1 + t[..., None]*c2
        grad = np.repeat(grad, H, axis=0)

        bbox = path.get_extents()
        xmin_bbox, xmax_bbox = bbox.xmin, bbox.xmax
        ymin_bbox, ymax_bbox = bbox.ymin, bbox.ymax
        if not np.isfinite([xmin_bbox, xmax_bbox, ymin_bbox, ymax_bbox]).all():
            continue
        if xmax_bbox <= xmin_bbox or ymax_bbox <= ymin_bbox:
            continue

        clip = mpatches.PathPatch(path, transform=ax.transData)
        im = ax.imshow(
            grad, origin='lower',
            extent=[xmin_bbox, xmax_bbox, ymin_bbox, ymax_bbox],
            interpolation='bilinear',
            aspect='auto',
            zorder=10000
        )
        im.set_clip_path(clip)


# ── 2  minimal Sankey builder (no colour conversion) ──────────────────
def sankey(g, category, palette, *, title='', w=700, h=700,
           holo_kws:dict={},
           include_intra_self=True,
           gradients: bool = True,
           return_figure: bool | None = None):
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
    holo_kws: dict
      params to pass to the hv.Sankey().opts() function
    include_intra_self : bool
      If True, aggregate intra-partition edges as number after label of node.
    """
    from holoviews import Dataset

    if find_undirected_duplicates(g):
        raise Exception('Graph must not contain undirected duplicates')
    
    backend = hv.Store.current_backend
    is_matplotlib = backend == 'matplotlib'

    if return_figure and not is_matplotlib:
        raise ValueError("return_figure is only supported when the 'matplotlib' backend is active.")

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
            # else:
            #     intra_self[c1] += 0.5
            #     intra_self[c2] += 0.5
            if i1 and i2:  # both immune
                immune_cats.update([c1, c2])
            else:   # both other
                other_cats.update([c1, c2])

    if not inter_flows and not (include_intra_self and intra_self):
        raise ValueError("No inter-partition edges (immune ↔ other) to display.")

    # Build link records
    link_records = []
    for i, ((s, t), v) in enumerate(inter_flows.items()):
        record = {
            'source': s,
            'target': t,
            'value' : v,
            'color': ('#00000000' if gradients and is_matplotlib else palette.get(s, 'gray')),
            'c_left':  palette.get(s, 'gray'),
            'c_right': palette.get(t, 'gray'),
            'eid': i,
        }
        link_records.append(record)

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

    
    if backend == 'bokeh':
        sk = hv.Sankey(
            (links_df[['source', 'target', 'value', 'color']], nodes_ds),
            kdims=['source', 'target'],
            vdims=['value', 'color']
        )
        if not include_intra_self:
            intra_self = {}
        sk = sk.opts(
            title=title,
            labels='label',
            edge_color='color',
            node_fill_color='color',
            width=w, height=h,
            bgcolor='white',
            **holo_kws
        )
    else:
        hooks = [label_hook(intra_flows=intra_self)]

        def _gradient_hook(p, _):
            ax = p.handles.get('axis')
            if ax is None:
                axes = p.handles.get('axes')
                if axes:
                    ax = axes[0]
            if ax is None:
                return []

            paths = []
            for child in ax.get_children():
                if isinstance(child, mpatches.PathPatch):
                    try:
                        child.set_facecolor((0, 0, 0, 0))
                        child.set_edgecolor((0, 0, 0, 0))
                    except Exception:
                        pass
                    paths.append(child.get_path())

            if not paths:
                for coll in ax.collections:
                    if isinstance(coll, PolyCollection) and hasattr(coll, 'get_paths'):
                        try:
                            coll.set_facecolor([(0, 0, 0, 0)] * len(coll.get_paths()))
                            coll.set_edgecolor([(0, 0, 0, 0)] * len(coll.get_paths()))
                        except Exception:
                            pass
                        paths.extend(list(coll.get_paths()))

            _paint_gradients_on_ax(ax, paths, palette)
            return []

        if gradients and is_matplotlib:
            hooks.append(_gradient_hook)

        hv_sk = hv.Sankey(
            (links_df[['source','target','value','color','c_left','c_right','eid']], nodes_ds),
            kdims=['source','target'],
            vdims=['value','color','c_left','c_right','eid']
        ).opts(
            title=title,
            labels='label',
            node_color='color',
            edge_color='color',
            edge_alpha=0.0 if (gradients and is_matplotlib) else 1.0,
            edge_linewidth=0.0 if (gradients and is_matplotlib) else 1.0,
            hooks=hooks,
            bgcolor='white',
            **holo_kws
        )

        if return_figure:
            return hv.render(hv_sk, backend='matplotlib')

        return hv_sk
        
    return sk