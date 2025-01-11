if __name__ == "__main__":
    import anndata as ad
    # import scanpy as sc
    import numpy as np
    import cupy as cp
    import faiss
    import cudf
    import cugraph
    import sys
    from scipy.sparse import csr_matrix
    # Load your AnnData object
    adata = ad.read_h5ad("/query_latent.h5ad")  # Ensure this path is correct
    print(adata.shape)
    adata = ad.read_h5ad("/query_latent.h5ad")  # Ensure this path is correct
    epit_types = ['Alveolar cell type 1', 'Alveolar cell type 2',  'ROS1+ healthy epithelial', 'transitional club/AT2', 'Club', 'Ciliated']
    epit_tumor_filter = np.logical_or(ref_latent.obs.cell_type.isin(epit_types), ref_latent.obs.cell_type.str.contains('Tumor'))
    print(adata.shape)
    # Ensure data is in numpy array format
    if not isinstance(adata.X, np.ndarray):
        data = adata.X.toarray()
    else:
        data = adata.X

    # Convert data to float32
    data = data.astype(np.float32)

    # Build the FAISS index with GPU support
    res = faiss.StandardGpuResources()
    nlist = int(sys.argv[1]) if len(sys.argv) > 1 else 1000  # Adjust as needed
    quantizer = faiss.IndexFlatL2(data.shape[1])
    index = faiss.IndexIVFFlat(quantizer, data.shape[1], nlist, faiss.METRIC_L2)
    index = faiss.index_cpu_to_gpu(res, 0, index)

    # Train the index
    index.train(data)

    # Add vectors to the index
    index.add(data)

    # Perform the search to get nearest neighbors
    n_neighbors = int(sys.argv[2]) if len(sys.argv) > 2 else 30  # Adjust as needed
    distances, indices = index.search(data, n_neighbors + 1)  # +1 to include self

    # Convert indices and distances to cuDF DataFrames
    indices_df = cudf.DataFrame(indices[:, 1:])  # Exclude self-match at index 0
    distances_df = cudf.DataFrame(distances[:, 1:])

    # Create source and destination node arrays
    src = cudf.Series(
        np.repeat(np.arange(data.shape[0]), n_neighbors)
    )
    dst = indices_df.iloc[:,0].reset_index(drop=True)
    weights = distances_df.iloc[:,0].reset_index(drop=True)

    # Create a cuGraph Graph
    G = cugraph.Graph()
    G.from_cudf_edgelist(
        cudf.DataFrame({'src': src, 'dst': dst,
                        # 'weights': weights
                       }),
        source='src',
        destination='dst',
        # edge_attr='weights',
        renumber=False
    )

    # Compute layout on the GPU
    pos_df = cugraph.force_atlas2(G)
    
    # Visualize with matplotlib
    plt.scatter(pos_df['x'], pos_df['y'], s=10, alpha=0.7)
    plt.title("ForceAtlas2 Layout")
    plt.savefig('foo.png') 
# Perform Leiden clustering
    res = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    partitions_df, modularity = cugraph.leiden(G, resolution=0.1)
    print("Modularity: ", modularity)

    # Map the cluster labels back to the original data
    partitions_df = partitions_df.to_pandas().set_index('vertex')
    adata.obs['leiden'] = partitions_df['partition'].astype(str).values
    print(len(adata.obs.leiden.unique()))
