import modal

app = modal.App("leiden-clustering-faiss-gpu")

# Define the container image with necessary dependencies
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:23.07-py3") # This one contains cuda 12.1.1 which is compatible with fiass-gpu 1.8.0
    .run_commands(
        [
            # Install Miniconda
            "wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.9.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh",
            "bash /tmp/miniconda.sh -b -p /opt/conda",
            "rm /tmp/miniconda.sh",
            # Initialize Conda
            "source /opt/conda/bin/activate",
            # Update Conda
            "/opt/conda/bin/conda update -y conda",
            # Add Conda to PATH
            "echo 'export PATH=/opt/conda/bin:$PATH' >> ~/.bashrc",
            # Install required packages using Conda
            "/opt/conda/bin/conda install -y python=3.11",
            "/opt/conda/bin/conda install -y -c nvidia -c conda-forge -c defaults "
            "anndata scanpy numpy scipy cupy",
            "/opt/conda/bin/conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0",
            # Clean up Conda packages
            "/opt/conda/bin/conda clean -ya",
        ]
    )
    .env({"PATH": "/opt/conda/bin:$PATH"})
    .copy_local_file(
        "/root/datos/maestria/netopaas/luca_explore/surgeries/Bishoff/query_latent.h5ad",
        "query_latent.h5ad",
    )
)

image2 = image.pip_install(
    [
        "modal-client",
    ]
).env({"PATH": "/opt/conda/bin:$PATH"}).copy_local_file(
        "/root/datos/maestria/netopaas/luca_explore/surgeries/Subcluster/query_latent.h5ad",
        "query_latent.h5ad",
    ).run_commands(
    [
        '/opt/conda/bin/conda install -y -c nvidia -c rapidsai cupy cudf cugraph',
        # "apt-get update",
        # "apt-get install -y libxau6 libxdmcp6 libxcb1 libx11-6 libxext6",
        # "rm -rf /var/lib/apt/lists/*",  # Clean up apt cache to reduce image size
        
    ]
    )

vol = modal.Volume.from_name("faiss-vol", create_if_missing=True)

# Specify that the function requires a GPU
@app.function(
    image=image2,
    gpu="any",  # Requests any available GPU
    timeout=6000,
    cpu=10,
    volumes={"/data": vol}
)
def leiden_clustering(nlist=1000, n_neighbors=30, resol=0.1):
    import anndata as ad
    # import scanpy as sc
    import numpy as np
    import cupy as cp
    import faiss
    import cudf
    import cugraph
    import sys
    # import matplotlib as plt
    from scipy.sparse import csr_matrix
    # Load your AnnData object
    adata = ad.read_h5ad("/query_latent.h5ad")  # Ensure this path is correct
    print(adata.shape)
    epit_types = ['Alveolar cell type 1', 'Alveolar cell type 2',  'ROS1+ healthy epithelial', 'transitional club/AT2', 'Club', 'Ciliated']
    epit_tumor_filter = np.logical_or(adata.obs.cell_type.isin(epit_types), adata.obs.cell_type.str.contains('Tumor'))
    adata = adata[epit_tumor_filter]
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
    # nlist = 1000  # Adjust as needed
    quantizer = faiss.IndexFlatL2(data.shape[1])
    index = faiss.IndexIVFFlat(quantizer, data.shape[1], nlist, faiss.METRIC_L2)
    index = faiss.index_cpu_to_gpu(res, 0, index)

    # Train the index
    index.train(data)

    # Add vectors to the index
    index.add(data)

    # Perform the search to get nearest neighbors
    # n_neighbors = 30  # Adjust as needed
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
                        'weights': weights
                       }),
        source='src',
        destination='dst',
        edge_attr='weights',
        renumber=False
    )

    # Compute layout on the GPU
    # pos_df = cugraph.force_atlas2(G)
    
    # # Visualize with matplotlib
    # plt.scatter(pos_df['x'], pos_df['y'], s=10, alpha=0.7)
    # plt.title("ForceAtlas2 Layout")
    # plt.savefig('/data/foo.png')

    # Perform Leiden clustering
    partitions_df, modularity = cugraph.leiden(G, resolution=resol)
    print("Modularity: ", modularity)

    # Map the cluster labels back to the original data
    partitions_df = partitions_df.to_pandas().set_index('vertex')
    adata.obs['leiden'] = partitions_df['partition'].astype(str).values
    print(len(adata.obs.leiden))
    return adata.obs

@app.local_entrypoint()
def main():
    leiden_df = leiden_clustering.remote(nlist=10, n_neighbors=20, resol=0.01)
    leiden_df.to_csv('atlas_leiden.csv')
