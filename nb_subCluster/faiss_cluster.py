import modal
import subprocess

subluster_path = '/root/datos/maestria/netopaas/luca_explore/surgeries/Subcluster'

app = modal.App("leiden-clustering-faiss-gpu")

# Define the container image with necessary dependencies
image = (
    modal.Image.from_registry("netopaas/faiss:cugraph-24-12") # This one contains cuda 12.1.1 which is compatible with faiss-gpu 1.8.0
    # .run_commands( # wE LEAVE THIS HERE TO SHOW HOW TO INSTALL FAISS-GPU
    #     [
    #         # Install Miniconda
    #         "wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.9.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh",
    #         "bash /tmp/miniconda.sh -b -p /opt/conda",
    #         "rm /tmp/miniconda.sh",
    #         "source /opt/conda/bin/activate",
    #         "/opt/conda/bin/conda update -y conda",
    #         "echo 'export PATH=/opt/conda/bin:$PATH' >> ~/.bashrc",
    #         "export PATH=/opt/conda/bin:$PATH",

    #         # Install required packages using Conda
    #         "/opt/conda/bin/conda install -y python=3.11",
    #         "/opt/conda/bin/conda install -y -c nvidia -c conda-forge -c defaults "
    #         "anndata scanpy numpy scipy",
    #         '/opt/conda/bin/conda install -y -c nvidia -c rapidsai cupy cudf cugraph==24.12',

            # '/opt/conda/bin/conda config --set solver classic',
    #         "/opt/conda/bin/conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0",
            
    #         '/opt/conda/bin/conda install -y -c nvidia -c rapidsai cuml',

    #         # Clean up Conda packages
    #         "/opt/conda/bin/conda clean -ya",
    #     ]
    # )
    .env({"PATH": "/opt/conda/bin:$PATH"})
    .run_commands(
    [
        # Strange behaviour because of using conda and pip, but there is no modal installer for conda
        'pip install --upgrade google-api-python-client',
        'pip install modal-client',
        '/opt/conda/bin/conda install -y grpclib'

    ]
    )
)

# upload to volume with modal volume put faiss-vol <path>
vol = modal.Volume.from_name("faiss-vol", create_if_missing=True)

# Specify that the function requires a GPU
@app.function(
    image=image,
    gpu="any",  # Requests any available GPU
    timeout=6000,
    cpu=10,
    volumes={"/data": vol}
)
def leiden_clustering(nlist=1000, n_neighbors=30, resol=0.1, nprobe= 50,
                    umap_params = None,
                    file_path='/data/query_latent_tumor.h5ad'):
    import anndata as ad
    # import scanpy as sc
    import numpy as np
    import cupy as cp
    from cupyx.scipy.sparse import coo_matrix
    import faiss
    import cudf
    import cugraph
    from cuml.manifold import UMAP

    print('cugraph ver:')
    print(cugraph.__version__)
    from scipy.sparse import csr_matrix
    # Load your AnnData object
    adata = ad.read_h5ad(file_path)  # Ensure this path is correct
    
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
    index.nprobe = nprobe  # More nprobe means more accurate search
    distances, indices = index.search(data, n_neighbors + 1)  # +1 to include self

    # Convert indices and distances to cuDF DataFrames
    indices_df = cudf.DataFrame(indices[:, 1:])  # Exclude self-match at index 0
    distances_df = cudf.DataFrame(distances[:, 1:])


    # Create source and destination node arrays
    src = cudf.Series(
        np.repeat(np.arange(data.shape[0]), n_neighbors)
    )

    # Flatten the indices and distances arrays to enusre all nodes are connected
    dst = indices_df.values.reshape(-1)
    weights = distances_df.values.reshape(-1)

    # Create a cuGraph Graph
    G = cugraph.Graph()

    knn_cusparse = cudf.DataFrame({'src': src, 'dst': dst,
                        'weights': weights
                       })
    
    G.from_cudf_edgelist(
        knn_cusparse,
        source='src',
        destination='dst',
        edge_attr='weights',
        renumber=False
    )


    ######## UMAP PART ###########
    data_cudf = cudf.DataFrame(data)

    if umap_params is None:
        umap_params = {
        "n_neighbors": 20,
        "n_components": 2,
        "min_dist": 0.1,
        "spread": 1.0,
        "random_state": 42,
        "method": "cuml-umap",
        }
    
    # Instantiate UMAP
    umap_model = UMAP(
        **umap_params
    )

    num_samples = data.shape[0]
    # rows = cp.repeat(cp.arange(num_samples), indices_df.shape[1])
    rows = cp.asarray(src)
    cols = cp.asarray(dst)
    vals = cp.asarray(weights)
    knn_sparse = coo_matrix((vals, (rows, cols)), shape=(num_samples, num_samples)).tocsr()
    

    # Fit-transform using the precomputed kNN
    #   - 'knn_indices' and 'knn_dists' must be passed here,
    #     so that UMAP won't recompute neighbors.
    umap_emb = umap_model.fit_transform(
        X=data_cudf,
        knn_graph=knn_sparse,
    )

    # 1. Convert the embedding to a NumPy array so we can store it in adata:
    umap_emb_host = cudf.DataFrame.to_numpy(umap_emb)

    # 2. Insert the embedding into adata.obsm:
    adata.obsm["X_umap"] = umap_emb_host

    # Optionally, you can also store the UMAP parameters in adata.uns for reference:
    adata.uns["umap_params"] = umap_params


    ############# Perform LEIDEN clustering #######################################
    partitions_df, modularity = cugraph.leiden(G, resolution=resol, max_iter=1000)
    print("Modularity: ", modularity)

    # Map the cluster labels back to the original data
    partitions_df = partitions_df.to_pandas().set_index('vertex')
    adata.obs['leiden'] = partitions_df['partition'].astype(str).values

    print('NUMCLUSTERS:')
    print(len(adata.obs.leiden.unique()))

    print('UPARAMS')
    print(umap_params)


    return adata.obs, umap_emb_host, umap_params

@app.local_entrypoint()
def main():
    import json
    import pandas as pd    

    # file_path = '/data/query_latent_tumor_early.h5ad'
    file_path = '/data/query_latent_tumor_late.h5ad'
    suffix = file_path.split('/')[-1].split('.')[0].split('_')[-1]

    umap_params = {
        "n_neighbors": 30,
        "n_components": 2,
        "min_dist": 0.03,
        "spread": 0.6,
        "random_state": 42,
        "method": "cuml-umap",
    }

    resol = 0.5

    # n_probe <= nlist means more accurate neighbours
    leiden_df, umap_emb, umap_params = leiden_clustering.remote(nlist=1, resol=resol, nprobe=1,
                                                                umap_params=umap_params, file_path=file_path)
    umap_params['leiden_resol'] = resol

    leiden_df.to_csv(f'{subluster_path}/atlas_{suffix}_leiden.csv')
    pd.DataFrame(umap_emb).to_csv(f'{subluster_path}/atlas_{suffix}_umap.csv')
    
    with open(f'{subluster_path}/atlas_{suffix}_uparams.json', 'w') as fp:
        json.dump(umap_params, fp)
