import modal
import anndata as ad
import pandas as pd
import numpy as np
import scvi

app = modal.App("DE - scvi")

# Specify your local data directory
local_data_dir = '/root/datos/maestria/netopaas/luca_explore/surgeries/Bishoff'

# Define the remote path where the data will be available in the remote function
remote_data_dir = "/data"

# Paths within the mounted directory
ext_name = "Deng_Liu_LUAD_2024"
name = 'Bishoff'
surgery_path = remote_data_dir

scvi_image = modal.Image.from_registry("ghcr.io/scverse/scvi-tools:py3.11-cu12-1.2.x-runtime")\
    .pip_install(
        "pandas==1.5.3",
    )\
    .copy_local_dir(local_data_dir, remote_data_dir)\
    # .copy_local_file(f'/root/datos/maestria/netopaas/luca_explore/surgeries/{name}_predicted_leiden.csv',
    # f'{remote_data_dir}/')


# Remote function with GPU and mounted local directory
@app.function(
    image=scvi_image,
    gpu='any',
    timeout=6000,
    cpu=10,
#    mounts=[
#        modal.Mount.from_local_dir(
#            local_path=local_data_dir,
#            remote_path=remote_data_dir
#        )
#    ]
)
def get_de():
    print("This code is running on a remote worker!")
    import torch
    import anndata as ad
    import pandas as pd
    import scvi
    import subprocess

    has_cuda = torch.cuda.is_available()
    print(f"It is {has_cuda} that torch can access CUDA")

    # Load the data and model from the mounted directory
    adata_query = ad.read_h5ad(f'{surgery_path}/query.h5ad')
    adata_query.obs['dataset'] = adata_query.obs['sample']
    model = scvi.model.SCANVI.load(surgery_path, adata_query)

    subprocess.check_call(['pip', 'install', '--upgrade', 'pandas'])

    # Prepare indices
    preds = pd.read_csv(f'{remote_data_dir}/{name}_predicted_leiden.csv')
    adata_query.obs['cell_type_adjusted'] = preds['cell_type_adjusted'].values
    cell_type_1 = "Tumor LUAD_LUSC"
    cell_idx1 = adata_query.obs["cell_type_adjusted"] == cell_type_1
    cell_type_2 = "Tumor LUAD_mixed"
    cell_idx2 = adata_query.obs["cell_type_adjusted"] == cell_type_2

    print(adata_query.obs['cell_type_adjusted'].value_counts())
    adata_query.obs['tumor_rest'] = adata_query.obs['cell_type_adjusted']

    cell_not_tumor = np.logical_not(adata_query.obs['cell_type_adjusted'].str.contains('Tumor'))

    adata_query.obs.loc[cell_not_tumor,'tumor_rest'] = 'Not Tumor'
    adata_query.obs['tumor_rest'].astype('category')

    # Perform differential expression analysis
    de_change_importance = model.differential_expression(
        idx1=cell_idx2,
        groupby='tumor_rest',
#        idx2=cell_idx2,
        weights="importance",
        filter_outlier_cells=False,
        batch_correction=True,
        delta=None
    )

    # Save the results to the mounted directory
    output_path = f'{surgery_path}/{name}_de_results.csv'
    # de_change_importance.to_csv(output_path)
    print(f"Differential expression analysis completed and results saved to {output_path}.")
    return de_change_importance

# Main entry point
@app.local_entrypoint()
def main():
    # Run the remote function
    print("Starting differential expression analysis on the remote worker...")
    result = get_de.remote()
    print("Differential expression analysis result:")
    print(result)
    result.to_csv(f'{name}_de_results.csv')
