import os
import pandas as pd
from multiprocessing.pool import ThreadPool

import papermill as pm

infercnv_dir = '/root/datos/maestria/netopaas/infercnv/'
ikarus_dir = '/root/datos/maestria/netopaas/ikarus'

dsets = pd.read_csv(f'{ikarus_dir}/../metadata/dsets.csv')

# Function to be executed in parallel
def execute_infercnv(id_):
    try:
        pm.execute_notebook(
            f'nb_infercnv/infercnv_param.ipynb',
            f'nb_infercnv/{id_}.ipynb',
            parameters=dict(id_=id_)
        )
    except Exception as e:
        print(f"Error in {id_}: {e}")

    return id_

# Number of threads in the ThreadPool
# num_threads = max(dsets.id.size, 30)  # Adjust this number based on your system's capabilities
num_threads = 10
    
with ThreadPool(num_threads -1 ) as pool:
    for result in pool.imap_unordered(execute_infercnv, list(dsets.id)):
        id_ = result
        print(f'Finished or errored: {id_}')