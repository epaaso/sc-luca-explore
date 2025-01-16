import os
from multiprocessing import Pool

import pandas as pd

import papermill as pm

ikarus_dir = '/root/datos/maestria/netopaas/ikarus'

dsets = pd.read_csv(f'{ikarus_dir}/../metadata/dsets.csv')

from multiprocessing.pool import ThreadPool
# Function to be executed in parallel
def execute_ikarus(id_):
    try:
        pm.execute_notebook(
            'nb_ikarus/ikarus_param.ipynb',
            f'nb_ikarus/{id_}.ipynb',
            parameters=dict(id_=id_)
        )
    except Exception as e:
        print(f"Error in {id_}: {e}")
    return id_

# Number of threads in the ThreadPool
# num_threads = max(dsets.id.size, 30)  # Adjust this number based on your system's capabilities
num_threads = 20
    
with Pool(num_threads -1 ) as pool:
    for result in pool.imap_unordered(execute_ikarus, list(dsets.id)):
        id_ = result
        print(f'Finished or errored: {id_}')
        
        if os.path.exists(f'{ikarus_dir}/{id_}.csv'):
            preds = pd.read_csv(f'{ikarus_dir}/{id_}.csv')
            print(preds)
            # adatas[id_].obs['final_pred'] = preds.iloc[:,1]