import numpy as np
import pandas as pd
import os

def create_top_bottom_csv(de_region, output_filename):
    """
    Creates a CSV file containing the top 10 and bottom 10 scores per cell type
    from the given de_region object.

    The output CSV will have columns: cell_type, gene, score, and rank_type (top or bottom).
    """
    data = []
    # Iterate over each cell type (field) in the recarrays
    cell_types = de_region["scores"].dtype.names
    for cell_type in cell_types:
        scores = np.array(de_region["scores"][cell_type])
        genes = np.array(de_region["names"][cell_type])
        
        # Create a DataFrame for this cell type
        df = pd.DataFrame({"gene": genes, "score": scores})
        
        # Get top 10 genes by score (largest scores)
        top10 = df.nlargest(10, "score")
        # Get bottom 10 genes by score (smallest scores)
        bottom10 = df.nsmallest(10, "score")
        
        # Add a column to indicate if a record is from top or bottom
        top10["rank_type"] = "top"
        bottom10["rank_type"] = "bottom"
        
        # Insert the cell type info
        top10["cell_type"] = cell_type
        bottom10["cell_type"] = cell_type
        
        data.append(top10)
        data.append(bottom10)
    
    # Concatenate all into a single DataFrame and reorder columns.
    final_df = pd.concat(data, ignore_index=True)
    final_df = final_df[["cell_type", "gene", "score", "rank_type"]]
    final_df.to_csv(output_filename, index=False)
    print(f"CSV saved to {output_filename}")


if __name__ == "__main__":
    # Simulate loading the de_region object as provided by fake_de_regions
    # Here we use np.core.records.fromarrays to build our test de_region.
    time = 'I-II'
    os.chdir('/root/host_home/luca/nb_DE_wilcox')
    de_region = np.load(f"wilcoxon_DE/{time}_averaged_tumorall.npy", allow_pickle=True).item()
    
    # Define the output CSV path
    output_csv = os.path.join(os.getcwd(), f"wilcoxon_DE/{time}_top_bottom_averaged_tumorall.csv")
    create_top_bottom_csv(de_region, output_csv)