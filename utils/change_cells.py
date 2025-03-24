import os

# ...existing code...
time = 'III-IV_leidenwu'


group_path = f'metadata/groups_{time}'
net_path = f'outputARACNE/net{time}'

rename_dict_early = {
    "Tumor Club_AT2_LUAD": "Tumor Secretory_LUAD",
    "Tumor LUAD": "Tumor Adeno_Squamous",
    "Tumor LUAD_AT1/2": "Tumor EMT_lowProlif",
    "Tumor LUAD_Club_AT2": "Tumor OxPhos_Prolif",
    "Tumor LUAD_Club_AT2/1": "Tumor TTF1+_Quiescent",
    "Tumor LUAD_NE_LUSC": "Tumor Evasive_Prolif",
    "Tumor LUAD_NE_mitotic": "Tumor Evasive_HighProlif",
    "Tumor LUAD_mitotic": "Tumor Hypoxic_Prolif",
    "Tumor LUSC": "Tumor Macrophage_Swallow",
    "Tumor LUSC_LUAD_mitotic": "Tumor NSCLC_Prolif"
}

rename_dict_late = {
    "Tumor AT2/1_LUAD": "Tumor Secretory_TAM",
    "Tumor Club/AT2_MSLN": "Tumor MSLN_Warburg_TherResist",
    "Tumor LUAD2": "Tumor Prolif_Stem",
    "Tumor LUAD_EMT": "Tumor EMT_lncRNA_Prolif",
    "Tumor LUAD_LUSC_mitotic": "Tumor NSCLC_HighProlif",
    "Tumor LUAD_MSLN_LUSC": "Tumor NSCLC_Metabolic",
    "Tumor LUAD_ROS1+": "Tumor Differentiated_DrugResp",
    "Tumor LUAD_mitotic": "Tumor Differentiated_Prolif",
    "Tumor LUSC": "Tumor Macrophage_Swallow",
    "Tumor LUSC_LUAD_NE": "Tumor LUSC_Hypoxia",
    "Tumor LUSC_mitotic_NE": "Tumor HighProlif_ImmuneCold",
    "Tumor NSCLC_mixed": "Tumor NSCLC_Basal_Inflammatory"
}

if "I-II" in time:
    rename_dict = rename_dict_early
else:    
    rename_dict = rename_dict_late


def rename_cells(group_path: str, net_path: str, rename_map: dict):
    """
    Renames cell references in both the network text file and in the CSV file.
    The rename_map dictionary has old->new mappings for the cell names.
    """
    filepath =  os.path.dirname(os.path.abspath(__file__))
    os.chdir(filepath)
    os.chdir('..')

    # 1) Rename in net file (netI-II_leidenwu.txt)
    with open(f'{net_path}.txt', 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        for old_name, new_name in rename_map.items():
            line = line.replace(f'{old_name}\t', f'{new_name}\t')
        updated_lines.append(line)

    with open(f'{net_path}_funcnames.txt', 'w') as f:
        f.writelines(updated_lines)

    # 2) Rename in groups CSV (groups_I-II_leidenwu.csv)
    import pandas as pd
    df = pd.read_csv(f'{group_path}.csv')
    if 'cell_type_adjusted' in df.columns:
        df['cell_type_adjusted'] = df['cell_type_adjusted'].replace(rename_map)
    df.to_csv(f'{group_path}_funcnames.csv', index=False)


# Example usage: call the rename function with our paths and dictionary
if __name__ == "__main__":
    rename_cells(group_path, net_path, rename_dict)