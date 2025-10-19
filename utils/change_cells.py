import os
from pathlib import Path
from typing import Optional

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


def rename_cells(
    group_path: str,
    net_path: str,
    rename_map: dict,
    *,
    net_r: bool = True,
    group_r: bool = True,
    auc_r: bool = True,
    matrix_path: Optional[str] = None,
    matrix_r: bool = False,
):
    """
    Renames cell references in both the network text file and in the CSV file.
    The rename_map dictionary has old->new mappings for the cell names.
    """
    filepath =  os.path.dirname(os.path.abspath(__file__))
    os.chdir(filepath)
    os.chdir('..')

    if net_r:    
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

    if group_r:
        # 2) Rename in groups CSV (groups_I-II_leidenwu.csv)
        import pandas as pd
        df = pd.read_csv(f'{group_path}.csv')
        if 'cell_type_adjusted' in df.columns:
            df['cell_type_adjusted'] = df['cell_type_adjusted'].replace(rename_map)
        df.to_csv(f'{group_path}_funcnames.csv', index=False)

    if auc_r:
        # 2) Rename in groups CSV (groups_I-II_leidenwu.csv)
        import pandas as pd
        df = pd.read_csv(f'{auc_path}.csv', index_col=0)

        df['cell_type'] = None
        df['ds'] = None
        
        for i, name in enumerate(df.index):
            arr = name.split('_')
            if 'extended' in name or 'UKIM' in name:
                df.at[name, 'cell_type'] = '_'.join(arr[:-1])
                df.at[name, 'ds'] = '_'.join(arr[-1:])

            else:
                df.at[name, 'cell_type'] = '_'.join(arr[:-2])
                df.at[name, 'ds'] = '_'.join(arr[-2:])
        df['cell_type'] = df['cell_type'].replace(rename_map)
        df.index = df['cell_type'] + '_' + df['ds']
        df.drop(columns=['cell_type', 'ds'], inplace=True)

        df.to_csv(f'{auc_path}_funcnames.csv', index=True)

    if matrix_r and matrix_path is not None:
        matrix_file = Path(f"{matrix_path}.txt")
        if not matrix_file.exists():
            raise FileNotFoundError(f"Matrix file not found: {matrix_file}")

        with matrix_file.open('r', encoding='utf-8') as fh:
            lines = fh.readlines()

        updated_lines: list[str] = []
        for idx, line in enumerate(lines):
            if idx == 0:
                updated_lines.append(line)
                continue

            stripped = line.rstrip('\n')
            if stripped == "":
                updated_lines.append(line)
                continue

            parts = stripped.split('\t')
            if not parts:
                updated_lines.append(line)
                continue

            original_name = parts[0]
            parts[0] = rename_map.get(original_name, original_name)
            updated_lines.append('\t'.join(parts) + '\n')

        output_file = matrix_file.with_name(matrix_file.stem + "_funcnames.txt")
        with output_file.open('w', encoding='utf-8') as fh:
            fh.writelines(updated_lines)




# Example usage: call the rename function with our paths and dictionary
if __name__ == "__main__":
    time = 'I-II_leidenwu'
    time_solo = time.split('_')[0]
    rename_dict = rename_dict_early if 'I-II' in time else rename_dict_late

    group_path = f'metadata/groups_{time}'
    net_path = f'outputARACNE/net{time}'
    auc_path = f'nb_DE_wilcox/wilcoxon_DE/auc_count_cellphonedb_{time_solo}'
    matrix_path = f'outputARACNE/matrix{time}'

    rename_cells(
        group_path,
        net_path,
        rename_dict,
        group_r=False,
        net_r=False,
        matrix_path=matrix_path,
        matrix_r=True,
    )