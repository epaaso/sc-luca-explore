import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_name = 'gseapy_gsea/heatmap_onlynormal_I-IIless'

heat_df = pd.read_csv(f'{file_name}.csv', index_col=0)

plt.figure(figsize=(15, 10))
sns.heatmap(heat_df, cmap='viridis')
plt.title(f'Hallmarks Scores by Cell Type for only normal cells')
plt.xlabel('Hallmarks')
plt.ylabel('Cell Types')
plt.savefig(f'{file_name}.png')