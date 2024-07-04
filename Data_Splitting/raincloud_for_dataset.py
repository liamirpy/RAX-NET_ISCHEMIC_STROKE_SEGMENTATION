import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt  # Import plotnine for half_violinplot

# Load data from CSV files
df2 = pd.read_csv('Lesion_information_for_80_percent_of_3D_MRI_Subject.csv')
df3 = pd.read_csv('Lesion_information_for_20_percent_of_3D_MRI_Subject.csv')
df1 = pd.read_csv('ATLAS_Lesion_information_for_3D_MRI_Subject.csv')

# Concatenate or merge relevant columns into a single DataFrame
df_combined = pd.concat([df1[['sum_voxel']], df2[['sum_voxel']], df3[['sum_voxel']]], axis=1)
df_combined.columns = ['655 Cases', '524 Cases', '112 Cases']  # Rename columns if needed

# Melt the DataFrame for seaborn compatibility
df_melted = df_combined.melt(var_name='Metric', value_name='Value')

# Create Subplots
f, ax = plt.subplots(figsize=(14, 7))

# Plot Half Violin Plot
pt.half_violinplot(x='Metric', y='Value', data=df_melted, palette="Set2", bw=.2, cut=0.,
                   scale="area", width=.6, inner=None, orient="v", ax=ax)

# Plot Strip Plot
sns.stripplot(x='Metric', y='Value', data=df_melted, palette="Set2", edgecolor="white",
              size=3, jitter=1, zorder=0, orient="v", ax=ax)

# Plot Box Plot
sns.boxplot(x='Metric', y='Value', data=df_melted, color="black", width=.15, zorder=10,
            showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
            showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
            saturation=1, orient="v", ax=ax)

# Customize Plot
ax.set_title('')
ax.set_xlabel('Metric')
ax.set_ylabel('Value')

# Customize Tick Labels
ax.set_xticklabels(['655 Cases', '524 Cases', '112 Cases'])

# Show Plot
plt.tight_layout()
plt.savefig(f'./LESION_VOXEL_DISTRIBUTION.png', dpi=650)  

plt.show()
