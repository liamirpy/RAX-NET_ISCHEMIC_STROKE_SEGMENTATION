

import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sns




def rainclound_plane(plane):
    df = pd.read_csv(f'./overall_results/{plane}_subjects_results.csv')

    # dice_values = df['mean_dice_before'].tolist()  # Convert 'dice' column to a list

    columns_to_plot = ['mean_dice_before', 'mean_dice_after','mean_precision_before','mean_precision_after','mean_recall_before','mean_recall_after']  # Replace with the names of the columns you want to plot
    pal = "Set2"  # Replace with desired color palette
    custom_labels = ['Dice before fusion', 'Dice after fusion', 'Precision before fusion', 'Precision before fusion','Recall before fusion','Recall before fusion']  # Custom labels for the columns

    df_melted = df[columns_to_plot].melt(var_name='Metric', value_name='Value')

    f, ax = plt.subplots(figsize=(14, 7))  

    pt.half_violinplot(x='Metric', y='Value', data=df_melted, palette=pal, bw=.2, cut=0.,
                    scale="area", width=.6, inner=None, orient="v", ax=ax)

    sns.stripplot(x='Metric', y='Value', data=df_melted, palette=pal, edgecolor="white",
                size=3, jitter=1, zorder=0, orient="v", ax=ax)

    sns.boxplot(x='Metric', y='Value', data=df_melted, color="black", width=.15, zorder=10,
                showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                saturation=1, orient="v", ax=ax)

    ax.set_title(f'{plane} Plane Metrics: Values Before and After Fusion')
    ax.set_ylabel('Metrics Value')
    ax.set_xlabel('Axial Plane')

    ax.set_xticks(range(len(custom_labels)))
    ax.set_xticklabels(custom_labels)

    # Step 10: Show Plot
    plt.tight_layout()
    plt.savefig(f'./plots/{plane}_plane_before_after_fusion.png', dpi=350)  

# plt.show()

rainclound_plane('axial')
rainclound_plane('sagittal')
rainclound_plane('coronal')






def rainclound_Fusion():
    df = pd.read_csv(f'./overall_results/fusion_subjects_results.csv')

    # dice_values = df['mean_dice_before'].tolist()  # Convert 'dice' column to a list

    columns_to_plot = ['dice', 'precision','recall']  # Replace with the names of the columns you want to plot
    pal = "Set2"  # Replace with desired color palette
    custom_labels = ['Dice', 'Precision','Recall']  # Custom labels for the columns

    df_melted = df[columns_to_plot].melt(var_name='Metric', value_name='Value')

    f, ax = plt.subplots(figsize=(14, 7))  

    pt.half_violinplot(x='Metric', y='Value', data=df_melted, palette=pal, bw=.2, cut=0.,
                    scale="area", width=.6, inner=None, orient="v", ax=ax)

    sns.stripplot(x='Metric', y='Value', data=df_melted, palette=pal, edgecolor="white",
                size=3, jitter=1, zorder=0, orient="v", ax=ax)

    sns.boxplot(x='Metric', y='Value', data=df_melted, color="black", width=.15, zorder=10,
                showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                saturation=1, orient="v", ax=ax)

    ax.set_title(f'Values of Metrics for 3D Subjects')
    ax.set_ylabel('Metrics Value')
    ax.set_xlabel('Axial Plane')

    ax.set_xticks(range(len(custom_labels)))
    ax.set_xticklabels(custom_labels)

    # Step 10: Show Plot
    plt.tight_layout()
    plt.savefig(f'./plots/values_of_metrics_for_3D_subjects.png', dpi=350)  

rainclound_Fusion()

