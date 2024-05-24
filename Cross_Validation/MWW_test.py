from scipy.stats import ks_2samp, mannwhitneyu
import pandas as pd

folds=[1,2,3,4,5]

# For axial 
for fold in folds:

    print('fold:',fold)

    axial_all_data='../Data_Splitting/CSV/Axial_Lesion_information_for_80_percent_of_3D_MRI_Subject.csv'
    df1 = pd.read_csv(axial_all_data)
    data1 = df1['sum_voxel']



    axial_first_fold=f'./K_Fold_Lesion/Axial_K_Fold_CSV/Axial_Lesion_fold_0{fold}.csv'
    df2 = pd.read_csv(axial_first_fold)
    data2 = df2['sum_voxel']



    mw_stat, mw_p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    print(f'Mann-Whitney U test statistic:  p-value_for_axial: {mw_p_value}')



    Sagittal_all_data='../Data_Splitting/CSV/Sagittal_Lesion_information_for_80_percent_of_3D_MRI_Subject.csv'
    df1 = pd.read_csv(Sagittal_all_data)
    data1 = df1['sum_voxel']


    Sagittal_first_fold=f'./K_Fold_Lesion/Sagittal_K_Fold_CSV/Sagittal_Lesion_fold_0{fold}.csv'
    df2 = pd.read_csv(Sagittal_first_fold)
    data2 = df2['sum_voxel']



    mw_stat, mw_p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    print(f'Mann-Whitney U test statistic:  p-value_for_sagittal: {mw_p_value}')





    Coronal_all_data='../Data_Splitting/CSV/Coronal_Lesion_information_for_80_percent_of_3D_MRI_Subject.csv'
    df1 = pd.read_csv(Coronal_all_data)
    data1 = df1['sum_voxel']



    Coronal_first_fold=f'./K_Fold_Lesion/Coronal_K_Fold_CSV/Coronal_Lesion_fold_0{fold}.csv'
    df2 = pd.read_csv(Coronal_first_fold)
    data2 = df2['sum_voxel']



    mw_stat, mw_p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    print(f'Mann-Whitney U test statistic:  p-value_for_Coronal: {mw_p_value}')










