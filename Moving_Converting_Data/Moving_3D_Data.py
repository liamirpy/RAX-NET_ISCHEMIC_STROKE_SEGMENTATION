### This is LIAMIRPY from the land of loneliness. 

## In this code we want to split the data from the original folder of data set to Test and Train folder 


'''
        Procedure:
    
    1- Split the data for 3D mode:

        - 20% for test data
        -80% for train data
    



'''




import os



import csv

import pandas as pd 


import re 

import shutil

import numpy as np


import nibabel as nib


import os 






ATLAS_Train_directory='../Dataset/ATLAS_2/Training'


Train_csv = '../Data_Splitting/CSV/Lesion_information_for_80_percent_of_3D_MRI_Subject.csv'


Fusion_evaluation_csv='../Data_Splitting/CSV/Lesion_information_for_20_percent_of_3D_MRI_Subject.csv'


Train_folder_mri='Train_3D_Data/mri'


Fusion_evaluation_folder_mri='Fusion_evaluation_3D_Data/mri'


Train_folder_mask='Train_3D_Data/mask'


Fusion_evaluation_folder_mask='Fusion_evaluation_3D_Data/mask'




def moving_3D_Data(train_csv,test_csv,train_folder_mri,test_folder_mri,train_folder_mask,test_folder_mask):

    pattern = r'r(\w{3})'


    with open(train_csv, 'r') as csvfile:


        reader = csv.reader(csvfile)


        next(reader)


        for row in reader:

            print(row[0])


            subject_name=row[0]


            match = re.search(pattern, subject_name)

            # Check if a match is found
            if match:
                # Extract the captured group and convert it to uppercase
                main_folder = match.group(0).upper()
                # print(result)
            else:
                print("No match found.")



            mri_file= f'{ATLAS_Train_directory}/{main_folder}/{subject_name}/ses-1/anat/{subject_name}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz'

            mask_file= f'{ATLAS_Train_directory}/{main_folder}/{subject_name}/ses-1/anat/{subject_name}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'


            shutil.copy(mri_file, train_folder_mri)

            shutil.copy(mask_file, train_folder_mask)







    with open(test_csv, 'r') as csvfile:


        reader = csv.reader(csvfile)


        next(reader)


        for row in reader:

            print(row[0])


            subject_name=row[0]


            match = re.search(pattern, subject_name)

            # Check if a match is found
            if match:
                # Extract the captured group and convert it to uppercase
                main_folder = match.group(0).upper()
                # print(result)
            else:
                print("No match found.")



            mri_file= f'{ATLAS_Train_directory}/{main_folder}/{subject_name}/ses-1/anat/{subject_name}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz'

            mask_file= f'{ATLAS_Train_directory}/{main_folder}/{subject_name}/ses-1/anat/{subject_name}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'


            shutil.copy(mri_file, test_folder_mri)

            shutil.copy(mask_file, test_folder_mask)





moving_3D_Data(Train_csv,Fusion_evaluation_csv,Train_folder_mri,Fusion_evaluation_folder_mri,Train_folder_mask,Fusion_evaluation_folder_mask)
