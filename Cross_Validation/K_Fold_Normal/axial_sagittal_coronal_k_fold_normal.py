### This is LIAMIRPY from the land of loneliness. 
### In this code we split the data for the K-fold validation for normal slice 








'''
            Approach:

        1- First we read each fold for lesion (csv file )
        2- Finding the subject that are in that folds 
        3- Finding the normal slice in each subject
        4- Adding the same number of normal slice near to lesion slice from same subject

      
'''


import csv

import numpy as np

import pandas as pd 

import random

import nibabel as nib


random.seed(42)


'''

min_axial 2
max_axial 179

min_sagittal 20
max_sagittal 176


min_coronal 27
min_coronal 227




'''


number_of_folds=5


train_folder_mask='../../Moving_Converting_Data/Train_3D_Data/mask'


axial_folds_csv='../K_Fold_Lesion/Axial_K_Fold_CSV'

sagittal_folds_csv='../K_Fold_Lesion/Sagittal_K_Fold_CSV'

coronal_folds_csv='../K_Fold_Lesion/Coronal_K_Fold_CSV'




def spliting_noramal_slice(axial_folds_csv,sagittal_folds_csv,coronal_folds_csv,train_folder_mask):



    for axial_fold in range(1,number_of_folds+1):

        print('fold',axial_fold)


        subjects_in_this_fold=[]

        number_of_slice_for_each_subject={}

        normal_slice=[]

        number_of_slice_in_lesion_fold=0



        with open(f'{axial_folds_csv}/Axial_Lesion_fold_0{axial_fold}.csv', 'r') as csvfile:


            reader = csv.reader(csvfile)


            next(reader)


            for row in reader:

                number_of_slice_in_lesion_fold +=1

                subject_name=row[0]

                if subject_name not in subjects_in_this_fold:
                    subjects_in_this_fold.append(subject_name)
                
                # try:
                #     number_of_slice_for_each_subject[subject_name] +=1
                # except:
                #     number_of_slice_for_each_subject[subject_name] =1 
        

        ## Read each subject 
                    

                    
        for subject in subjects_in_this_fold:


            mask_nii_nii=nib.load(f'{train_folder_mask}/{subject}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')
            mask_nii = mask_nii_nii.get_fdata()

            for ax in range(mask_nii.shape[2]):
                single_mask_slice=mask_nii[:,:,ax]
                single_mask_slice[single_mask_slice > 0] = 1

                if sum(sum(single_mask_slice)) ==0:

                    normal_slice.append({'subject_name':subject,'slice':ax})

        

        ## Choose the number of random from the normal slice and add to that fold 
        random.shuffle(normal_slice)

        num_elements_to_choose = number_of_slice_in_lesion_fold

        # Choose elements randomly from different parts of the list
        chosen_indices = random.sample(range(len(normal_slice)), num_elements_to_choose)
        chosen_elements = [normal_slice[i] for i in chosen_indices]




        ### save the csv file 



    # CSV file path
        csv_file_path = f'Axial_K_Fold_CSV/Axial_Normal_fold_0{axial_fold}.csv'

        # Writing data to CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = chosen_elements[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Writing header
            writer.writeheader()

            # Writing data
            writer.writerows(chosen_elements)

        print(f'Data has been written to {csv_file_path}.')









    for coronal_fold in range(1,number_of_folds+1):

        print('fold',coronal_fold)


        subjects_in_this_fold=[]

        number_of_slice_for_each_subject={}

        normal_slice=[]

        number_of_slice_in_lesion_fold=0



        with open(f'{coronal_folds_csv}/Coronal_Lesion_fold_0{coronal_fold}.csv', 'r') as csvfile:


            reader = csv.reader(csvfile)


            next(reader)


            for row in reader:

                number_of_slice_in_lesion_fold +=1

                subject_name=row[0]

                if subject_name not in subjects_in_this_fold:
                    subjects_in_this_fold.append(subject_name)
                
                # try:
                #     number_of_slice_for_each_subject[subject_name] +=1
                # except:
                #     number_of_slice_for_each_subject[subject_name] =1 
        

        ## Read each subject 
                    

                    
        for subject in subjects_in_this_fold:


            mask_nii_nii=nib.load(f'{train_folder_mask}/{subject}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')
            mask_nii = mask_nii_nii.get_fdata()

            for co in range(  mask_nii.shape[2] ):
                single_mask_slice=mask_nii[:,co,:]
                single_mask_slice[single_mask_slice > 0] = 1

                if sum(sum(single_mask_slice)) ==0:

                    normal_slice.append({'subject_name':subject,'slice':co})

        

        ## Choose the number of random from the normal slice and add to that fold 
        random.shuffle(normal_slice)

        num_elements_to_choose = number_of_slice_in_lesion_fold

        # Choose elements randomly from different parts of the list
        chosen_indices = random.sample(range(len(normal_slice)), num_elements_to_choose)
        chosen_elements = [normal_slice[i] for i in chosen_indices]




        ### save the csv file 



    # CSV file path
        csv_file_path = f'Coronal_K_Fold_CSV/Coronal_Normal_fold_0{coronal_fold}.csv'

        # Writing data to CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = chosen_elements[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Writing header
            writer.writeheader()

            # Writing data
            writer.writerows(chosen_elements)

        print(f'Data has been written to {csv_file_path}.')




    for sagittal_fold in range(1,number_of_folds+1):

        print('fold',sagittal_fold)


        subjects_in_this_fold=[]

        number_of_slice_for_each_subject={}

        normal_slice=[]

        number_of_slice_in_lesion_fold=0



        with open(f'{sagittal_folds_csv}/Sagittal_Lesion_fold_0{sagittal_fold}.csv', 'r') as csvfile:


            reader = csv.reader(csvfile)


            next(reader)


            for row in reader:

                number_of_slice_in_lesion_fold +=1

                subject_name=row[0]

                if subject_name not in subjects_in_this_fold:
                    subjects_in_this_fold.append(subject_name)
                
                # try:
                #     number_of_slice_for_each_subject[subject_name] +=1
                # except:
                #     number_of_slice_for_each_subject[subject_name] =1 
        

        ## Read each subject 
                    

                    
        for subject in subjects_in_this_fold:


            mask_nii_nii=nib.load(f'{train_folder_mask}/{subject}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')
            mask_nii = mask_nii_nii.get_fdata()

            for co in range(  mask_nii.shape[2] ):
                single_mask_slice=mask_nii[:,co,:]
                single_mask_slice[single_mask_slice > 0] = 1

                if sum(sum(single_mask_slice)) ==0:

                    normal_slice.append({'subject_name':subject,'slice':co})

        

        ## Choose the number of random from the normal slice and add to that fold 
        random.shuffle(normal_slice)

        num_elements_to_choose = number_of_slice_in_lesion_fold

        # Choose elements randomly from different parts of the list
        chosen_indices = random.sample(range(len(normal_slice)), num_elements_to_choose)
        chosen_elements = [normal_slice[i] for i in chosen_indices]




        ### save the csv file 



    # CSV file path
        csv_file_path = f'Sagittal_K_Fold_CSV/Sagittal_Normal_fold_0{sagittal_fold}.csv'

        # Writing data to CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = chosen_elements[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Writing header
            writer.writeheader()

            # Writing data
            writer.writerows(chosen_elements)

        print(f'Data has been written to {csv_file_path}.')



    return 'Done'






                
spliting_noramal_slice(axial_folds_csv,sagittal_folds_csv,coronal_folds_csv,train_folder_mask)
