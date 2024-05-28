### THIS IS LIAMIRPY FROM LONLENESS LAND.

### LOAD DATA FOR THE TRANING THE MODEL FOR SEGMENTATION


'''
            Approach:
    
    1- Read the csv file

    2- Find the subject in each batches
    
    3- find each subject from the dataset folder 




'''


import numpy as np

import nibabel as nib
import os 


import csv

import pandas as pd





def normalize_to_255(array):
    """
    Normalize a NumPy array between 0 and 255.
    
    Parameters:
        array (numpy.ndarray): Input array.
    
    Returns:
        numpy.ndarray: Normalized array.
    """
    # Normalize the array to be between 0 and 1
    normalized_array = array.astype(np.float32) / np.max(array)
    
    # Scale the values to be between 0 and 255
    normalized_array *= 255
    
    return normalized_array.astype(np.uint8)








data_directory='../Moving_Converting_Data/axial_2D'


def load_data(csv_directory,data_directory,fold=1):




    train_csv= f'{csv_directory}/fold_{fold}/Train_lesion_normal_batches.csv'
    
    validation_csv= f'{csv_directory}/fold_{fold}/Validation_lesion_normal_batches.csv'


    ## READ TRAINING DATA 


    df = pd.read_csv(train_csv)

    number_of_batch=df['batch'].max()



    train_data_mri=np.zeros((number_of_batch,32,208,240,1),dtype=np.float32)

    train_data_mask=np.zeros((number_of_batch,32,208,240,1),dtype=np.float32)




    for batch in range(1,number_of_batch+1):


        in_batch=0


        # Filter rows based on the batch number

        batch_data = df[df['batch'] == batch]

        # Display the filtered data

        # print(batch_data)
        print(batch_data)

        for index, row in batch_data.iterrows():

            

            subject_name= row['subject_name']

            slice= row['slice']


            mri_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_T1.nii.gz')

            mri_numpy=mri_nii.get_fdata()

            mri_numpy =normalize_to_255(mri_numpy)

            train_data_mri[batch-1,in_batch,6:6+197,3:3+233,0] = mri_numpy

            



            mask_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_mask.nii.gz')

            mask_numpy=mask_nii.get_fdata()

            mask_numpy=np.where(mask_numpy > 0, 1, 0)


            train_data_mask[batch-1,in_batch,6:6+197,3:3+233,0] = mask_numpy

            in_batch +=1











    df = pd.read_csv(validation_csv)

    number_of_batch=df['batch'].max()



    validation_data_mri=np.zeros((number_of_batch,32,208,240,1),dtype=np.float32)

    validation_data_mask=np.zeros((number_of_batch,32,208,240,1),dtype=np.float32)




    for batch in range(1,number_of_batch+1):


        in_batch=0


        # Filter rows based on the batch number

        batch_data = df[df['batch'] == batch]

        # Display the filtered data

        # print(batch_data)

        for index, row in batch_data.iterrows():

            

            subject_name= row['subject_name']

            slice= row['slice']


            mri_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_T1.nii.gz')

            mri_numpy=mri_nii.get_fdata()

            mri_numpy =normalize_to_255(mri_numpy)


            train_data_mri[batch-1,in_batch,6:6+197,3:3+233,0] = mri_numpy

            



            mask_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_mask.nii.gz')

            mask_numpy=mask_nii.get_fdata()

            mask_numpy=np.where(mask_numpy > 0, 1, 0)


            train_data_mask[batch-1,in_batch,6:6+197,3:3+233,0] = mask_numpy

            in_batch +=1


    




    return train_data_mri, train_data_mask, validation_data_mri, validation_data_mask




            









train_data_mri, train_data_mask, validation_data_mri, validation_data_mask=load_data('../Distribution_Batch/Axial_Batches_CSV',data_directory)




np.save('train_data_mri_fold_1.npy',train_data_mri)
np.save('train_data_mask_fold_1.npy',train_data_mask)
np.save('validation_data_mri_fold_1.npy',validation_data_mri)
np.save('validation_data_mask_fold_1.npy',validation_data_mask)





















