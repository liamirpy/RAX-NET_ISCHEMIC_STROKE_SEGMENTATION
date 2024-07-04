## This is liamirpy

## In this part we want to predict for evalutation data and predict the result and save the result
## We do it for each plane for all folds




## Axial 

import numpy as np

import pandas as pd

import os 
import time
import nibabel as nib
from keras.models import load_model
from statistic import Statistics
import csv 
import keras.backend as K


stats = Statistics()





axial_fold_csv_directory='../../Distribution_Batch/Axial_Batches_CSV'
sagittal_fold_csv_directory='../../Distribution_Batch/Sagittal_Batches_CSV'
coronal_fold_csv_directory='../../Distribution_Batch/Coronal_Batches_CSV'



axial_data_directory='../../Moving_Converting_Data/axial_2D/'
sagittal_data_directory='../../Moving_Converting_Data/sagittal_2D/'
coronal_data_directory='../../Moving_Converting_Data/coronal_2D/'




axial_saved_model_folds='../../Model_Comparison/D_Net//axial'
sagittal_saved_model_folds='../../Model_Comparison/D_Net//sagittal'
coronal_saved_model_folds='../../Model_Comparison/D_Net//coronal'




number_of_folds=5

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
    
    return normalized_array.astype(np.float32)






def data_shape(data):
    """
    Transforms the input data shape from (N, 208, 240, 1) to (N, 192, 192, 4) 
    by first reshaping to (N, 192, 192, 1) and then repeating the last dimension 4 times.

    Parameters:
    - data (numpy.ndarray): The input data with shape (N, 208, 240, 1).

    Returns:
    - numpy.ndarray: The transformed data with shape (N, 192, 192, 4).
    """
    # Ensure the input data has the correct initial shape
    if data.shape[-1] != 1:
        raise ValueError("The last dimension of the input data must be 1.")
    
    # Reshape each image to (192, 192, 1)
    reshaped_data = np.zeros((data.shape[0], 192, 192, 1),dtype=np.float32)
    for i in range(data.shape[0]):
        reshaped_data[i] = np.resize(data[i], (192, 192, 1))
    
    return reshaped_data
def transform_data_shape( data):
    """
    Transforms the input data shape from (N, 208, 240, 1) to (N, 192, 192, 4) 
    by first reshaping to (N, 192, 192, 1) and then repeating the last dimension 4 times.

    Parameters:
    - data (numpy.ndarray): The input data with shape (N, 208, 240, 1).

    Returns:
    - numpy.ndarray: The transformed data with shape (N, 192, 192, 4).
    """
    # Ensure the input data has the correct initial shape
    if data.shape[-1] != 1:
        raise ValueError("The last dimension of the input data must be 1.")
    
    # Reshape each image to (192, 192, 1)
    reshaped_data = np.zeros((data.shape[0], 192, 192, 1),dtype=np.float32)
    for i in range(data.shape[0]):
        reshaped_data[i] = np.resize(data[i], (192, 192, 1))
    
    # Determine the new shape with the repeated dimension
    new_shape = (data.shape[0], 192, 192, 4)
    
    # Create a new array to hold the transformed data
    transformed_data = np.zeros(new_shape)
    
    # Apply transformation for each batch
    for t in range(4):
        transformed_data[:, :, :, t] = reshaped_data[:, :, :, 0]
    
    return transformed_data

for plane in [axial_fold_csv_directory]:

# for plane in [coronal_fold_csv_directory]:



    for fold in range(1,number_of_folds+1):

        results=[]

        validation_data_csv= pd.read_csv(f'{plane}/fold_{fold}/Validation_lesion_normal_batches.csv')


        if plane == axial_fold_csv_directory:


            model=load_model(f'{axial_saved_model_folds}/fold_{fold}/axial_fold_{fold}.h5',custom_objects={
                'dice_coef': stats.dice_coef,
                'precision':stats.precision,
                'recall':stats.recall,
                'hausdorff_distance':stats.hausdorff_distance,'K':K,'EML':stats.EML})
            

            single_slice_mri=train_data_mri=np.zeros((1,197,233,1),dtype=np.float32)

            single_slice_true_label=train_data_mri=np.zeros((1,197,233,1),dtype=np.float32)

            for index, row in validation_data_csv.iterrows():


                subject = row['subject_name']

                slice_number = row['slice']


                mri_nii=nib.load(f'{axial_data_directory}/{subject}/{subject}_slice_{slice_number}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()

                mri_numpy =normalize_to_255(mri_numpy)

                single_slice_mri[0,:,:,0]=mri_numpy
                expand_mri=transform_data_shape(single_slice_mri)

                # expand_mri=np.expand_dims(single_slice_mri,axis=0)



                t0=time.time()
                prediction = model.predict(expand_mri)
                t1=time.time()


                binary_prediction = (prediction > 0.5).astype(np.float32)





                mask_nii=nib.load(f'{axial_data_directory}/{subject}/{subject}_slice_{slice_number}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()
                mask_numpy=np.where(mask_numpy > 0, 1, 0)

                single_slice_true_label[0,:,:,0]=mask_numpy
                expand_true_label=data_shape(single_slice_true_label)

                print('bbb',binary_prediction.shape)
                print('gg',expand_true_label.shape)




                dice_value=stats.dice_coef(expand_true_label,binary_prediction)
                precision_value=stats.precision(expand_true_label,binary_prediction)
                recall_value=stats.recall(expand_true_label,binary_prediction)
                hddd_value=stats.hausdorff_distance(expand_true_label,binary_prediction)


                result={'subject':subject,'slice':slice_number,'dice':dice_value.numpy(),'presision':precision_value.numpy(),'recall':recall_value.numpy(),'hausdorff_distance':hddd_value.numpy(),'time':t1 - t0}
                results.append(result)


            
            lesion_coronal_distribution = f'./Axial/result_of_axial_fold_{fold}.csv'

            with open(lesion_coronal_distribution, 'w', newline='') as csvfile:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Writing header
                writer.writeheader()

                # Writing data
                writer.writerows(results)

            print(f'Data has been written to {lesion_coronal_distribution}.')

            








            




        if plane == sagittal_fold_csv_directory:


            model=load_model(f'{sagittal_saved_model_folds}/fold_{fold}/sagittal_fold_{fold}.h5'
                             ,
                             custom_objects={
                'dice_coef': stats.dice_coef,
                'precision':stats.precision,
                'recall':stats.recall,
                'hausdorff_distance':stats.hausdorff_distance,'K':K,'EML':stats.EML})
            

            single_slice_mri=train_data_mri=np.zeros((1,233,189,1),dtype=np.float32)
            single_slice_true_label=train_data_mri=np.zeros((1,233,189,1),dtype=np.float32)

            for index, row in validation_data_csv.iterrows():


                subject = row['subject_name']

                slice_number = row['slice']


                mri_nii=nib.load(f'{sagittal_data_directory}/{subject}/{subject}_slice_{slice_number}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()



                single_slice_mri[0,:,:,0]=mri_numpy
                expand_mri=transform_data_shape(single_slice_mri)



                t0=time.time()
                prediction = model.predict(expand_mri)
                t1=time.time()

                binary_prediction = (prediction > 0.5).astype(np.float32)



                mask_nii=nib.load(f'{sagittal_data_directory}/{subject}/{subject}_slice_{slice_number}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()
                mask_numpy=np.where(mask_numpy > 0, 1, 0)



                single_slice_true_label[0,:,:,0]=mask_numpy
                expand_true_label=data_shape(single_slice_true_label)






                dice_value=stats.dice_coef(expand_true_label,binary_prediction)
                precision_value=stats.precision(expand_true_label,binary_prediction)
                recall_value=stats.recall(expand_true_label,binary_prediction)
                hddd_value=stats.hausdorff_distance(expand_true_label,binary_prediction)


                result={'subject':subject,'slice':slice_number,'dice':dice_value.numpy(),'presision':precision_value.numpy(),'recall':recall_value.numpy(),'hausdorff_distance':hddd_value.numpy(),'time':t1 - t0}
                results.append(result)


            lesion_coronal_distribution = f'./Sagittal/result_of_sagittal_fold_{fold}.csv'

            with open(lesion_coronal_distribution, 'w', newline='') as csvfile:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Writing header
                writer.writeheader()

                # Writing data
                writer.writerows(results)

            print(f'Data has been written to {lesion_coronal_distribution}.')

            

            
            



        if plane == coronal_fold_csv_directory:


            model=load_model(f'{coronal_saved_model_folds}/fold_{fold}/coronal_fold_{fold}.h5',custom_objects={
                'dice_coef': stats.dice_coef,
                'precision':stats.precision,
                'recall':stats.recall,
                'hausdorff_distance':stats.hausdorff_distance,'K':K,'EML':stats.EML})
            


            single_slice_mri=train_data_mri=np.zeros((1,197,189,1),dtype=np.float32)
            single_slice_true_label=train_data_mri=np.zeros((1,197,189,1),dtype=np.float32)

            for index, row in validation_data_csv.iterrows():


                subject = row['subject_name']

                slice_number = row['slice']


                mri_nii=nib.load(f'{coronal_data_directory}/{subject}/{subject}_slice_{slice_number}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()


                single_slice_mri[0,:,:,0]=mri_numpy
                expand_mri=transform_data_shape(single_slice_mri)


                t0=time.time()
                prediction = model.predict(expand_mri)
                t1=time.time()
                binary_prediction = (prediction > 0.5).astype(np.float32)


                mask_nii=nib.load(f'{coronal_data_directory}/{subject}/{subject}_slice_{slice_number}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()
                mask_numpy=np.where(mask_numpy > 0, 1, 0)

                single_slice_true_label[0,:,:,0]=mask_numpy
                expand_true_label=data_shape(single_slice_true_label)







                dice_value=stats.dice_coef(expand_true_label,binary_prediction)
                precision_value=stats.precision(expand_true_label,binary_prediction)
                recall_value=stats.recall(expand_true_label,binary_prediction)
                hddd_value=stats.hausdorff_distance(expand_true_label,binary_prediction)


                result={'subject':subject,'slice':slice_number,'dice':dice_value.numpy(),'presision':precision_value.numpy(),'recall':recall_value.numpy(),'hausdorff_distance':hddd_value.numpy(),'time':t1 - t0}
                results.append(result)



            lesion_coronal_distribution = f'./Coronal/result_of_coronal_fold_{fold}.csv'

            with open(lesion_coronal_distribution, 'w', newline='') as csvfile:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Writing header
                writer.writeheader()

                # Writing data
                writer.writerows(results)

            print(f'Data has been written to {lesion_coronal_distribution}.')

            





