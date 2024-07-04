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


stats = Statistics()


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





axial_fold_csv_directory='../Distribution_Batch/Axial_Batches_CSV'
sagittal_fold_csv_directory='../Distribution_Batch/Sagittal_Batches_CSV'
coronal_fold_csv_directory='../Distribution_Batch/Coronal_Batches_CSV'



axial_data_directory='../Moving_Converting_Data/axial_2D/'
sagittal_data_directory='../Moving_Converting_Data/sagittal_2D/'
coronal_data_directory='../Moving_Converting_Data/coronal_2D/'




axial_saved_model_folds='../Training/axial'
sagittal_saved_model_folds='../Training/sagittal'
coronal_saved_model_folds='../Training/coronal'




number_of_folds=5



# for plane in [axial_fold_csv_directory,sagittal_fold_csv_directory,coronal_fold_csv_directory]:

for plane in [sagittal_fold_csv_directory,coronal_fold_csv_directory]:


    for fold in range(1,number_of_folds+1):

        results=[]

        validation_data_csv= pd.read_csv(f'{plane}/fold_{fold}/Validation_lesion_normal_batches.csv')


        if plane == axial_fold_csv_directory:


            model=load_model(f'{axial_saved_model_folds}/fold_{fold}/axial_fold_{fold}.h5',custom_objects={'FocalTverskyLoss':   
                                                                                                                            
                        stats.FocalTverskyLoss,'dice_coef': stats.dice_coef,'precision':stats.precision,'recall':stats.recall,'hausdorff_distance':stats.hausdorff_distance})
            

            single_slice_mri=train_data_mri=np.zeros((208,240,1),dtype=np.float32)

            single_slice_true_label=train_data_mri=np.zeros((208,240,1),dtype=np.float32)

            for index, row in validation_data_csv.iterrows():


                subject = row['subject_name']

                slice_number = row['slice']


                mri_nii=nib.load(f'{axial_data_directory}/{subject}/{subject}_slice_{slice_number}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()

                mri_numpy =normalize_to_255(mri_numpy)

                single_slice_mri[6:6+197,3:3+233,0]=mri_numpy

                expand_mri=np.expand_dims(single_slice_mri,axis=0)



                t0=time.time()
                prediction = model.predict(expand_mri)
                t1=time.time()


                binary_prediction = (prediction > 0.5).astype(np.float32)





                mask_nii=nib.load(f'{axial_data_directory}/{subject}/{subject}_slice_{slice_number}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()
                mask_numpy=np.where(mask_numpy > 0, 1, 0)

                single_slice_true_label[6:6+197,3:3+233,0]=mask_numpy
                expand_true_label=np.expand_dims(single_slice_true_label,axis=0)



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


            model=load_model(f'{sagittal_saved_model_folds}/fold_{fold}/sagittal_fold_{fold}.h5',custom_objects={'FocalTverskyLoss':   
                                                                                                                            
                        stats.FocalTverskyLoss,'dice_coef': stats.dice_coef,'precision':stats.precision,'recall':stats.recall,'hausdorff_distance':stats.hausdorff_distance})
            

            single_slice_mri=train_data_mri=np.zeros((240,208,1),dtype=np.float32)
            single_slice_true_label=train_data_mri=np.zeros((240,208,1),dtype=np.float32)

            for index, row in validation_data_csv.iterrows():


                subject = row['subject_name']

                slice_number = row['slice']


                mri_nii=nib.load(f'{sagittal_data_directory}/{subject}/{subject}_slice_{slice_number}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()

                mri_numpy =normalize_to_255(mri_numpy)


                single_slice_mri[3:3+233,9:9+189,0]=mri_numpy

                expand_mri=np.expand_dims(single_slice_mri,axis=0)

                t0=time.time()
                prediction = model.predict(expand_mri)
                t1=time.time()

                binary_prediction = (prediction > 0.5).astype(np.float32)



                mask_nii=nib.load(f'{sagittal_data_directory}/{subject}/{subject}_slice_{slice_number}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()
                mask_numpy=np.where(mask_numpy > 0, 1, 0)



                single_slice_true_label[3:3+233,9:9+189,0]=mask_numpy
                expand_true_label=np.expand_dims(single_slice_true_label,axis=0)





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


            model=load_model(f'{coronal_saved_model_folds}/fold_{fold}/coronal_fold_{fold}.h5',custom_objects={'FocalTverskyLoss':   
                                                                                                                            
                        stats.FocalTverskyLoss,'dice_coef': stats.dice_coef,'precision':stats.precision,'recall':stats.recall,'hausdorff_distance':stats.hausdorff_distance})
            


            single_slice_mri=train_data_mri=np.zeros((208,208,1),dtype=np.float32)
            single_slice_true_label=train_data_mri=np.zeros((208,208,1),dtype=np.float32)

            for index, row in validation_data_csv.iterrows():


                subject = row['subject_name']

                slice_number = row['slice']


                mri_nii=nib.load(f'{coronal_data_directory}/{subject}/{subject}_slice_{slice_number}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()

                mri_numpy =normalize_to_255(mri_numpy)

                single_slice_mri[5:5+197,9:9+189,0]=mri_numpy

                expand_mri=np.expand_dims(single_slice_mri,axis=0)

                t0=time.time()
                prediction = model.predict(expand_mri)
                t1=time.time()
                binary_prediction = (prediction > 0.5).astype(np.float32)


                mask_nii=nib.load(f'{coronal_data_directory}/{subject}/{subject}_slice_{slice_number}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()
                mask_numpy=np.where(mask_numpy > 0, 1, 0)

                single_slice_true_label[5:5+197,9:9+189,0]=mask_numpy
                expand_true_label=np.expand_dims(single_slice_true_label,axis=0)



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

            





