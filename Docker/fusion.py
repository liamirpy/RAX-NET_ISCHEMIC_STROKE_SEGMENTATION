# This is Liamirpy from Lonleness Land 

'''
In this part we want to develop the fusion approach .
The fusion meaning that to combine various part to imporve the performance.
In this specifict object the fusion is combination of Segmentation model of
axial, sagittal and coronal to imporve the perfomance of overall segmentation 
of single subject.

'''
import os 
import nibabel as nib
import numpy as np
from keras.models import load_model
from statistic import Statistics
import csv
import pandas as pd
import argparse

parser = argparse.ArgumentParser()


## Read the nii file 


stats = Statistics()






    ## 



class Fusion:


    def __init__(self,
                 
                 data_folder,

                 axial_model_directory,

                 sagittal_model_directory,

                 coronal_model_directory,

                 evaluation=False,

                 results_folder='./Results',
                        
                 ):
        

    
        self.data_folder=data_folder

        self.evaluation=evaluation

        self.results_folder=results_folder

        self.axial_model=load_model(f'{axial_model_directory}',custom_objects={'FocalTverskyLoss':   
                                                                                                                            
                        stats.FocalTverskyLoss,'dice_coef': stats.dice_coef,'precision':stats.precision,'recall':stats.recall,'hausdorff_distance':stats.hausdorff_distance})
            

        self.sagittal_model=load_model(f'{sagittal_model_directory}',custom_objects={'FocalTverskyLoss':   
                                                                                                                            
                        stats.FocalTverskyLoss,'dice_coef': stats.dice_coef,'precision':stats.precision,'recall':stats.recall,'hausdorff_distance':stats.hausdorff_distance})
            

        self.coronal_model=load_model(f'{coronal_model_directory}',custom_objects={'FocalTverskyLoss':   
                                                                                                                            
                        stats.FocalTverskyLoss,'dice_coef': stats.dice_coef,'precision':stats.precision,'recall':stats.recall,'hausdorff_distance':stats.hausdorff_distance})
            
        
        self.read_data_and_predict()







    


    def create_folder(self,folder_path):
        # Check if the folder exists
        if os.path.exists(folder_path):
            return f"The folder '{folder_path}' already exists. It's fine."
        else:
            # Create the folder
            os.makedirs(folder_path)
            return f"The folder '{folder_path}' has been created successfully."
        
    

    def create_folder_sub_folder_of_result(self,subject_name):

        # Create a Results folder 

        self.create_folder(f'./{self.results_folder}')

        # Create a subject folder 

        subject_directory=f'./{self.results_folder}/'+subject_name.split('_T1w')[0]

        self.create_folder(subject_directory)


        # Create axial folder and sub folder 


        self.axial_directory=f'./{self.results_folder}/'+subject_name.split('_T1w')[0]+'/axial'

        self.create_folder(self.axial_directory)

   


        # Create Sagittal folder and sub folder 


        self.sagittal_directory=f'./{self.results_folder}/'+subject_name.split('_T1w')[0]+'/sagittal'

        self.create_folder(self.sagittal_directory)

       




        # Create Coronal folder and sub folder 


        self.coronal_directory=f'./{self.results_folder}/'+subject_name.split('_T1w')[0]+'/coronal'

        self.create_folder(self.coronal_directory)





        # Create Fusion folder 


        self.fusion_directory=f'./{self.results_folder}/'+subject_name.split('_T1w')[0]+'/fusion'

        self.create_folder(self.fusion_directory)









    def normalize_to_255(self,array):
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
    




    def read_data_and_predict(self):


        mri_list = os.listdir(self.data_folder)

        for subject_name in mri_list:

            self.create_folder_sub_folder_of_result(subject_name)


            mri_nii=nib.load( self.data_folder +'/'+ subject_name )

            self.header= mri_nii.header.copy()

            self.mri_numpy = mri_nii.get_fdata()

           
            ## Define the size of input data

            self.sagittal_size=mri_nii.shape[0]

            self.coronal_size=mri_nii.shape[1]

            self.axial_size=mri_nii.shape[2]

            ## Put each data to seperate numpy with normalization and padding

            self.sagittal_prediction=np.zeros((self.mri_numpy.shape[0],self.mri_numpy.shape[1],self.mri_numpy.shape[2]))

            self.coronal_prediction=np.zeros((self.mri_numpy.shape[0],self.mri_numpy.shape[1],self.mri_numpy.shape[2]))
                
            self.axial_prediction=np.zeros((self.mri_numpy.shape[0],self.mri_numpy.shape[1],self.mri_numpy.shape[2]))




            self.binary_sagittal_prediction=np.zeros((self.mri_numpy.shape[0],self.mri_numpy.shape[1],self.mri_numpy.shape[2]))

            self.binary_coronal_prediction=np.zeros((self.mri_numpy.shape[0],self.mri_numpy.shape[1],self.mri_numpy.shape[2]))
                
            self.binary_axial_prediction=np.zeros((self.mri_numpy.shape[0],self.mri_numpy.shape[1],self.mri_numpy.shape[2]))





            for ax in range(self.axial_size):

                print("Axial, Slice:",ax,"Totoal slices:",self.axial_size)

                axial_mri_numpy=np.zeros((208,240,1),dtype=np.float32)



                single_slice_axial=self.mri_numpy[:,:,ax]

                normalized_single_slice_axial=self.normalize_to_255(single_slice_axial)

                axial_mri_numpy[6:6+197,3:3+233,0]=normalized_single_slice_axial

                expand_mri=np.expand_dims(axial_mri_numpy,axis=0)

                prediction = self.axial_model.predict(expand_mri)

                binary_prediction = (prediction > 0.5).astype(np.float32)

                self.axial_prediction[:,:,ax]= prediction[0,6:6+197,3:3+233,0]

                self.binary_axial_prediction[:,:,ax]= binary_prediction[0,6:6+197,3:3+233,0]










            
            for sa in range(self.sagittal_size):

                print("Sagittal, Slice:",sa,"Totoal slices:",self.sagittal_size)

                sagittal_mri_numpy=np.zeros((240,208,1),dtype=np.float32)


                single_slice_sagittal=self.mri_numpy[sa,:,:]

                normalized_single_slice_sagittal=self.normalize_to_255(single_slice_sagittal)

                sagittal_mri_numpy[3:3+233,9:9+189,0]=normalized_single_slice_sagittal

                expand_mri=np.expand_dims(sagittal_mri_numpy,axis=0)

                prediction = self.sagittal_model.predict(expand_mri)

                binary_prediction = (prediction > 0.5).astype(np.float32)

                self.sagittal_prediction[sa,:,:]= prediction[0,3:3+233,9:9+189,0]  

                self.binary_sagittal_prediction[sa,:,:]= binary_prediction[0,3:3+233,9:9+189,0]  





            for co in range(self.coronal_size):

                print("Coronal, Slice:",co,"Totoal slices:",self.coronal_size)

                coronal_mri_numpy=np.zeros((208,208,1),dtype=np.float32)

                single_slice_coronal=self.mri_numpy[:,co,:]

                normalized_single_slice_coronal=self.normalize_to_255(single_slice_coronal)

                coronal_mri_numpy[5:5+197,9:9+189,0]=normalized_single_slice_coronal

                expand_mri=np.expand_dims(coronal_mri_numpy,axis=0)

                prediction = self.coronal_model.predict(expand_mri)

                binary_prediction = (prediction > 0.5).astype(np.float32)

                self.coronal_prediction[:,co,:]= prediction[0,5:5+197,9:9+189,0] 
                self.binary_coronal_prediction[:,co,:]= binary_prediction[0,5:5+197,9:9+189,0]  







            self.saving_planes_segmentation_results_before_fusion()

            self.apply_fusion()








    def saving_planes_segmentation_results_before_fusion(self):


        axial_prediction = nib.nifti1.Nifti1Image(self.binary_axial_prediction, None, header=self.header)

        nib.save(axial_prediction, f'{self.axial_directory}/axial_before_fusion.nii.gz')


        sagittal_prediction = nib.nifti1.Nifti1Image(self.binary_sagittal_prediction, None, header=self.header)

        nib.save(sagittal_prediction, f'{self.sagittal_directory}/sagittal_before_fusion.nii.gz')


        coronal_prediction = nib.nifti1.Nifti1Image(self.binary_coronal_prediction, None, header=self.header)

        nib.save(coronal_prediction, f'{self.coronal_directory}/coronal_before_fusion.nii.gz')



    



    def apply_fusion(self):


        fusion_combination= (self.axial_prediction + self.sagittal_prediction + self.coronal_prediction) / 3

        fusion_result=np.where(fusion_combination >= 0.5, 1, 0)



        ## Saving Fusion Results and MRI and Mask


        fusion_nii = nib.nifti1.Nifti1Image(fusion_result, None, header=self.header)

        nib.save(fusion_nii, f'{self.fusion_directory}/segmentation_mask.nii.gz')



        mri_nii = nib.nifti1.Nifti1Image(self.mri_numpy, None, header=self.header)

        nib.save(mri_nii, f'{self.fusion_directory}/mri_T1.nii.gz')












Fusion('./data','./trained_models/axial.h5','./trained_models/sagittal.h5','./trained_models/coronal.h5',False,'./results')

print('The Segmentation Finished')








                






