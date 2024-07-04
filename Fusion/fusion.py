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


        mri_list = os.listdir(self.data_folder+'/mri')

        for subject_name in mri_list:

            self.create_folder_sub_folder_of_result(subject_name)


            mri_nii=nib.load( self.data_folder+'/mri/' +'/'+ subject_name )

            self.header= mri_nii.header.copy()

            self.mri_numpy = mri_nii.get_fdata()

            mask_name=subject_name.split('_T1w')[0] + '_label-L_desc-T1lesion_mask.nii.gz'

            mask_nii=nib.load( f'{self.data_folder}/mask/{mask_name}' )

            self.mask_numpy = mask_nii.get_fdata()

            self.mask_numpy=np.where(self.mask_numpy > 0, 1, 0)


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

                axial_mask_numpy=np.zeros((208,240,1),dtype=np.float32)


                single_slice_axial=self.mri_numpy[:,:,ax]

                normalized_single_slice_axial=self.normalize_to_255(single_slice_axial)

                axial_mri_numpy[6:6+197,3:3+233,0]=normalized_single_slice_axial

                expand_mri=np.expand_dims(axial_mri_numpy,axis=0)

                prediction = self.axial_model.predict(expand_mri)

                binary_prediction = (prediction > 0.5).astype(np.float32)

                self.axial_prediction[:,:,ax]= prediction[0,6:6+197,3:3+233,0]

                self.binary_axial_prediction[:,:,ax]= binary_prediction[0,6:6+197,3:3+233,0]




                axial_mask_numpy[6:6+197,3:3+233,0]=self.mask_numpy[:,:,ax]

                true_label=np.expand_dims(axial_mask_numpy,axis=0)


                if self.evaluation:

                    dice_value,precision_value,recall_value,hddd_value= self.evaluation_2D(true_label,binary_prediction)

                    last_slice_number= self.axial_size -1

                    self.saving_2D_evaluation_result(subject_name,ax,last_slice_number,"axial",'before',dice_value,precision_value,recall_value,hddd_value)








            
            for sa in range(self.sagittal_size):

                print("Sagittal, Slice:",sa,"Totoal slices:",self.sagittal_size)

                sagittal_mri_numpy=np.zeros((240,208,1),dtype=np.float32)

                sagittal_mask_numpy=np.zeros((240,208,1),dtype=np.float32)

                single_slice_sagittal=self.mri_numpy[sa,:,:]

                normalized_single_slice_sagittal=self.normalize_to_255(single_slice_sagittal)

                sagittal_mri_numpy[3:3+233,9:9+189,0]=normalized_single_slice_sagittal

                expand_mri=np.expand_dims(sagittal_mri_numpy,axis=0)

                prediction = self.sagittal_model.predict(expand_mri)

                binary_prediction = (prediction > 0.5).astype(np.float32)

                self.sagittal_prediction[sa,:,:]= prediction[0,3:3+233,9:9+189,0]  

                self.binary_sagittal_prediction[sa,:,:]= binary_prediction[0,3:3+233,9:9+189,0]  




                sagittal_mask_numpy[3:3+233,9:9+189,0]=self.mask_numpy[sa,:,:]
                true_label=np.expand_dims(sagittal_mask_numpy,axis=0)


                if self.evaluation:

                    dice_value,precision_value,recall_value,hddd_value= self.evaluation_2D(true_label,binary_prediction)

                    last_slice_number= self.sagittal_size -1

                    self.saving_2D_evaluation_result(subject_name,sa,last_slice_number,"sagittal",'before',dice_value,precision_value,recall_value,hddd_value)








            for co in range(self.coronal_size):

                print("Coronal, Slice:",co,"Totoal slices:",self.coronal_size)

                coronal_mri_numpy=np.zeros((208,208,1),dtype=np.float32)
                coronal_mask_numpy=np.zeros((208,208,1),dtype=np.float32)

                single_slice_coronal=self.mri_numpy[:,co,:]

                normalized_single_slice_coronal=self.normalize_to_255(single_slice_coronal)

                coronal_mri_numpy[5:5+197,9:9+189,0]=normalized_single_slice_coronal

                expand_mri=np.expand_dims(coronal_mri_numpy,axis=0)

                prediction = self.coronal_model.predict(expand_mri)

                binary_prediction = (prediction > 0.5).astype(np.float32)

                self.coronal_prediction[:,co,:]= prediction[0,5:5+197,9:9+189,0] 
                self.binary_coronal_prediction[:,co,:]= binary_prediction[0,5:5+197,9:9+189,0]  






                coronal_mask_numpy[5:5+197,9:9+189,0]=self.mask_numpy[:,co,:]

                true_label=np.expand_dims(coronal_mask_numpy,axis=0)


                if self.evaluation:

                    dice_value,precision_value,recall_value,hddd_value= self.evaluation_2D(true_label,binary_prediction)

                    last_slice_number= self.sagittal_size -1

                    self.saving_2D_evaluation_result(subject_name,co,last_slice_number,"coronal",'before',dice_value,precision_value,recall_value,hddd_value)




            self.saving_planes_segmentation_results_before_fusion()

            self.apply_fusion(subject_name)








    def evaluation_2D(self,true_label,predict):

        dice_value=stats.dice_coef(true_label,predict)

        precision_value=stats.precision(true_label,predict)

        recall_value=stats.recall(true_label,predict)

        hddd_value=stats.hausdorff_distance(true_label,predict)



        return dice_value,precision_value,recall_value,hddd_value
    
      

    


    def saving_2D_evaluation_result(self,subject_name,slice,last_slice_number,plan,after_or_before_fusion,dice_value,precision_value,recall_value,hddd_value):

        if slice==0:

            self.results=[]

        result={'subject':subject_name,'slice':slice,'dice':dice_value.numpy(),'presision':precision_value.numpy(),'recall':recall_value.numpy(),'hausdorff_distance':hddd_value.numpy()}
        self.results.append(result)



        if slice==last_slice_number:

            result={'subject':subject_name,'slice':slice,'dice':dice_value.numpy(),'presision':precision_value.numpy(),'recall':recall_value.numpy(),'hausdorff_distance':hddd_value.numpy()}
            self.results.append(result)

            if after_or_before_fusion=='before':

                if plan=='axial':

                    csv_file = f'./{self.axial_directory}/axial_before_fusion.csv'

                    with open(csv_file, 'w', newline='') as csvfile:
                        fieldnames = self.results[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        # Writing header
                        writer.writeheader()

                        # Writing data
                        writer.writerows(self.results)

                    print(f'Data has been written to {csvfile}.')




                if plan=='sagittal':

                    csv_file = f'./{self.sagittal_directory}/sagittal_before_fusion.csv'

                    with open(csv_file, 'w', newline='') as csvfile:
                        fieldnames = self.results[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        # Writing header
                        writer.writeheader()

                        # Writing data
                        writer.writerows(self.results)

                    print(f'Data has been written to {csvfile}.')

                if plan=='coronal':

                    csv_file = f'./{self.coronal_directory}/coronal_before_fusion.csv'

                    with open(csv_file, 'w', newline='') as csvfile:
                        fieldnames = self.results[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        # Writing header
                        writer.writeheader()

                        # Writing data
                        writer.writerows(self.results)

                    print(f'Data has been written to {csvfile}.')




            if after_or_before_fusion=='after':

                if plan=='axial':

                    csv_file = f'./{self.axial_directory}/axial_after_fusion.csv'

                    with open(csv_file, 'w', newline='') as csvfile:
                        fieldnames = self.results[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        # Writing header
                        writer.writeheader()

                        # Writing data
                        writer.writerows(self.results)

                    print(f'Data has been written to {csvfile}.')




                if plan=='sagittal':

                    csv_file = f'./{self.sagittal_directory}/sagittal_after_fusion.csv'

                    with open(csv_file, 'w', newline='') as csvfile:
                        fieldnames = self.results[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        # Writing header
                        writer.writeheader()

                        # Writing data
                        writer.writerows(self.results)

                    print(f'Data has been written to {csvfile}.')

                if plan=='coronal':

                    csv_file = f'./{self.coronal_directory}/coronal_after_fusion.csv'

                    with open(csv_file, 'w', newline='') as csvfile:
                        fieldnames = self.results[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        # Writing header
                        writer.writeheader()

                        # Writing data
                        writer.writerows(self.results)

                    print(f'Data has been written to {csvfile}.')





    def saving_planes_segmentation_results_before_fusion(self):


        axial_prediction = nib.nifti1.Nifti1Image(self.binary_axial_prediction, None, header=self.header)

        nib.save(axial_prediction, f'{self.axial_directory}/axial_before_fusion.nii.gz')


        sagittal_prediction = nib.nifti1.Nifti1Image(self.binary_sagittal_prediction, None, header=self.header)

        nib.save(sagittal_prediction, f'{self.sagittal_directory}/sagittal_before_fusion.nii.gz')


        coronal_prediction = nib.nifti1.Nifti1Image(self.binary_coronal_prediction, None, header=self.header)

        nib.save(coronal_prediction, f'{self.coronal_directory}/coronal_before_fusion.nii.gz')



    def comparison_before_after_fusion(self,before, after, saving_directory):
    # Read CSV files
        df1 = pd.read_csv(before)
        df2 = pd.read_csv(after)


        columns=['dice','presision','recall']

        
        # Calculate mean and std for specified columns
        mean_std_df1 = df1[columns].agg(['mean', 'std'])
        mean_std_df2 = df2[columns].agg(['mean', 'std'])
        
        # Rename index to include file names
        mean_std_df1.index = ['before_mean', 'before_std']
        mean_std_df2.index = ['after_mean', 'after_std']
        
        # Calculate the difference between means of the two dataframes
        difference = mean_std_df1.loc['before_mean'] - mean_std_df2.loc['after_mean']
        
        # Convert difference to a DataFrame
        difference_df = pd.DataFrame(difference).transpose()
        difference_df.index = ['difference']
        
        # Create a new DataFrame to store results
        results_df = pd.concat([mean_std_df1, mean_std_df2, difference_df])
        
        # Write results to a new CSV file
        results_df.to_csv(f'{saving_directory}/comparison_before_after_fusion.csv')
        
        print("Results of comparison of before and after fusion have been written ")



    





    def evaluation_3D(self,true_label,predict):

        dice_value=stats.dice_coef(true_label,predict)

        precision_value=stats.precision(true_label,predict)

        recall_value=stats.recall(true_label,predict)



        volume_difference=stats.volume_difference(true_label,predict)




        return dice_value,precision_value,recall_value,volume_difference
    
      



    def saving_3D_evaluation_result(self,dice_value,precision_value,recall_value,
                                    volume_difference):


        self.results=[]

        result={'dice':dice_value.numpy(),'presision':precision_value.numpy(),
                'recall':recall_value.numpy(),
                'volume_difference':volume_difference}
        

       
        
        self.results.append(result)


        csv_file = f'./{self.fusion_directory}/fusion_evaluation.csv'

        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = self.results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Writing header
            writer.writeheader()

            # Writing data
            writer.writerows(self.results)

        print(f'Data has been written to {csvfile}.')







    def apply_fusion(self,subject_name):


        fusion_combination= (self.axial_prediction + self.sagittal_prediction + self.coronal_prediction) / 3

        fusion_result=np.where(fusion_combination >= 0.5, 1, 0)


        if self.evaluation:
        ## Evaluation axial after fusion


            for ax in range(self.axial_size):

                axial_fusion_numpy=np.zeros((208,240,1),dtype=np.float32)

                axial_mask_numpy=np.zeros((208,240,1),dtype=np.float32)

                axial_fusion_numpy[6:6+197,3:3+233,0]=fusion_result[:,:,ax]


                expand_axial_fusion=np.expand_dims(axial_fusion_numpy,axis=0)



                axial_mask_numpy[6:6+197,3:3+233,0]=self.mask_numpy[:,:,ax]

                true_label=np.expand_dims(axial_mask_numpy,axis=0)



                dice_value,precision_value,recall_value,hddd_value= self.evaluation_2D(true_label,expand_axial_fusion)

                last_slice_number= self.axial_size -1

                self.saving_2D_evaluation_result(subject_name,ax,last_slice_number,"axial",'after',dice_value,precision_value,recall_value,hddd_value)





            for sa in range(self.sagittal_size):

                sagittal_fusion_numpy=np.zeros((240,208,1),dtype=np.float32)
                sagittal_mask_numpy=np.zeros((240,208,1),dtype=np.float32)

                sagittal_fusion_numpy[3:3+233,9:9+189,0]=fusion_result[sa,:,:]


                expand_sagittal_fusion=np.expand_dims(sagittal_fusion_numpy,axis=0)



                sagittal_mask_numpy[3:3+233,9:9+189,0]=self.mask_numpy[sa,:,:]

                true_label=np.expand_dims(sagittal_mask_numpy,axis=0)



                dice_value,precision_value,recall_value,hddd_value= self.evaluation_2D(true_label,expand_sagittal_fusion)

                last_slice_number= self.sagittal_size -1

                self.saving_2D_evaluation_result(subject_name,sa,last_slice_number,"sagittal",'after',dice_value,precision_value,recall_value,hddd_value)







            for co in range(self.coronal_size):

                coronal_fusion_numpy=np.zeros((208,208,1),dtype=np.float32)

                coronal_mask_numpy=np.zeros((208,208,1),dtype=np.float32)



                coronal_fusion_numpy[5:5+197,9:9+189,0]=fusion_result[:,co,:]


                expand_coronal_fusion=np.expand_dims(coronal_fusion_numpy,axis=0)



                coronal_mask_numpy[5:5+197,9:9+189,0]=self.mask_numpy[:,co,:]

                true_label=np.expand_dims(coronal_mask_numpy,axis=0)



                dice_value,precision_value,recall_value,hddd_value= self.evaluation_2D(true_label,expand_coronal_fusion)

                last_slice_number= self.coronal_size -1

                self.saving_2D_evaluation_result(subject_name,co,last_slice_number,"coronal",'after',dice_value,precision_value,recall_value,hddd_value)

            

            ## Comparision before and after 

            self.comparison_before_after_fusion(f'{self.axial_directory}/axial_before_fusion.csv', f'{self.axial_directory}/axial_after_fusion.csv', self.axial_directory)
            self.comparison_before_after_fusion(f'{self.sagittal_directory}/sagittal_before_fusion.csv', f'{self.sagittal_directory}/sagittal_after_fusion.csv', self.sagittal_directory)
            self.comparison_before_after_fusion(f'{self.coronal_directory}/coronal_before_fusion.csv', f'{self.coronal_directory}/coronal_after_fusion.csv', self.coronal_directory)




            ## 3D evaluation

            dice_value,precision_value,recall_value,volume_difference=self.evaluation_3D(self.mask_numpy,fusion_result)

            self.saving_3D_evaluation_result(dice_value,precision_value,recall_value,volume_difference)





            

        ## Saving Fusion Results and MRI and Mask


        fusion_nii = nib.nifti1.Nifti1Image(fusion_result, None, header=self.header)

        nib.save(fusion_nii, f'{self.fusion_directory}/segmentation_mask.nii.gz')



        mri_nii = nib.nifti1.Nifti1Image(self.mri_numpy, None, header=self.header)

        nib.save(mri_nii, f'{self.fusion_directory}/mri_T1.nii.gz')



        mask_nii = nib.nifti1.Nifti1Image(self.mask_numpy, None, header=self.header)

        nib.save(mask_nii, f'{self.fusion_directory}/mask.nii.gz')














parser.add_argument('-data', '--data_folder', type=str, help='The data folder : Sholud be two sub folder mri and mask')
parser.add_argument('-models', '--trained_model', type=str, help='Directory of trained model : axial.h5 , sagittal.h5, coronal.h5')

parser.add_argument('-save', '--saving_directory', type=str, help='The directory want to save results')

args = parser.parse_args






args = parser.parse_args()




d=Fusion(f'./{args.data_folder}',f'./{args.trained_model}/axial.h5',f'./{args.trained_model}/sagittal.h5',f'./{args.trained_model}/coronal.h5',True,args.saving_directory)










                






