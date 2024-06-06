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
import argparse
parser = argparse.ArgumentParser()





class load_data:




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








    def axial_read_and_load_data_in_numpy(self,train_csv,validation_csv,data_directory,fold):

        self.fold=fold



        ## READ TRAINING DATA 


        train = pd.read_csv(train_csv)

        number_of_batch=train['batch'].max()



        self.train_data_mri=np.zeros((number_of_batch,32,208,240,1),dtype=np.float32)

        self.train_data_mask=np.zeros((number_of_batch,32,208,240,1),dtype=np.float32)




        for batch in range(1,number_of_batch+1):


            in_batch=0


            # Filter rows based on the batch number

            batch_data = train[train['batch'] == batch]

            # Display the filtered data

            # print(batch_data)
            print(batch_data)

            for index, row in batch_data.iterrows():

                

                subject_name= row['subject_name']

                slice= row['slice']


                mri_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()

                mri_numpy =self.normalize_to_255(mri_numpy)

                self.train_data_mri[batch-1,in_batch,6:6+197,3:3+233,0] = mri_numpy

                



                mask_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()

                mask_numpy=np.where(mask_numpy > 0, 1, 0)


                self.train_data_mask[batch-1,in_batch,6:6+197,3:3+233,0] = mask_numpy

                in_batch +=1











        validation = pd.read_csv(validation_csv)

        number_of_batch=validation['batch'].max()



        self.validation_data_mri=np.zeros((number_of_batch,32,208,240,1),dtype=np.float32)

        self.validation_data_mask=np.zeros((number_of_batch,32,208,240,1),dtype=np.float32)




        for batch in range(1,number_of_batch+1):


            in_batch=0


            # Filter rows based on the batch number

            batch_data = validation[validation['batch'] == batch]

            # Display the filtered data

            # print(batch_data)

            for index, row in batch_data.iterrows():

                

                subject_name= row['subject_name']

                slice= row['slice']


                mri_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()

                mri_numpy =self.normalize_to_255(mri_numpy)


                self.validation_data_mri[batch-1,in_batch,6:6+197,3:3+233,0] = mri_numpy

                



                mask_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()

                mask_numpy=np.where(mask_numpy > 0, 1, 0)


                self.validation_data_mask[batch-1,in_batch,6:6+197,3:3+233,0] = mask_numpy

                in_batch +=1







    


    def sagittal_read_and_load_data_in_numpy(self,train_csv,validation_csv,data_directory,fold):

        self.fold=fold



        ## READ TRAINING DATA 


        train = pd.read_csv(train_csv)

        number_of_batch=train['batch'].max()



        self.train_data_mri=np.zeros((number_of_batch,32,240,208,1),dtype=np.float32)

        self.train_data_mask=np.zeros((number_of_batch,32,240,208,1),dtype=np.float32)






        for batch in range(1,number_of_batch+1):


            in_batch=0


            # Filter rows based on the batch number

            batch_data = train[train['batch'] == batch]

            # Display the filtered data

            # print(batch_data)
            print(batch_data)

            for index, row in batch_data.iterrows():

                

                subject_name= row['subject_name']

                slice= row['slice']


                mri_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()

                mri_numpy =self.normalize_to_255(mri_numpy)

                self.train_data_mri[batch-1,in_batch,3:3+233,9:9+189,0] = mri_numpy

                



                mask_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()

                mask_numpy=np.where(mask_numpy > 0, 1, 0)


                self.train_data_mask[batch-1,in_batch,3:3+233,9:9+189,0] = mask_numpy

                in_batch +=1











        validation = pd.read_csv(validation_csv)

        number_of_batch=validation['batch'].max()



        self.validation_data_mri=np.zeros((number_of_batch,32,240,208,1),dtype=np.float32)

        self.validation_data_mask=np.zeros((number_of_batch,32,240,208,1),dtype=np.float32)




        for batch in range(1,number_of_batch+1):


            in_batch=0


            # Filter rows based on the batch number

            batch_data = validation[validation['batch'] == batch]

            # Display the filtered data

            # print(batch_data)

            for index, row in batch_data.iterrows():

                

                subject_name= row['subject_name']

                slice= row['slice']


                mri_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()

                mri_numpy =self.normalize_to_255(mri_numpy)


                self.validation_data_mri[batch-1,in_batch,3:3+233,9:9+189,0] = mri_numpy

                



                mask_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()

                mask_numpy=np.where(mask_numpy > 0, 1, 0)


                self.validation_data_mask[batch-1,in_batch,3:3+233,9:9+189,0] = mask_numpy

                in_batch +=1
    











    def coronal_read_and_load_data_in_numpy(self,train_csv,validation_csv,data_directory,fold):

        self.fold=fold



        ## READ TRAINING DATA 


        train = pd.read_csv(train_csv)

        number_of_batch=train['batch'].max()



        self.train_data_mri=np.zeros((number_of_batch,32,208,208,1),dtype=np.float32)

        self.train_data_mask=np.zeros((number_of_batch,32,208,208,1),dtype=np.float32)




        for batch in range(1,number_of_batch+1):


            in_batch=0


            # Filter rows based on the batch number

            batch_data = train[train['batch'] == batch]

            # Display the filtered data

            # print(batch_data)
            print(batch_data)

            for index, row in batch_data.iterrows():

                

                subject_name= row['subject_name']

                slice= row['slice']


                mri_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()

                mri_numpy =self.normalize_to_255(mri_numpy)

                self.train_data_mri[batch-1,in_batch,5:5+197,9:9+189,0] = mri_numpy

                


                mask_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()

                mask_numpy=np.where(mask_numpy > 0, 1, 0)


                self.train_data_mask[batch-1,in_batch,5:5+197,9:9+189,0] = mask_numpy

                in_batch +=1











        validation = pd.read_csv(validation_csv)

        number_of_batch=validation['batch'].max()



        self.validation_data_mri=np.zeros((number_of_batch,32,208,208,1),dtype=np.float32)

        self.validation_data_mask=np.zeros((number_of_batch,32,208,208,1),dtype=np.float32)




        for batch in range(1,number_of_batch+1):


            in_batch=0


            # Filter rows based on the batch number

            batch_data = validation[validation['batch'] == batch]

            # Display the filtered data

            # print(batch_data)

            for index, row in batch_data.iterrows():

                

                subject_name= row['subject_name']

                slice= row['slice']


                mri_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()

                mri_numpy =self.normalize_to_255(mri_numpy)


                self.validation_data_mri[batch-1,in_batch,5:5+197,9:9+189,0] = mri_numpy

                



                mask_nii=nib.load(f'{data_directory}/{subject_name}/{subject_name}_slice_{slice}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()

                mask_numpy=np.where(mask_numpy > 0, 1, 0)


                self.validation_data_mask[batch-1,in_batch,5:5+197,9:9+189,0] = mask_numpy

                in_batch +=1
    
    


    def saving(self,saved_directory):



        saved_directory = f"{saved_directory}/fold_{self.fold}"
        
        
        # Check if the directory exists
        if not os.path.exists(saved_directory):
            # If not, create the directory
            os.makedirs(saved_directory)
            print(f"Directory {saved_directory} created.")
        else:
            print(f"Directory {saved_directory} already exists.")





        np.save(f'{saved_directory}/train_data_mri_fold_{self.fold}.npy',self.train_data_mri)
        np.save(f'{saved_directory}/train_data_mask_fold_{self.fold}.npy',self.train_data_mask)
        np.save(f'{saved_directory}/validation_data_mri_fold_{self.fold}.npy',self.validation_data_mri)
        np.save(f'{saved_directory}/validation_data_mask_fold_{self.fold}.npy',self.validation_data_mask)






        





parser.add_argument('-tcsv', '--train_csv_directory', type=str, help='Train CSV directory.')
parser.add_argument('-vcsv', '--validation_csv_directory', type=str, help='Validation CSV directory')
parser.add_argument('-dd', '--data_directory', type=str, help='The directory of data')
parser.add_argument('-p', '--plane', type=str, help='For which plane')
parser.add_argument('-fold', '--fold', type=int, help='The fold considered as validation fold for naming output')
parser.add_argument('-sd', '--save_directory', type=str, help='Directory to save the output CSV.')



# python3    -tcsv  -vcsv -dd -flod - sd 








args = parser.parse_args()







load_data_class=load_data()


if args.plane == 'a':


    load_data_class.axial_read_and_load_data_in_numpy(train_csv=args.train_csv_directory,validation_csv=args.validation_csv_directory,data_directory=args.data_directory,
                                                fold=args.fold)

    load_data_class.saving(saved_directory=args.save_directory)





if args.plane == 's':


    load_data_class.sagittal_read_and_load_data_in_numpy(train_csv=args.train_csv_directory,validation_csv=args.validation_csv_directory,data_directory=args.data_directory,
                                                fold=args.fold)

    load_data_class.saving(saved_directory=args.save_directory)





if args.plane == 'c':


    load_data_class.coronal_read_and_load_data_in_numpy(train_csv=args.train_csv_directory,validation_csv=args.validation_csv_directory,data_directory=args.data_directory,
                                                fold=args.fold)

    load_data_class.saving(saved_directory=args.save_directory)




#  python3 load_data_with_batch_distribution.py -tcsv ../Distribution_Batch/Axial_Batches_CSV/fold_1/Train_lesion_normal_batches.csv -vcsv ../Distribution_Batch/Axial_Batches_CSV/fold_1/Validation_lesion_normal_batches.csv -dd ../Moving_Converting_Data/axial_2D/ -fold 1 -sd ./Axial


#  python3 load_data_with_batch_distribution.py -tcsv ../Distribution_Batch/Sagittal_Batches_CSV/fold_1/Train_lesion_normal_batches.csv -vcsv ../Distribution_Batch/Sagittal_Batches_CSV/fold_1/Validation_lesion_normal_batches.csv -dd ../Moving_Converting_Data/sagittal_2D/ -p s -fold 1 -sd ./Axial













