#This code was written by Liamirpy from the Loneliness Land.

# This code is for calculating the size of the lesion for each patient
## and categorized the patient based on the lesion size
##
## WHY WE WANT TO DO THIS?
## The resion is that for training and evalution model it is important to
## have a lesion in various sizes to the evalutation be fair and reliable.


''' 
                            Procedure :

    1- Calculating the number of lesion and the size of lesion (The number of voxel):

            The voxels that are connected to each other are a single lesion

            OUTPUT:

            for all the patient ( M ) 

            number of lesions (N), [first_lesion_size, ....,Nth_lesion_size]= patient M 

    2- Generating the CSV file like this :

                N : Is the number of lesion ( In this dataset we know the maximum number of lesion is 8

                The_subject_name | number_of_lesion |max_size | The lesion_size_1 | ........ |The lesion_size_N 

                -------- This step is done by ccalculate_3D_distribution__________
    

    ATTENTION_1: The first and second steps are calcuated in combined code and calcuating the number of
                connected component and size of each that and save in csv file(step 2).

    
    ATTENTION_2: The first and second step calculated the lesion size and saved as a csv file.
                 Based on this csv file we are able to plot the lesion size distribution for 3D MRI.

                 The csv file is lesion_distribution_3D.csv

    3- Because our model will train on 2D images we need to calculate the lesion size distribution 
       for axial,sagittal, and coronal.

                ---------- This step is done by the calculate_axial[or sagittal or coronal]_distribution---------

        The csv files are :
            1- Lesion_distribution_axial.csv
            2- Lesion distribution_sagittal.csv
            3- Lesion_distribution_coronal.csv



'''
import csv
import os
import argparse
import numpy as np
import pandas as pd 
import nibabel as nib
from scipy import ndimage


## First approach





parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, help='The ATLAS Directory : The data should be in ATLAS/Training ')

parser.add_argument('--plane', type=str, help='To choose extracting the plane information. Option : All, Axial,Sagittal ')

args = parser.parse_args()
















class Extract_information:


    def calculate_lesion_sizes_3D(self,mask_file):
        # Load the NIfTI file
        mask_data = nib.load(mask_file).get_fdata()

        # Create a binary mask for lesions (assuming lesions are represented by the value 1)
        lesion_mask = (mask_data >0 ).astype(int)

        # Label connected components in the binary mask
        labeled_lesions, num_lesions = ndimage.label(lesion_mask,structure=np.ones((3,3,3),dtype=np.int8))

        # Dictionary to store the size of each lesion
        lesion_sizes = {}

        # Loop through each labeled lesion and calculate its size
        for label in range(1, num_lesions + 1):
            # Create a mask for the current lesion
            current_lesion_mask = (labeled_lesions == label).astype(int)

            # Calculate the number of voxels in the current lesion
            lesion_size = np.sum(current_lesion_mask)

            # Store the result in the dictionary
            lesion_sizes[label] = lesion_size


        # Calculate the total number of lesions
        num_lesions = len(lesion_sizes)

        sorted_dict_desc = dict(sorted(lesion_sizes.items(), key=lambda item: item[1], reverse=True))


        return num_lesions, sorted_dict_desc




    def calculate_lesion_sizes_2D(self,slice):
        # Load the NIfTI file

        # Create a binary mask for lesions (assuming lesions are represented by the value 1)
        lesion_mask = (slice >0 ).astype(int)

        # Label connected components in the binary mask
        labeled_lesions, num_lesions = ndimage.label(lesion_mask,structure=np.ones((3,3),dtype=np.int8))

        # Dictionary to store the size of each lesion
        lesion_sizes = {}

        # Loop through each labeled lesion and calculate its size
        for label in range(1, num_lesions + 1):
            # Create a mask for the current lesion
            current_lesion_mask = (labeled_lesions == label).astype(int)

            # Calculate the number of voxels in the current lesion
            lesion_size = np.sum(current_lesion_mask)

            # Store the result in the dictionary
            lesion_sizes[label] = lesion_size


        # Calculate the total number of lesions
        num_lesions = len(lesion_sizes)

        sorted_dict_desc = dict(sorted(lesion_sizes.items(), key=lambda item: item[1], reverse=True))


        return num_lesions, sorted_dict_desc











    def Extract_3D_information(self,atlas_directory):

        self.data_list=[]





        self.training_directory= atlas_directory 


        training_main_folders=os.listdir(self.training_directory)
        sorted_training_folders = sorted(training_main_folders)
            

        for folder in sorted_training_folders:
            sub_folders=os.listdir(f'{self.training_directory}/{folder}')
            sorted_sub_folders = sorted(sub_folders)

            for sub_folder in sorted_sub_folders:

                print(sub_folder)


                mask_file= f'{self.training_directory}/{folder}/{sub_folder}/ses-1/anat/{sub_folder}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'

            
            


                lesion_distribution={'subject_name':sub_folder,'number_of_lesion':0,'sum_voxel':0,'max_size':0,'lesion_size_1':0,'lesion_size_2':0,'lesion_size_3':0,'lesion_size_4':0,'lesion_size_5':0,'lesion_size_6':0,'lesion_size_7':0,'lesion_size_8':0,'lesion_size_9':0,'lesion_size_10':0}



                num_lesions, lesion_sizes = self.calculate_lesion_sizes_3D(mask_file)

                lesion_distribution['number_of_lesion']= num_lesions

                sum_voxel=0
                maximum_size=[]

                for label, size in lesion_sizes.items():
                    sum_voxel +=size
                    # print(f"Lesion {label}: {size} voxels")
                    if label < 11:
                        lesion_distribution[f'lesion_size_{label}']=size

                    maximum_size.append(size)
                
                        


                lesion_distribution['sum_voxel']= sum_voxel


                lesion_distribution['max_size']= max(maximum_size)

                

                self.data_list.append(lesion_distribution)





        # CSV file path
        csv_file_path = './CSV/ATLAS_Lesion_information_for_3D_MRI_Subject.csv'

        # Writing data to CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = self.data_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Writing header
            writer.writeheader()

            # Writing data
            writer.writerows(self.data_list)

        print(f'Data has been written to {csv_file_path}.')





########### 2D information 
        



    def axial_lesion_information(self,atlas_directory):

        self.data_list=[]



        self.training_directory= atlas_directory 


        training_main_folders=os.listdir(self.training_directory)
        sorted_training_folders = sorted(training_main_folders)
            

        for folder in sorted_training_folders:
            sub_folders=os.listdir(f'{self.training_directory}/{folder}')
            sorted_sub_folders = sorted(sub_folders)

            for sub_folder in sorted_sub_folders:

                print(sub_folder)


                mask_file= f'{self.training_directory}/{folder}/{sub_folder}/ses-1/anat/{sub_folder}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'

                mask_data = nib.load(mask_file).get_fdata()

                # Create a binary mask for lesions (assuming lesions are represented by the value 1)
                mask_data = (mask_data >0 ).astype(int)
            
                for axial in range((mask_data.shape[2])):


                    axial_slice=mask_data[:,:,axial]


                    lesion_distribution={'subject_name':sub_folder,'slide':0,'number_of_lesion':0,'sum_voxel':0,'max_size':0,'lesion_size_1':0,'lesion_size_2':0,'lesion_size_3':0,'lesion_size_4':0,'lesion_size_5':0,'lesion_size_6':0,'lesion_size_7':0,'lesion_size_8':0,'lesion_size_9':0,'lesion_size_10':0}


                    if sum(sum(axial_slice)) !=0:

                        num_lesions, lesion_sizes = self.calculate_lesion_sizes_2D(axial_slice)

                        lesion_distribution['number_of_lesion']= num_lesions

                        sum_voxel=0
                        maximum_size=[]

                        for label, size in lesion_sizes.items():
                            sum_voxel +=size
                            # print(f"Lesion {label}: {size} voxels")
                            if label < 11:
                                lesion_distribution[f'lesion_size_{label}']=size

                            maximum_size.append(size)
                        
                                
                        lesion_distribution['sum_voxel']= sum_voxel

                        lesion_distribution['slide']= axial

                        lesion_distribution['max_size']= max(maximum_size)

                        self.data_list.append(lesion_distribution)





        # CSV file path
                        
        csv_file_path = './CSV/ATLAS_Axial_Lesion_information.csv'

        # Writing data to CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = self.data_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Writing header
            writer.writeheader()

            # Writing data
            writer.writerows(self.data_list)

        print(f'Data has been written to {csv_file_path}.')




    def sagittal_lesion_information(self,atlas_directory):
            
        self.data_list=[]



        self.training_directory= atlas_directory 


        training_main_folders=os.listdir(self.training_directory)
        sorted_training_folders = sorted(training_main_folders)
            

        for folder in sorted_training_folders:
            sub_folders=os.listdir(f'{self.training_directory}/{folder}')
            sorted_sub_folders = sorted(sub_folders)

            for sub_folder in sorted_sub_folders:

                print(sub_folder)


                mask_file= f'{self.training_directory}/{folder}/{sub_folder}/ses-1/anat/{sub_folder}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'

                mask_data = nib.load(mask_file).get_fdata()

                # Create a binary mask for lesions (assuming lesions are represented by the value 1)
                mask_data = (mask_data >0 ).astype(int)
            
                for sagittal in range((mask_data.shape[0])):


                    sagittal_slice=mask_data[sagittal,:,:]


                    lesion_distribution={'subject_name':sub_folder,'slide':0,'number_of_lesion':0,'sum_voxel':0,'max_size':0,'lesion_size_1':0,'lesion_size_2':0,'lesion_size_3':0,'lesion_size_4':0,'lesion_size_5':0,'lesion_size_6':0,'lesion_size_7':0,'lesion_size_8':0,'lesion_size_9':0,'lesion_size_10':0}


                    if sum(sum(sagittal_slice)) !=0:

                        num_lesions, lesion_sizes = self.calculate_lesion_sizes_2D(sagittal_slice)

                        lesion_distribution['number_of_lesion']= num_lesions

                        sum_voxel=0
                        maximum_size=[]

                        for label, size in lesion_sizes.items():
                            sum_voxel +=size
                            # print(f"Lesion {label}: {size} voxels")
                            if label < 11:
                                lesion_distribution[f'lesion_size_{label}']=size

                            maximum_size.append(size)
                        
                                
                        lesion_distribution['sum_voxel']= sum_voxel

                        lesion_distribution['slide']= sagittal

                        lesion_distribution['max_size']= max(maximum_size)

                        self.data_list.append(lesion_distribution)





        # CSV file path
        csv_file_path = './CSV/ATLAS_Sagittal_Lesion_information.csv'

        # Writing data to CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = self.data_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Writing header
            writer.writeheader()

            # Writing data
            writer.writerows(self.data_list)

        print(f'Data has been written to {csv_file_path}.')





    ## Coronal ##



    def coronal_lesion_information(self,atlas_directory):

        self.data_list=[]



        self.training_directory= atlas_directory 


        training_main_folders=os.listdir(self.training_directory)
        sorted_training_folders = sorted(training_main_folders)
            

        for folder in sorted_training_folders:
            sub_folders=os.listdir(f'{self.training_directory}/{folder}')
            sorted_sub_folders = sorted(sub_folders)

            for sub_folder in sorted_sub_folders:

                print(sub_folder)


                mask_file= f'{self.training_directory}/{folder}/{sub_folder}/ses-1/anat/{sub_folder}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'

                mask_data = nib.load(mask_file).get_fdata()

                # Create a binary mask for lesions (assuming lesions are represented by the value 1)
                mask_data = (mask_data >0 ).astype(int)
            
                for coronal in range((mask_data.shape[1])):


                    coronal_slice=mask_data[:,coronal,:]


                    lesion_distribution={'subject_name':sub_folder,'slide':0,'number_of_lesion':0,'sum_voxel':0,'max_size':0,'lesion_size_1':0,'lesion_size_2':0,'lesion_size_3':0,'lesion_size_4':0,'lesion_size_5':0,'lesion_size_6':0,'lesion_size_7':0,'lesion_size_8':0,'lesion_size_9':0,'lesion_size_10':0}


                    if sum(sum(coronal_slice)) !=0:

                        num_lesions, lesion_sizes = self.calculate_lesion_sizes_2D(coronal_slice)

                        lesion_distribution['number_of_lesion']= num_lesions

                        sum_voxel=0
                        maximum_size=[]

                        for label, size in lesion_sizes.items():
                            sum_voxel +=size
                            # print(f"Lesion {label}: {size} voxels")
                            if label < 11:
                                lesion_distribution[f'lesion_size_{label}']=size

                            maximum_size.append(size)
                        
                                
                        lesion_distribution['sum_voxel']= sum_voxel

                        lesion_distribution['slide']= coronal

                        lesion_distribution['max_size']= max(maximum_size)

                        self.data_list.append(lesion_distribution)





        # CSV file path
        csv_file_path = './CSV/ATLAS_Coronal_Lesion_information.csv'

        # Writing data to CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = self.data_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Writing header
            writer.writeheader()

            # Writing data
            writer.writerows(self.data_list)

        print(f'Data has been written to {csv_file_path}.')






Extract_information_class=Extract_information()



if args.plane == 'axial':

    Extract_information_class.axial_lesion_information(args.directory)

elif args.plane == 'sagittal':

    Extract_information_class.sagittal_lesion_information(args.directory)

elif args.plane == 'coronal':

    Extract_information_class.coronal_lesion_information(args.directory)

elif args.plane == 'all':

    Extract_information_class.Extract_3D_information(args.directory)


