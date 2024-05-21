#### This code is for calculating the lesion distribution for each planes.


'''     Procedure:

    1- Read the training_validation_data_3D_80_percentage.csv file 
    2- For each patient find the slices in each planes that have lesion
    3- For each planes creat a csv file and put all that slices in that and save it.
            
'''

import csv



def read_csv_file_and_find_subjects_slides(filename,target_subject):
    data_list = [] # Dictionary to store the data

    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            if row['subject_name'] == target_subject:
                subject_name = row['subject_name']
                slide = row['slide']
                number_of_lesion = row['number_of_lesion']
                sum_voxel = row['sum_voxel']
                max_size = row['max_size']

                
                # Store the data in the dictionary
                data_list.append({'subject_name':subject_name,'slide':slide, 'number_of_lesion': number_of_lesion,'sum_voxel':sum_voxel,'max_size':max_size})
    
    return data_list







import pandas as pd 

import csv 

import random




    # Read the CSV file


def find_the_training_slides(axial_csv_file,sagittal_csv_file,coronal_csv_file):

    training_validation_data = './CSV/Lesion_information_for_80_percent_of_3D_MRI_Subject.csv'


    all_train_axial_slides=[]
    all_train_sagittal_slides=[]
    all_train_coronal_slides=[]


    with open(training_validation_data, 'r') as csvfile:
        reader = csv.reader(csvfile)
            
        
        next(reader)

        for row in reader:

            print(row[0])

            all_train_axial_slides +=read_csv_file_and_find_subjects_slides(axial_csv_file,row[0])
            all_train_sagittal_slides +=read_csv_file_and_find_subjects_slides(sagittal_csv_file,row[0])
            all_train_coronal_slides +=read_csv_file_and_find_subjects_slides(coronal_csv_file,row[0])



    lesion_axial_distribution = './CSV/Axial_Lesion_information_for_80_percent_of_3D_MRI_Subject.csv'

            # Writing data to CSV file
    with open(lesion_axial_distribution, 'w', newline='') as csvfile:
        fieldnames = all_train_axial_slides[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Writing header
        writer.writeheader()

        # Writing data
        writer.writerows(all_train_axial_slides)

    print(f'Data has been written to {lesion_axial_distribution}.')




    lesion_sagittal_distribution = './CSV/Sagittal_Lesion_information_for_80_percent_of_3D_MRI_Subject.csv'

            # Writing data to CSV file
    with open(lesion_sagittal_distribution, 'w', newline='') as csvfile:
        fieldnames = all_train_axial_slides[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Writing header
        writer.writeheader()

        # Writing data
        writer.writerows(all_train_sagittal_slides)

    print(f'Data has been written to {lesion_sagittal_distribution}.')





    lesion_coronal_distribution = './CSV/Coronal_Lesion_information_for_80_percent_of_3D_MRI_Subject.csv'

            # Writing data to CSV file
    with open(lesion_coronal_distribution, 'w', newline='') as csvfile:
        fieldnames = all_train_axial_slides[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Writing header
        writer.writeheader()

        # Writing data
        writer.writerows(all_train_coronal_slides)

    print(f'Data has been written to {lesion_coronal_distribution}.')










find_the_training_slides('../Dataset/CSV/ATLAS_Axial_Lesion_information.csv','../Dataset/CSV/ATLAS_Sagittal_Lesion_information.csv','../Dataset/CSV/ATLAS_Coronal_Lesion_information.csv')






