#### This code is for spliting the dataset for training the models and also for evaluation of models;


'''     Procedure:

    1- Spliting the 3D MRI cases in to two group: 
        
            - 80% : For training and evaluation 2D classification and Segmentation models 
                    - The models will train and evaluate for 2D images
                    - 2D images are in three planes (axial,sagittal, coronal)

                    ------- In this step the evaluation models are based on k-fold -------------
                             Training_validation_3D.csv   

            - 20% : Evaluating the model for 3D MRI images 

                        -In the first step our model will train on 2D images for 
                            each planes(axial,sagittal,coronal). After that we use 
                            the aggregation function for creating and correcting 
                            the final 3D mask prediction.
                            (This part is kind of combination of all planes)
                        
                        - For evaluating the last part of proposed method(aggregation function)
                          we need the number of 3D MRI images that models not seen 
                          during the training procedure. 
                        
                        Test_3D.csv

            ATTENTION: 

                    ---- For all of spliting part we use the distribution of lesion size ----


                
'''
import pandas as pd 

import csv 

import random





def categorized_3D_MRI_to_each_bins():


    bins = [0,5000,30000,60000,100000,200000,float('inf')]

    ### Put each cases in the bin group that it's blongs and save final csv as train_validation_3D.csv

    csv_file_path = '../Dataset/CSV/ATLAS_Lesion_information_for_3D_MRI_Subject.csv'



    data_bins={'data_bin_0':[],'data_bin_1':[],'data_bin_2':[],'data_bin_3':[],'data_bin_4':[],'data_bin_5':[]}

    # Create an empty dictionary to store the data
    data_bin = {'subject_name':'','sum_voxel':0}

    # Read the CSV file

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        # Iterate over each row in the CSV file
        next(reader)

        for row in reader:
            data_bin = {'subject_name':'','sum_voxel':0}

            sum_voxel=int(row[2])

            subject_name=row[0]

            if bins[0] < sum_voxel <= bins[1]:

                data_bin ['subject_name']= subject_name
                data_bin ['sum_voxel']= sum_voxel

                data_bins['data_bin_0'].append(data_bin)



            elif bins[1] < sum_voxel <= bins[2]:
                            
                data_bin ['subject_name']= subject_name
                data_bin ['sum_voxel']= sum_voxel

                data_bins['data_bin_1'].append(data_bin)


            elif bins[2] < sum_voxel <= bins[3]:

                data_bin ['subject_name']=subject_name
                data_bin ['sum_voxel']= sum_voxel

                data_bins['data_bin_2'].append(data_bin)


            elif bins[3]< sum_voxel <=bins[4]:

                data_bin ['subject_name']= subject_name
                data_bin ['sum_voxel']= sum_voxel

                data_bins['data_bin_3'].append(data_bin)

            
            elif bins[4] < sum_voxel <= bins[5]:
                            
                data_bin ['subject_name']= subject_name
                data_bin ['sum_voxel']= sum_voxel

                data_bins['data_bin_4'].append(data_bin)


            
            elif bins[5] < sum_voxel <= bins[6]:

                data_bin ['subject_name']= subject_name
                data_bin ['sum_voxel']= sum_voxel

                data_bins['data_bin_5'].append(data_bin)


    return data_bins





data_bins=categorized_3D_MRI_to_each_bins()




### Based on the data bins randomly choose the 20% of each bins data as save it in csv file 
### The rest cases save as Training validation csv file 






sampled_data_bins_20 = {}
remaining_data_bins_80 = {}

# Specify the percentage to sample (20% in this case)
percentage_to_sample = 20

# Iterate over each bin in the data_bins dictionary
for bin_key, bin_values in data_bins.items():
    # Calculate the number of elements to sample (20% of the total)
    num_elements_to_sample = round(len(bin_values) * percentage_to_sample / 100)

    # Randomly sample 20% of the data
    sampled_data = random.sample(bin_values, num_elements_to_sample)

    # Store the sampled data in the new dictionary
    sampled_data_bins_20[bin_key] = sampled_data

    # Store the remaining data in the other dictionary
    remaining_data = [value for value in bin_values if value not in sampled_data]
    remaining_data_bins_80[bin_key] = remaining_data

# Print the resulting dictionaries



### Creating the single csv file as a final result 
concatinated_all_20_sample=[]
concatinated_all_80_sample=[]

for key,value in sampled_data_bins_20.items():

    concatinated_all_20_sample.extend(value)


for key,value in remaining_data_bins_80.items():

    concatinated_all_80_sample.extend(value)






csv_file_path = f'./CSV/Lesion_information_for_20_percent_of_3D_MRI_Subject.csv'

    # Writing data to CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = concatinated_all_20_sample[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Writing header
    writer.writeheader()

    # Writing data
    writer.writerows(concatinated_all_20_sample)

print(f'Data has been written to {csv_file_path}.')







csv_file_path = f'./CSV/Lesion_information_for_80_percent_of_3D_MRI_Subject.csv'

    # Writing data to CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = concatinated_all_80_sample[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Writing header
    writer.writeheader()

    # Writing data
    writer.writerows(concatinated_all_80_sample)

print(f'Data has been written to {csv_file_path}.')


















