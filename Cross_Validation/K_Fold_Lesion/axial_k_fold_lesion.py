## This is LIAMIRPY from the land of loneliness. 

### In this code we split the data for the K-fold validation 
### We consider the distribution of the lesion sizes 
### We consider to put AS MUCH AS possible the lesions from same case to same fold



'''

            Approach:

        1- For each planes put the slices to the bins that blong

        2- Defining the K as the number of fold 

        3- Initialization: Randomly choose the slice from first distibution

        4- Based on the subject name we looking for the same case from other slices

        5- Repeat the second and third step 

        6- If the slice from the case that chosed in initialization step:   

            - Chose randomly from first distribution 
            - Chose randomly from second distribution
            - Looking for the same case from second distribution in other distribution
            - Repeat this approach and if the are not same case we go randomly through the
                rest of distribution group         

                
'''


import csv

import numpy as np

import pandas as pd 

import random

random.seed(42)


axial_bins=[0,200,500,1000,2000,3000,float('inf')]







def put_data_to_bins_axial():

    csv_file_path = '../../Data_Splitting/CSV/Axial_Lesion_information_for_80_percent_of_3D_MRI_Subject.csv'

    data_bins_axial={
        
                'data_bin_1':[],'data_bin_2':[],'data_bin_3':[],'data_bin_4':[],'data_bin_5':[],
                'data_bin_6':[]

            
            }

    # Create an empty dictionary to store the data

    # Read the CSV file

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)


        
        # Iterate over each row in the CSV file
        
        next(reader)

        num_rows = 0
    

    # Iterate over the rows and count them

        for row in reader:
            data_bins = {'subject_name':'','slice':0,'number_of_lesion':0,'sum_voxel':0, 'max_size':0}
            num_rows += 1

            max_size=int(row[4])

            subject_name=row[0]
            slice=row[1]

            number_of_lesion=row[2]
            sum_voxel=int(row[3])

            if axial_bins[0] < sum_voxel <= axial_bins[1]:

                data_bins ['subject_name']= subject_name
                data_bins ['slice']= slice
                data_bins ['number_of_lesion']= number_of_lesion
                data_bins ['sum_voxel']= sum_voxel
                data_bins ['max_size']= max_size

                data_bins_axial['data_bin_1'].append(data_bins)



            elif axial_bins[1] < sum_voxel <= axial_bins[2]:


                data_bins ['subject_name']= subject_name
                data_bins ['slice']= slice
                data_bins ['number_of_lesion']= number_of_lesion
                data_bins ['sum_voxel']= sum_voxel
                data_bins ['max_size']= max_size
                
                
                data_bins_axial['data_bin_2'].append(data_bins)


            elif axial_bins[2] < sum_voxel <= axial_bins[3]:

                data_bins ['subject_name']= subject_name
                data_bins ['slice']= slice
                data_bins ['number_of_lesion']= number_of_lesion
                data_bins ['sum_voxel']= sum_voxel
                data_bins ['max_size']= max_size


                data_bins_axial['data_bin_3'].append(data_bins)


            elif axial_bins[3]< sum_voxel <=axial_bins[4]:

                data_bins ['subject_name']= subject_name
                data_bins ['slice']= slice
                data_bins ['number_of_lesion']= number_of_lesion
                data_bins ['sum_voxel']= sum_voxel
                data_bins ['max_size']= max_size

                data_bins_axial['data_bin_4'].append(data_bins)

            
            elif axial_bins[4] < sum_voxel <= axial_bins[5]:
                            
                data_bins ['subject_name']= subject_name
                data_bins ['slice']= slice
                data_bins ['number_of_lesion']= number_of_lesion
                data_bins ['sum_voxel']= sum_voxel
                data_bins ['max_size']= max_size

                data_bins_axial['data_bin_5'].append(data_bins)


            elif axial_bins[5] < sum_voxel <= axial_bins[6]:
                            
                data_bins ['subject_name']= subject_name
                data_bins ['slice']= slice
                data_bins ['number_of_lesion']= number_of_lesion
                data_bins ['sum_voxel']= sum_voxel
                data_bins ['max_size']= max_size

                data_bins_axial['data_bin_6'].append(data_bins)






    return num_rows,data_bins_axial


number_of_axial_slice,data_bins_axial=put_data_to_bins_axial()



print(number_of_axial_slice)


## Split data for k-fold 


axial_folds={

                'axial_fold_1':[],'axial_fold_2':[],'axial_fold_3':[],'axial_fold_4':[],'axial_fold_5':[],'axial_fold_6':[]
                

            
            }



len_axial_bin=6
number_of_fold=5

number_of_extra_slice=str(number_of_axial_slice/number_of_fold)[-1]



bin_allowness_for_each_fold={
        
                'data_bin_1':int(len(data_bins_axial['data_bin_1'])/number_of_fold),    
                'data_bin_2':int(len(data_bins_axial['data_bin_2'])/number_of_fold) ,  
                'data_bin_3':int(len(data_bins_axial['data_bin_3'])/number_of_fold), 
                'data_bin_4':int(len(data_bins_axial['data_bin_4'])/number_of_fold),  
                'data_bin_5':int(len(data_bins_axial['data_bin_5'])/number_of_fold),   
                'data_bin_6':int(len(data_bins_axial['data_bin_6'])/number_of_fold),        
                                      
            }




size_each_fold= int(len(data_bins_axial['data_bin_1'])/5)\
        +int(len(data_bins_axial['data_bin_2'])/5)\
        +int(len(data_bins_axial['data_bin_3'])/5) \
        +int(len(data_bins_axial['data_bin_4'])/5) \
        + int(len(data_bins_axial['data_bin_5'])/5)\
        + int(len(data_bins_axial['data_bin_6'])/5)



bin_picked_for_each_fold={
        
                'data_bin_1':{'fold_1':0,'fold_2':0,'fold_3':0,'fold_4':0,'fold_5':0},
                'data_bin_2':{'fold_1':0,'fold_2':0,'fold_3':0,'fold_4':0,'fold_5':0},
                'data_bin_3':{'fold_1':0,'fold_2':0,'fold_3':0,'fold_4':0,'fold_5':0},
                'data_bin_4':{'fold_1':0,'fold_2':0,'fold_3':0,'fold_4':0,'fold_5':0},
                'data_bin_5':{'fold_1':0,'fold_2':0,'fold_3':0,'fold_4':0,'fold_5':0},
                'data_bin_6':{'fold_1':0,'fold_2':0,'fold_3':0,'fold_4':0,'fold_5':0}

            }


### Choose randomly from the first distribution 



def find_empty_fold():
 
    for fold in range(1,number_of_fold+1):
            # print(len(axial_folds[f'axial_fold_{fold}']))
        if len(axial_folds[f'axial_fold_{fold}']) < size_each_fold:
            empty_fold=fold
            break

        else:
            empty_fold = 0


    return empty_fold





sum_fold=len(axial_folds['axial_fold_1'])\
    +len(axial_folds['axial_fold_2'])\
    +len(axial_folds['axial_fold_3'])\
    +len(axial_folds['axial_fold_4'])\
    +len(axial_folds['axial_fold_5'])



while( sum_fold < size_each_fold * number_of_fold ): 

    sum_fold=len(axial_folds['axial_fold_1'])\
        +len(axial_folds['axial_fold_2'])\
        +len(axial_folds['axial_fold_3'])\
        +len(axial_folds['axial_fold_4'])\
        +len(axial_folds['axial_fold_5'])


    

    for bins_intialization in range(1,len_axial_bin+1):


        len_bin_n=len(data_bins_axial[f'data_bin_{bins_intialization}'])


        tt=0

        while(len_bin_n > 0): 

            len_bin_n_start=len(data_bins_axial[f'data_bin_{bins_intialization}'])



            initial_random = random.choice(data_bins_axial[f'data_bin_{bins_intialization}'])


            initial_random_subject_name=initial_random['subject_name']



            desired_subject_name = initial_random_subject_name



            for bins_searching in range(1,len_axial_bin+1):

                d=data_bins_axial[f'data_bin_{bins_searching}']

                desired_entries = [entry for entry in d if entry['subject_name'] == desired_subject_name]

                empty_fold=find_empty_fold()


                if empty_fold == 0:

                    break



                for entry in desired_entries:

                    # i


                    if (bin_picked_for_each_fold[f'data_bin_{bins_searching}'][f'fold_{empty_fold}'] ) < bin_allowness_for_each_fold[f'data_bin_{bins_searching}']:

                        item={'subject_name': entry['subject_name'], 'slice': entry['slice'],'number_of_lesion':entry['number_of_lesion'] ,'sum_voxel': entry['sum_voxel'], 'max_size': entry['max_size']}


                        axial_folds[f'axial_fold_{empty_fold}'].append(item)


                        data_bins_axial[f'data_bin_{bins_searching}'].remove(item)

                        bin_picked_for_each_fold[f'data_bin_{bins_searching}'][f'fold_{empty_fold}'] +=1





            len_bin_n=len(data_bins_axial[f'data_bin_{bins_intialization}'])



            if empty_fold==0:
                break


            if (len_bin_n_start - len_bin_n ==0):
                tt +=1
            

        

            
            if ((bin_picked_for_each_fold[f'data_bin_{bins_intialization}'][f'fold_{empty_fold}'] ) == bin_allowness_for_each_fold[f'data_bin_{bins_intialization}']) :
                break



    if empty_fold==0:
        break







for fo in range(1,6):


    axial_slices=axial_folds[f'axial_fold_{fo}']
    # CSV file path
    csv_file_path = f'./Axial_K_Fold_CSV/Axial_Lesion_fold_0{fo}.csv'

    # Writing data to CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = axial_slices[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Writing header
        writer.writeheader()

        # Writing data
        writer.writerows(axial_slices)

    print(f'Data has been written to {csv_file_path}.')










