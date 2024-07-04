#### This code is for spliting the dataset to batches based on the lesion distribution in the folds.

'''     Procedure:

    1- Read all the csv files of folds
    2- Categorized the data based on the defined bins
    3-Calculate the propotion of each bins 
    4- Calculate the propotion of each bin data for the single batch
    5- Return CSV  



    ### The fold directory should contain all the fold csv files and end with fold_{number}.csv ex: axial_fold_01.csv

    ### Maximum number of fold is 10

 

                
'''
import pandas as pd 
import re
import os
import csv 
import random
import glob
import argparse
import random
import ast

random.seed(42)
import argparse

parser = argparse.ArgumentParser()





class DistributionBatch:

    def __init__(self,
                 
                 batch_size=32,

                 lesion_folds_csv_directory='.',

                 normal_folds_csv_directory='.',

                 based_on_which_column='',

                 validation_fold=[1],

                 bins=[0,500,1000],

                 allowed_reapeted_data_in_different_batches=False,

                 data_augmentation_to_balance_distribution=False,

                 allowed_reapeted_data_in_single_batch=False,


                 augmentation_list=['crop','v_filp','h_flip','style_transfer'],
                                  
                 save_csv=False,

                 save_csv_directory='./batches',

                 ):


        ## It should be a name of column in the csv file
        self.based_on=based_on_which_column

        self.batch_size= int(batch_size /2)

        self.normal_csv_folder_directory=normal_folds_csv_directory


        self.allowed_reapeted_data_in_different_batches=allowed_reapeted_data_in_different_batches

        self.data_augmentation=data_augmentation_to_balance_distribution


        self.allowed_reapeted_data_in_single_batch= allowed_reapeted_data_in_single_batch
        


        self.save_csv=save_csv

        self.csv_directory=lesion_folds_csv_directory


        self.augmentation_list=augmentation_list



        self.total_train_data=0

        self.total_validation_data=0

        self.valid_directory=False

        self.train_calss_balanced=False

        self.validation_class_balanced=False

        self.save_csv_directory=save_csv_directory

        self.validation_fold=validation_fold

        self.bins=bins

        self.number_of_bins= len(self.bins) -1 

        #### {'bin_1':[{subject: , 'slice: ... }]}


        self.train_bins_data={}

        self.validation_bins_data={}


        self.number_of_data_in_each_train_bin={}
        self.number_of_data_in_each_validation_bin={}




        # {'bin_1': number of data in self.Train_bins_data['bin_1'] /   total number of data in all bins}
        self.train_bins_propotion={}

        self.validation_bins_propotion={}



        ## {'bin_1': train_bins_propotion * batch_size}
        self.number_data_from_each_bin_for_train_batch={}
        self.number_data_from_each_bin_for_validation_batch={}





        self.Train_batch={}

        self.Validation_batch={}


        self.Train_batch_for_csv={}

        self.Validation_batch_for_csv={}

        ## Read data 
        self.read_data_and_put_in_bins_category()


        ## Calculate the whole data 
        self.cal_total_data()

        if not self.data_augmentation:

            ## Calculate the propotion of each bins 
            self.cal_propotion_data_in_each_bin()
            ## Chanage the propotion to round number and make sure the sum is 100
            self.round_the_propotion_of_bin()

            ## {'bin_1': train_bins_propotion * batch_size}
            self.cal_number_of_data_from_each_bin_for_each_batch()

            self.round_the_propotion_of_number_of_data_from_each_bin_for_each_batch()


            self.number_of_possible_train_batch = int(self.total_train_data / self.batch_size)
            self.number_of_possible_validation_batch = int(self.total_validation_data / self.batch_size)


            self.generating_the_batches()



        if self.data_augmentation:

            self.augmentation()
        
            
            self.cal_total_data()



            ## Calculate the propotion of each bins 
            self.cal_propotion_data_in_each_bin()
            ## Chanage the propotion to round number and make sure the sum is 100
            self.round_the_propotion_of_bin()

            ## {'bin_1': train_bins_propotion * batch_size}
            self.cal_number_of_data_from_each_bin_for_each_batch()

            self.round_the_propotion_of_number_of_data_from_each_bin_for_each_batch()


            self.number_of_possible_train_batch = int(self.total_train_data / self.batch_size)
            self.number_of_possible_validation_batch = int(self.total_validation_data / self.batch_size)


            self.generating_the_batches()




        




        if self.save_csv:
            self.saving_in_csv()
        









    def chech_folder_container(self,csv_directory):

        csv_pattern = r'^.*_fold_\d+\.csv$'

        csv_files=os.listdir(csv_directory)

        for csv_file in csv_files:
            
            if re.match(csv_pattern, csv_file):

                pass

            else:

                raise ValueError("The csv files are not correct. Make sure all the csv file are in the pattern of _fold_01.csv")
        
        self.valid_directory = True




    

    def create_empty_bins_categories(self):


        for bin in range(self.number_of_bins):

            self.train_bins_data[f'bin_{bin+1}'] = []
            self.validation_bins_data[f'bin_{bin+1}'] = []




    
    def find_the_bins_group(self,value):


        for bin in range(self.number_of_bins):



            if self.bins[bin] <= value < self.bins[bin+1]:

                return bin+1
        

        raise ValueError("The value is not available in the range of defined bins. Chech the bins and the value of your data")
    




    def cal_total_data(self):


        self.total_train_data=0


        for _,value in enumerate(self.train_bins_data):

            self.total_train_data += len(self.train_bins_data[value])


            self.number_of_data_in_each_train_bin[value]=len(self.train_bins_data[value])




        
        for _,value in enumerate(self.validation_bins_data):

            self.total_validation_data += len(self.validation_bins_data[value])

            self.number_of_data_in_each_validation_bin[value]=len(self.validation_bins_data[value])


    def cal_propotion_data_in_each_bin(self):

        ## Data Propotion in each bin for the training data ( Formula : number of sample in each bins / total number of data )

 

        for _,value in enumerate(self.train_bins_data):

            self.train_bins_propotion[value] = len(self.train_bins_data[value]) / self.total_train_data



        for _,value in enumerate(self.validation_bins_data):

            self.validation_bins_propotion[value] = len(self.validation_bins_data[value]) / self.total_validation_data





    def round_the_propotion_of_bin(self):

        ### For train bins 

        self.train_bins_propotion = {key: round(value * 100) for key, value in self.train_bins_propotion.items()}

        # Calculate the sum of all values

        total_sum = sum(self.train_bins_propotion.values())

        # Adjust the values to ensure their sum equals 100
        if total_sum != 100:
            # Find the key with the maximum value
            max_key = max(self.train_bins_propotion, key=self.train_bins_propotion.get)
            
            # Adjust the value of the key to make the sum equal to 100
            self.train_bins_propotion[max_key] -= total_sum - 100



        

        ## For test bins 

        self.validation_bins_propotion = {key: round(value * 100) for key, value in self.validation_bins_propotion.items()}

        # Calculate the sum of all values

        total_sum = sum(self.validation_bins_propotion.values())

        # Adjust the values to ensure their sum equals 100
        if total_sum != 100:
            # Find the key with the maximum value
            max_key = max(self.validation_bins_propotion, key=self.validation_bins_propotion.get)
            
            # Adjust the value of the key to make the sum equal to 100
            self.validation_bins_propotion[max_key] -= total_sum - 100


    






            



    def read_data_and_put_in_bins_category(self):




        self.chech_folder_container(self.csv_directory)


        if self.valid_directory:

            number_of_folds=len(os.listdir(self.csv_directory))

            ## Creating the empty bins dictionary

            self.create_empty_bins_categories()


            for fold in range(1,number_of_folds+1):

                if fold != 10:
                    pattern = f'*_fold_0{fold}.csv'
                else:
                    pattern = '*_fold_10.csv'

                try:


                    fold_directory = (glob.glob(os.path.join(self.csv_directory, pattern)))[0]



                    if fold not in self.validation_fold:

                        with open(fold_directory, 'r') as csvfile:

                            reader = csv.DictReader(csvfile)

                            next(reader)

                            for row in reader:

                                try:

                                    based_on_row=row[self.based_on]


                                    num_bin=self.find_the_bins_group(float(based_on_row))

                                    if self.data_augmentation:

                                        row['augmentation']='None'


                                    self.train_bins_data[f'bin_{num_bin}'].append(row)


                                except ValueError as e:
                                    print("The column not exist. Check the number of column")



                                




                    else:

                        with open(fold_directory, 'r') as csvfile:

                            reader = csv.DictReader(csvfile)

                            next(reader)

                            for row in reader:

                                try:

                                    based_on_row=row[self.based_on]



                                    num_bin=self.find_the_bins_group(float(based_on_row))



                                    self.validation_bins_data[f'bin_{num_bin}'].append(row)


                                except ValueError as e:
                                    print("The column not exist. Check the number of column")



                except ValueError as e:
                    print("ERROR: The fold {} is not available. Check the name of files and make sure the numbers are in orders.")

            
           

    

    def cal_number_of_data_from_each_bin_for_each_batch(self):



       


        for _,value in enumerate(self.train_bins_data):

            self.number_data_from_each_bin_for_train_batch[value] = self.train_bins_propotion[value] * self.batch_size / 100


        for _,value in enumerate(self.validation_bins_data):

            self.number_data_from_each_bin_for_validation_batch[value] = self.validation_bins_propotion[value] * self.batch_size / 100

        



    def round_the_propotion_of_number_of_data_from_each_bin_for_each_batch(self):

        ### For train bins 



        self.number_data_from_each_bin_for_train_batch = {key: round(value) for key, value in self.number_data_from_each_bin_for_train_batch.items()}


        

        # Calculate the sum of all values

        total_sum = sum(self.number_data_from_each_bin_for_train_batch.values())

        # Adjust the values to ensure their sum equals 100
        if total_sum != self.batch_size:
            # Find the key with the maximum value
            max_key = max(self.number_data_from_each_bin_for_train_batch, key=self.number_data_from_each_bin_for_train_batch.get)
            
            # Adjust the value of the key to make the sum equal to 100
            self.number_data_from_each_bin_for_train_batch[max_key] -= total_sum - self.batch_size


        ## For test bins 

        self.number_data_from_each_bin_for_validation_batch = {key: round(value ) for key, value in self.number_data_from_each_bin_for_validation_batch.items()}

        # Calculate the sum of all values

        total_sum = sum(self.number_data_from_each_bin_for_validation_batch.values())

        # Adjust the values to ensure their sum equals 100
        if total_sum != self.batch_size:
            # Find the key with the maximum value
            max_key = max(self.number_data_from_each_bin_for_validation_batch, key=self.number_data_from_each_bin_for_validation_batch.get)
            
            # Adjust the value of the key to make the sum equal to 100
            self.number_data_from_each_bin_for_validation_batch[max_key] -= total_sum - self.batch_size

    





    def generating_the_batches(self):

        if not self.data_augmentation:

            if not self.allowed_reapeted_data_in_different_batches and not self.allowed_reapeted_data_in_single_batch:


                availble_train_batch=[]
                availble_validation_batch=[]


                for _,num in enumerate(self.number_data_from_each_bin_for_train_batch):

                    if self.number_data_from_each_bin_for_train_batch[num] !=0:

                        availble_train_batch.append(int(self.number_of_data_in_each_train_bin[num]/self.number_data_from_each_bin_for_train_batch[num] ))


                for _,num in enumerate(self.number_data_from_each_bin_for_validation_batch):

                    if self.number_data_from_each_bin_for_validation_batch[num] !=0:

                        availble_validation_batch.append(int(self.number_of_data_in_each_validation_bin[num]/self.number_data_from_each_bin_for_validation_batch[num] ))


                self.number_of_possible_train_batch=int(min(availble_train_batch))
                self.number_of_possible_validation_batch=int(min(availble_validation_batch))


                for batch in range(self.number_of_possible_train_batch):

                    self.Train_batch[f'batch_{batch+1}'] = []


                for batch in range(self.number_of_possible_validation_batch):

                    self.Validation_batch[f'batch_{batch+1}'] = []  



                    

                for batch in range(self.number_of_possible_train_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_train_batch):

                        num= self.number_data_from_each_bin_for_train_batch[value]

                

                        selected_values= random.sample(self.train_bins_data[value], num)


                        self.Train_batch[f'batch_{batch+1}'] += selected_values


                        for item in selected_values:

                            self.train_bins_data[value].remove(dict(item))



                for batch in range(self.number_of_possible_validation_batch):

                
                    for _,value in enumerate(self.number_data_from_each_bin_for_validation_batch):

                        num= self.number_data_from_each_bin_for_validation_batch[value]


                        selected_values= random.sample(self.validation_bins_data[value], num)


                        self.Validation_batch[f'batch_{batch+1}'] += selected_values

                        for item in selected_values:

                            self.validation_bins_data[value].remove(item)
            







            if self.allowed_reapeted_data_in_different_batches and not self.allowed_reapeted_data_in_single_batch:

                availble_train_batch=[]
                availble_validation_batch=[]


                for _,num in enumerate(self.number_data_from_each_bin_for_train_batch):

                    if self.number_data_from_each_bin_for_train_batch[num] !=0:

                        availble_train_batch.append(int(self.number_of_data_in_each_train_bin[num]/self.number_data_from_each_bin_for_train_batch[num] ))


                for _,num in enumerate(self.number_data_from_each_bin_for_validation_batch):

                    if self.number_data_from_each_bin_for_validation_batch[num] !=0:

                        availble_validation_batch.append(int(self.number_of_data_in_each_validation_bin[num]/self.number_data_from_each_bin_for_validation_batch[num] ))


                self.number_of_possible_train_batch=int(min(availble_train_batch))

                self.number_of_possible_validation_batch=int(min(availble_validation_batch))


                for batch in range(self.number_of_possible_train_batch):

                    self.Train_batch[f'batch_{batch+1}'] = []


                for batch in range(self.number_of_possible_validation_batch):

                    self.Validation_batch[f'batch_{batch+1}'] = []  



                    

                for batch in range(self.number_of_possible_train_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_train_batch):

                        num= self.number_data_from_each_bin_for_train_batch[value]

                

                        selected_values= random.sample(self.train_bins_data[value], num)




                        for selected_value in list(selected_values):

                            if dict(selected_value) in self.Train_batch[f'batch_{batch+1}'] :
                                continue
                            else:
                                self.Train_batch[f'batch_{batch+1}'].append( selected_value)








                for batch in range(self.number_of_possible_validation_batch):


            
                    for _,value in enumerate(self.number_data_from_each_bin_for_validation_batch):

                        num= self.number_data_from_each_bin_for_validation_batch[value]

                

                        selected_values= random.sample(self.validation_bins_data[value], num)




                        for selected_value in list(selected_values):

                            if dict(selected_value) in self.Validation_batch[f'batch_{batch+1}'] :
                                continue
                            else:
                                self.Validation_batch[f'batch_{batch+1}'].append(selected_value)





            if not self.allowed_reapeted_data_in_different_batches and  self.allowed_reapeted_data_in_single_batch:


                availble_train_batch=[]
                availble_validation_batch=[]


                for _,num in enumerate(self.number_data_from_each_bin_for_train_batch):

                    if self.number_data_from_each_bin_for_train_batch[num] !=0:

                        availble_train_batch.append(int(self.number_of_data_in_each_train_bin[num]/self.number_data_from_each_bin_for_train_batch[num] ))


                for _,num in enumerate(self.number_data_from_each_bin_for_validation_batch):

                    if self.number_data_from_each_bin_for_validation_batch[num] !=0:

                        availble_validation_batch.append(int(self.number_of_data_in_each_validation_bin[num]/self.number_data_from_each_bin_for_validation_batch[num] ))


                self.number_of_possible_train_batch=int(min(availble_train_batch))
                self.number_of_possible_validation_batch=int(min(availble_validation_batch))


                for batch in range(self.number_of_possible_train_batch):

                    self.Train_batch[f'batch_{batch+1}'] = []


                for batch in range(self.number_of_possible_validation_batch):

                    self.Validation_batch[f'batch_{batch+1}'] = []  



                    
                for batch in range(self.number_of_possible_train_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_train_batch):

                        num= self.number_data_from_each_bin_for_train_batch[value]

                



                        other_batches=[]

                        for other_batch in range(self.number_of_possible_train_batch):

                            if other_batch != batch:
                                other_batches += self.Train_batch[f'batch_{other_batch+1}']




                        selected_values= random.sample(self.train_bins_data[value], num)




                        selected=0
                        # selected_2=0

                        for selected_value in list(selected_values):

                            if dict(selected_value) not in other_batches:

                                selected +=1

                                self.Train_batch[f'batch_{batch+1}'].append(dict(selected_value))
                        

                        diff= num - selected
                        

                        while(num - selected > 0):
                            # else:
                            extera_select= random.sample(self.train_bins_data[value], 1)[0]
                            # print(extera_select)
                            if dict(extera_select) not in other_batches:
                                selected +=1
                                self.Train_batch[f'batch_{batch+1}'].append(dict(extera_select))
                    







                            






                    
                for batch in range(self.number_of_possible_validation_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_validation_batch):

                        num= self.number_data_from_each_bin_for_validation_batch[value]

                



                        other_batches=[]

                        for other_batch in range(self.number_of_possible_validation_batch):

                            if other_batch != batch:
                                other_batches += self.Validation_batch[f'batch_{other_batch+1}']




                        selected_values= random.sample(self.validation_bins_data[value], num)




                        selected=0
                        # selected_2=0

                        for selected_value in list(selected_values):

                            if dict(selected_value) not in other_batches:

                                selected +=1

                                self.Validation_batch[f'batch_{batch+1}'].append(dict(selected_value))
                        

                        diff= num - selected
                        

                        while(num - selected > 0):
                            # else:
                            extera_select= random.sample(self.validation_bins_data[value], 1)[0]

                            if dict(extera_select) not in other_batches:
                                selected +=1
                                self.Validation_batch[f'batch_{batch+1}'].append(dict(extera_select))
                    



                                

            

            if  self.allowed_reapeted_data_in_different_batches and  self.allowed_reapeted_data_in_single_batch:

                availble_train_batch=[]
                availble_validation_batch=[]


                for _,num in enumerate(self.number_data_from_each_bin_for_train_batch):

                    if self.number_data_from_each_bin_for_train_batch[num] !=0:

                        availble_train_batch.append(int(self.number_of_data_in_each_train_bin[num]/self.number_data_from_each_bin_for_train_batch[num] ))


                for _,num in enumerate(self.number_data_from_each_bin_for_validation_batch):

                    if self.number_data_from_each_bin_for_validation_batch[num] !=0:

                        availble_validation_batch.append(int(self.number_of_data_in_each_validation_bin[num]/self.number_data_from_each_bin_for_validation_batch[num] ))


                self.number_of_possible_train_batch=int(min(availble_train_batch))
                self.number_of_possible_validation_batch=int(min(availble_validation_batch))


                for batch in range(self.number_of_possible_train_batch):

                    self.Train_batch[f'batch_{batch+1}'] = []


                for batch in range(self.number_of_possible_validation_batch):

                    self.Validation_batch[f'batch_{batch+1}'] = []  



                    

                for batch in range(self.number_of_possible_train_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_train_batch):

                        num= self.number_data_from_each_bin_for_train_batch[value]

                

                        selected_values= random.sample(self.train_bins_data[value], num)



                        self.Train_batch[f'batch_{batch+1}'] += selected_values








                for batch in range(self.number_of_possible_validation_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_validation_batch):

                        num= self.number_data_from_each_bin_for_validation_batch[value]

                

                        selected_values= random.sample(self.validation_bins_data[value], num)



                        self.Validation_batch[f'batch_{batch+1}'] += selected_values



        if  self.data_augmentation:



            if not self.allowed_reapeted_data_in_different_batches and not self.allowed_reapeted_data_in_single_batch:



                availble_train_batch=[]
                availble_validation_batch=[]


                for _,num in enumerate(self.number_data_from_each_bin_for_train_batch):

                    if self.number_data_from_each_bin_for_train_batch[num] !=0:

                        availble_train_batch.append(int(self.number_of_data_in_each_train_bin[num]/self.number_data_from_each_bin_for_train_batch[num] ))


                for _,num in enumerate(self.number_data_from_each_bin_for_validation_batch):

                    if self.number_data_from_each_bin_for_validation_batch[num] !=0:

                        availble_validation_batch.append(int(self.number_of_data_in_each_validation_bin[num]/self.number_data_from_each_bin_for_validation_batch[num] ))


                self.number_of_possible_train_batch=int(min(availble_train_batch))

                self.number_of_possible_validation_batch=int(min(availble_validation_batch))


                for batch in range(self.number_of_possible_train_batch):

                    self.Train_batch[f'batch_{batch+1}'] = []


                for batch in range(self.number_of_possible_validation_batch):

                    self.Validation_batch[f'batch_{batch+1}'] = []  



                    

                for batch in range(self.number_of_possible_train_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_train_batch):

                        num= self.number_data_from_each_bin_for_train_batch[value]

                

                        selected_values= random.sample(self.train_bins_data[value], num)


                        self.Train_batch[f'batch_{batch+1}'] += selected_values


                        for item in selected_values:

                            self.train_bins_data[value].remove(dict(item))



                for batch in range(self.number_of_possible_validation_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_validation_batch):

                        num= self.number_data_from_each_bin_for_validation_batch[value]

                

                        selected_values= random.sample(self.validation_bins_data[value], num)


                        self.Validation_batch[f'batch_{batch+1}'] += selected_values


                        for item in selected_values:

                            self.validation_bins_data[value].remove(dict(item))





            if self.allowed_reapeted_data_in_different_batches and not self.allowed_reapeted_data_in_single_batch:

                availble_train_batch=[]
                availble_validation_batch=[]


                for _,num in enumerate(self.number_data_from_each_bin_for_train_batch):

                    if self.number_data_from_each_bin_for_train_batch[num] !=0:

                        availble_train_batch.append(int(self.number_of_data_in_each_train_bin[num]/self.number_data_from_each_bin_for_train_batch[num] ))


                for _,num in enumerate(self.number_data_from_each_bin_for_validation_batch):

                    if self.number_data_from_each_bin_for_validation_batch[num] !=0:

                        availble_validation_batch.append(int(self.number_of_data_in_each_validation_bin[num]/self.number_data_from_each_bin_for_validation_batch[num] ))


                self.number_of_possible_train_batch=int(min(availble_train_batch))

                self.number_of_possible_validation_batch=int(min(availble_validation_batch))


                for batch in range(self.number_of_possible_train_batch):

                    self.Train_batch[f'batch_{batch+1}'] = []


                for batch in range(self.number_of_possible_validation_batch):

                    self.Validation_batch[f'batch_{batch+1}'] = []  



                    

                for batch in range(self.number_of_possible_train_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_train_batch):

                        num= self.number_data_from_each_bin_for_train_batch[value]

                

                        selected_values= random.sample(self.train_bins_data[value], num)



                        selected = 0

                 

                        diff= num - selected
                        

                        while(num - selected > 0):
              
                            find=0

                            extera_select= random.sample(self.train_bins_data[value], 1)[0]

                            if dict(extera_select) not in (self.Train_batch[f'batch_{batch+1}']):
                                print(extera_select)
                                selected +=1

                                self.Train_batch[f'batch_{batch+1}'].append(dict(extera_select))
                            
                           



                for batch in range(self.number_of_possible_validation_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_validation_batch):

                        num= self.number_data_from_each_bin_for_validation_batch[value]

                

                        selected_values= random.sample(self.validation_bins_data[value], num)



                        selected = 0

                 

                        diff= num - selected
                        

                        while(num - selected > 0):
              
                            find=0

                            extera_select= random.sample(self.validation_bins_data[value], 1)[0]

                            if dict(extera_select) not in (self.Train_batch[f'batch_{batch+1}']):

                                selected +=1

                                self.Validation_batch[f'batch_{batch+1}'].append(dict(extera_select))
                            
                    





            if not self.allowed_reapeted_data_in_different_batches and  self.allowed_reapeted_data_in_single_batch:

                availble_train_batch=[]
                availble_validation_batch=[]


                for _,num in enumerate(self.number_data_from_each_bin_for_train_batch):

                    if self.number_data_from_each_bin_for_train_batch[num] !=0:

                        availble_train_batch.append(int(self.number_of_data_in_each_train_bin[num]/self.number_data_from_each_bin_for_train_batch[num] ))


                for _,num in enumerate(self.number_data_from_each_bin_for_validation_batch):

                    if self.number_data_from_each_bin_for_validation_batch[num] !=0:

                        availble_validation_batch.append(int(self.number_of_data_in_each_validation_bin[num]/self.number_data_from_each_bin_for_validation_batch[num] ))


                self.number_of_possible_train_batch=int(min(availble_train_batch))

                self.number_of_possible_validation_batch=int(min(availble_validation_batch))


                for batch in range(self.number_of_possible_train_batch):

                    self.Train_batch[f'batch_{batch+1}'] = []


                for batch in range(self.number_of_possible_validation_batch):

                    self.Validation_batch[f'batch_{batch+1}'] = []  



                    
                for batch in range(self.number_of_possible_train_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_train_batch):

                        num= self.number_data_from_each_bin_for_train_batch[value]

                



                        other_batches=[]

                        for other_batch in range(self.number_of_possible_train_batch):

                            if other_batch != batch:
                                other_batches += self.Train_batch[f'batch_{other_batch+1}']




                        selected_values= random.sample(self.train_bins_data[value], num)




                        selected=0

                        for selected_value in list(selected_values):

                            if dict(selected_value) not in other_batches:

                                selected +=1

                                self.Train_batch[f'batch_{batch+1}'].append(dict(selected_value))
                        

                        diff= num - selected

                        valid_repeated_data=[]
                        

                        while(num - selected > 0):

                            
                            # else:
                            find=0

                            extera_select= random.sample(self.train_bins_data[value], 1)[0]

                            if dict(extera_select) not in other_batches:
                                selected +=1
                                find +=1
                                self.Train_batch[f'batch_{batch+1}'].append(dict(extera_select))

                                valid_repeated_data.append(dict(extera_select))
                            
                            if find==0:

                                try:

                                    extera_select= random.sample(valid_repeated_data, 1)[0]

                                    if dict(extera_select) not in other_batches:
                                        selected +=1
                                        self.Train_batch[f'batch_{batch+1}'].append(dict(extera_select))
                                
                                except:
                                    continue




                for batch in range(self.number_of_possible_validation_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_validation_batch):

                        num= self.number_data_from_each_bin_for_validation_batch[value]

                



                        other_batches=[]

                        for other_batch in range(self.number_of_possible_validation_batch):

                            if other_batch != batch:

                                other_batches += self.Validation_batch[f'batch_{other_batch+1}']




                        selected_values= random.sample(self.validation_bins_data[value], num)




                        selected=0

                        for selected_value in list(selected_values):

                            if dict(selected_value) not in other_batches:

                                selected +=1

                                self.Validation_batch[f'batch_{batch+1}'].append(dict(selected_value))
                        

                        diff= num - selected

                        valid_repeated_data=[]
                        

                        while(num - selected > 0):

                            
                            # else:
                            find=0

                            extera_select= random.sample(self.validation_bins_data[value], 1)[0]

                            if dict(extera_select) not in other_batches:
                                selected +=1
                                find +=1
                                self.Validation_batch[f'batch_{batch+1}'].append(dict(extera_select))

                                valid_repeated_data.append(dict(extera_select))
                            
                            if find==0:

                                try:

                                    extera_select= random.sample(valid_repeated_data, 1)[0]

                                    if dict(extera_select) not in other_batches:
                                        selected +=1
                                        self.Validation_batch[f'batch_{batch+1}'].append(dict(extera_select))
                                
                                except:
                                    continue

                    









                            






                                

            

            if  self.allowed_reapeted_data_in_different_batches and  self.allowed_reapeted_data_in_single_batch:

                availble_train_batch=[]
                availble_validation_batch=[]


                for _,num in enumerate(self.number_data_from_each_bin_for_train_batch):

                    if self.number_data_from_each_bin_for_train_batch[num] !=0:

                        availble_train_batch.append(int(self.number_of_data_in_each_train_bin[num]/self.number_data_from_each_bin_for_train_batch[num] ))


                for _,num in enumerate(self.number_data_from_each_bin_for_validation_batch):

                    if self.number_data_from_each_bin_for_validation_batch[num] !=0:

                        availble_validation_batch.append(int(self.number_of_data_in_each_validation_bin[num]/self.number_data_from_each_bin_for_validation_batch[num] ))


                self.number_of_possible_train_batch=int(min(availble_train_batch))

                self.number_of_possible_validation_batch=int(min(availble_validation_batch))


                for batch in range(self.number_of_possible_train_batch):

                    self.Train_batch[f'batch_{batch+1}'] = []


                for batch in range(self.number_of_possible_validation_batch):

                    self.Validation_batch[f'batch_{batch+1}'] = []  



                    

                for batch in range(self.number_of_possible_train_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_train_batch):

                        num= self.number_data_from_each_bin_for_train_batch[value]

                

                        selected_values= random.sample(self.train_bins_data[value], num)



                        self.Train_batch[f'batch_{batch+1}'] += selected_values




                for batch in range(self.number_of_possible_validation_batch):


                
                    for _,value in enumerate(self.number_data_from_each_bin_for_validation_batch):

                        num= self.number_data_from_each_bin_for_validation_batch[value]

                

                        selected_values= random.sample(self.validation_bins_data[value], num)



                        self.Validation_batch[f'batch_{batch+1}'] += selected_values










    def augmentation(self):


        max_data_in_single_bins = max(self.number_of_data_in_each_train_bin.values())

        which_bin_have_maximum_data = [key for key, value in self.number_of_data_in_each_train_bin.items() if value == max_data_in_single_bins][0]


        number_of_augmented_data_for_each_bins={}

        
        for _,bins in enumerate(self.number_of_data_in_each_train_bin):
            
            number_of_augmented_data_for_each_bins[bins]= max_data_in_single_bins - self.number_of_data_in_each_train_bin[bins]
            
                
    
        if not self.allowed_reapeted_data_in_different_batches and not self.allowed_reapeted_data_in_single_batch:

            for _,value in enumerate(self.train_bins_data):


                if len(self.train_bins_data[value]) !=0 and value != which_bin_have_maximum_data :


                    new_datas=[]

            


                    for aug_method in self.augmentation_list:

                        # print(aug_method)

                        for each_data in self.train_bins_data[value]:


                            if len(new_datas) != (max_data_in_single_bins - len(self.train_bins_data[value])):


                                new_data=each_data.copy()

                                new_data['augmentation']= aug_method



                                new_datas.append(dict(new_data))

                    self.train_bins_data[value] += new_datas


                    if len(self.train_bins_data[value]) != max_data_in_single_bins:

                        raise ValueError(" The number of augmentation method is not enought")
            




        else:

            for _,value in enumerate(self.train_bins_data):



                if len(self.train_bins_data[value]) !=0 and value != which_bin_have_maximum_data:

             

                    while(len(self.train_bins_data[value]) != max_data_in_single_bins):


                        random_samples = dict(random.sample(self.train_bins_data[value], 1)[0])


                        aug_method = random.sample(self.augmentation_list, 1)[0]



                        new_data= random_samples.copy()
                        new_data['augmentation']= aug_method

                        # if random_samples not in self.train_bins_data[value]:



                        self.train_bins_data[value].append(new_data)
        

       

    def saving_in_csv(self):


        ### Saving LESION BATCH

        if not os.path.exists(self.save_csv_directory):
            # If not, create the directory
            os.makedirs(self.save_csv_directory)

            print(f"Directory '{self.save_csv_directory}' created.")
        else:
            print(f"Directory '{self.save_csv_directory}' already exists.")

    
        fieldnames = list(self.Train_batch[next(iter(self.Train_batch))][0].keys())


        fieldnames.append('batch')

        save_directory=self.save_csv_directory+'/fold_'+ str(self.validation_fold[0])

        if not os.path.exists(save_directory):
            # If not, create the directory
            os.makedirs(save_directory)

            print(f"Directory '{save_directory}' created.")
        else:
            print(f"Directory '{save_directory}' already exists.")



        with open(f'./{save_directory}/Train_lesion_batches.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for batch in sorted(self.Train_batch.keys()):
                for entry in self.Train_batch[batch]:
                    entry['batch'] = batch.split('_')[1]  # Extract the batch number
                    writer.writerow(entry)




    
        fieldnames = list(self.Validation_batch[next(iter(self.Validation_batch))][0].keys())


        fieldnames.append('batch')


        with open(f'./{save_directory}/Validation_lesion_batches.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for batch in sorted(self.Validation_batch.keys()):
                for entry in self.Validation_batch[batch]:
                    entry['batch'] = batch.split('_')[1]  # Extract the batch number
                    writer.writerow(entry)
        


        ### SAVING NORMAL BATCH



        csv_files = glob.glob(f'{self.normal_csv_folder_directory}/*.csv')


        save_normal_csv_directory=f'./{save_directory}/Train_normal_batches.csv'
        normal_Train = []

        normal_Validation=[]

        for file in csv_files:
            for val in self.validation_fold:
                if str(val) not in file:
                    normal_Train.append(pd.read_csv(file))

                else:
                    normal_Validation.append(pd.read_csv(file))


        

        ## NUMBER OF TRAIN DATA FOR LESION 





        normal_Train_concatenated = pd.concat(normal_Train, ignore_index=True)

        validation_concatenated = pd.concat(normal_Validation, ignore_index=True)




        normal_Train_concatenated['batch'] = normal_Train_concatenated.index // int(self.batch_size) + 1

        validation_concatenated['batch'] = validation_concatenated.index // int(self.batch_size) + 1


        validation_concatenated.to_csv(f'{save_directory}/Validation_normal_batches.csv', index=False)


        # # # Save the modified DataFrame back to a new CSV file
        normal_Train_concatenated.to_csv(f'{save_directory}/Train_normal_batches.csv', index=False)










        train_lesion_batch_csv=pd.read_csv(f'./{save_directory}/Train_lesion_batches.csv')

        number_of_batch_in_lesion_csv=train_lesion_batch_csv['batch'].max()




        df = pd.read_csv(f'{save_directory}/Train_normal_batches.csv')


        filtered_df = df[df['batch'] <= number_of_batch_in_lesion_csv]


        # # Print or do further processing with the filtered DataFrame
        concatenated_lesion_normal = pd.concat([filtered_df, train_lesion_batch_csv], ignore_index=False)

        # # Save the concatenated DataFrame to a new CSV file
        concatenated_lesion_normal.to_csv(f'{save_directory}/Train_lesion_normal_batches.csv')









        validation_lesion_batch_csv=pd.read_csv(f'./{save_directory}/Validation_lesion_batches.csv')

        number_of_batch_in_normal_csv=validation_lesion_batch_csv['batch'].max()




        df = pd.read_csv(f'{save_directory}/Validation_normal_batches.csv')


        filtered_df = df[df['batch'] <= number_of_batch_in_normal_csv]


        # # Print or do further processing with the filtered DataFrame
        concatenated_lesion_normal = pd.concat([filtered_df, validation_lesion_batch_csv], ignore_index=False)

        # # Save the concatenated DataFrame to a new CSV file
        concatenated_lesion_normal.to_csv(f'{save_directory}/Validation_lesion_normal_batches.csv')




        print("It's done. @Liamirpy")

            




parser.add_argument('-bs', '--batch_size', type=int, help='Batch size.')
parser.add_argument('-lcsv', '--lesion_csv_dir', type=str, help='Directory containing CSV files of lesion slices (format: ..._fold_01.csv).')

parser.add_argument('-ncsv', '--normal_csv_dir', type=str, help='Directory containing CSV files of normal slices (format: ..._fold_01.csv).')
parser.add_argument('-col', '--column', type=str, help='Column in the CSV file to base operations on.')
parser.add_argument('-vf', '--val_fold', type=int, help='Validation fold number.')
parser.add_argument('-bins', '--bins', type=str, help='Bins to be used for data distribution.')
parser.add_argument('-aug', '--augment', type=str, help='Enable data augmentation to balance distribution (True or False).')
parser.add_argument('-auglist', '--augment_list', type=str, help='List of augmentation methods to apply.')
parser.add_argument('-rdb', '--repeat_diff_batches', type=str, help='Allow repeated data across different batches (True or False).')
parser.add_argument('-rsb', '--repeat_same_batch', type=str, help='Allow repeated data within a single batch (True or False).')
parser.add_argument('-sd', '--save_dir', type=str, help='Directory to save the output CSV.')

args = parser.parse_args




args = parser.parse_args()

args.augment = args.augment == 'True'
args.repeat_diff_batches = args.repeat_diff_batches == 'True'
args.repeat_same_batch = args.repeat_same_batch == 'True'

if args.augment_list == 'None':
    args.augment_list = ''

if args.bins:
    args.bins = ast.literal_eval(args.bins)
    args.bins.append(float('inf'))




DistributionBatch(
    batch_size=args.batch_size,

    lesion_folds_csv_directory=args.lesion_csv_dir,

    normal_folds_csv_directory=args.normal_csv_dir,

    based_on_which_column=args.column,

    validation_fold=[args.val_fold],

    bins=args.bins,

    data_augmentation_to_balance_distribution=args.augment,

    allowed_reapeted_data_in_different_batches=args.repeat_diff_batches,

    allowed_reapeted_data_in_single_batch=args.repeat_same_batch,

    augmentation_list=args.augment_list,

    save_csv=True,

    save_csv_directory=args.save_dir
    
)
