import pandas as pd

import os
import numpy as np
import csv 



# Specify the directory path

# Get a list of all folders (directories) in the specified directory









class plotting:

    def __init__(self,
                 
                 results_directory='',

                 saving_plot_directory='./plot',
                   
                 ):
        


        self.results_directory=results_directory
        self.saving_plot_directory=saving_plot_directory

        self.subjects = [folder for folder in os.listdir(self.results_directory) if os.path.isdir(os.path.join(self.results_directory, folder))]

        self.plot_plane('axial')

        self.plot_plane('sagittal')

        self.plot_plane('coronal')

        self.plot_3d()
        






    def plot_plane(self,plan):

        dice_before_fusion=[]
        precision_before_fusion=[]
        recall_before_fusion=[]

        dice_after_fusion=[]
        precision_after_fusion=[]
        recall_after_fusion=[]

        for subject in self.subjects:


            df = pd.read_csv(f'{self.results_directory}/{subject}/{plan}/comparison_before_after_fusion.csv', index_col=0)

            means_dice_before_fusion = df.loc['before_mean', 'dice']
            dice_before_fusion.append(means_dice_before_fusion)

            means_dice_after_fusion = df.loc['after_mean', 'dice']
            dice_after_fusion.append(means_dice_after_fusion)



            means_precision_before_fusion = df.loc['before_mean', 'presision']
            precision_before_fusion.append(means_precision_before_fusion)

            means_precision_after_fusion = df.loc['after_mean', 'presision']
            precision_after_fusion.append(means_precision_after_fusion)




            means_recall_before_fusion = df.loc['before_mean', 'recall']
            recall_before_fusion.append(means_recall_before_fusion)


            means_recall_after_fusion = df.loc['after_mean', 'recall']
            recall_after_fusion.append(means_recall_after_fusion)






        



    


    # def cal_3d_results(self):


    #     dice_fusion=[]
    #     precision_fusion=[]
    #     recall_fusion=[]

    #     lesion_f1_score=[]
    #     simple_lesion_count=[]
    #     volume_difference=[]


    #     subjects_results=[]



    #     for subject in self.subjects:

    #         df = pd.read_csv(f'{directory_path}/{subject}/fusion/fusion_evaluation.csv')

    #         df.columns = df.columns.str.strip()


    #         dice=df['dice']

    #         dice_fusion.append(dice)



    #         precision=df['presision']
    #         precision_fusion.append(precision)


    #         recall=df['recall']
    #         recall_fusion.append(recall)




    #         volume__diff=df['volume_difference']
    #         volume_difference.append(volume__diff)

    #     # if (dice.to_numpy())[0] > 0.5:

    #         result={'subject_name':subject,'dice':(dice.to_numpy())[0],
    #                 'precision':(precision.to_numpy())[0],
    #                 'recall':(recall.to_numpy())[0],
    #                 'volum_diff':(volume__diff.to_numpy())[0]
    #                 }
    #         subjects_results.append(result)





    #     csv_file = f'./{self.saving_final_results_directory}/fusion_subject_results.csv'

    #     with open(csv_file, 'w', newline='') as csvfile:
    #         fieldnames = subjects_results[0].keys()
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #         # Writing header
    #         writer.writeheader()

    #         # Writing data
    #         writer.writerows(subjects_results)

    #     print(f'Data has been written to {csvfile}.')





        

    #     results={'Mean_dice_fusion':np.mean(dice_fusion),
    #              'Std_dice_fusion':np.std(dice_fusion),
                 
    #              'Mean_precision_fusion':np.mean(precision_fusion),
    #              'Std_precision_fusion':np.std(precision_fusion),



    #              'Mean_recall_fusion':np.mean(recall_fusion),
    #              'Std_recall_fusion':np.std(recall_fusion),

    #              'Mean_volume_difference_fusion':np.mean(volume_difference),
    #              'Std_volume_difference_fusion':np.std(volume_difference)}
        

    #     csv_file = f'{self.saving_final_results_directory}/Fusion_results.csv'

    #     with open(csv_file, mode='w', newline='') as file:
    #         writer = csv.DictWriter(file, fieldnames=results.keys())
    #         writer.writeheader()
    #         writer.writerow(results)

    #     print(f"Data has been written to {csv_file}")



# directory_path = './Results'


# plotting(directory_path,'./plots')             







