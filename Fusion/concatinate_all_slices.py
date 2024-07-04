import pandas as pd

import os
import numpy as np
import csv 



# Specify the directory path

# Get a list of all folders (directories) in the specified directory




results_directory='./Results'

subjects = [folder for folder in os.listdir(results_directory) if os.path.isdir(os.path.join(results_directory, folder))]




def concatinate_all(plan):


    all_csv=[]





    for subject in subjects:

            df = pd.read_csv(f'{results_directory}/{subject}/{plan}/comparison_before_after_fusion.csv', index_col=0)

            means_dice_before_fusion = df.loc['before_mean', 'dice']

            means_dice_after_fusion = df.loc['after_mean', 'dice']



            means_precision_before_fusion = df.loc['before_mean', 'presision']

            means_precision_after_fusion = df.loc['after_mean', 'presision']




            means_recall_before_fusion = df.loc['before_mean', 'recall']


            means_recall_after_fusion = df.loc['after_mean', 'recall']


            result={'subject':subject,'mean_dice_before':means_dice_before_fusion,'mean_dice_after':means_dice_after_fusion,
                    'mean_precision_before':means_precision_before_fusion,'mean_precision_after':means_precision_after_fusion,
                    'mean_recall_before':means_recall_before_fusion,'mean_recall_after':means_recall_after_fusion}
            
            all_csv.append(result)
            






    csv_file = f'./overall_results/{plan}_subjects_results.csv'

            # Writing data to CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = all_csv[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Writing header
        writer.writeheader()

        # Writing data
        writer.writerows(all_csv)

    print(f'Data has been written to {csv_file}.')

            






concatinate_all('axial')
concatinate_all('sagittal')
concatinate_all('coronal')





