import numpy as np
from keras.models import load_model
import csv

from tensorflow import keras
from statistic import Statistics
stats = Statistics()



import os



for loss in ["FocalTverskyLoss","FocalLoss","TverskyLoss","DiceCoefLoss"]:


    metric_names=['hausdorff_distance', 'dice_coef', 'recall', 'precision']

    # Load the model


    if loss=="FocalTverskyLoss":

        model = load_model('focaltversky.h5',custom_objects={'FocalTverskyLoss':stats.FocalTverskyLoss,'dice_coef': stats.dice_coef,
                                                    'precision':stats.precision,'recall':stats.recall,
                                                    'hausdorff_distance':stats.hausdorff_distance})
    
    if loss=="FocalLoss":

        model = load_model('focalloss.h5',custom_objects={'FocalLoss':stats.FocalLoss,'dice_coef': stats.dice_coef,
                                                    'precision':stats.precision,'recall':stats.recall,
                                                    'hausdorff_distance':stats.hausdorff_distance})



    if loss=="TverskyLoss":

        model = load_model('tverskyloss.h5',custom_objects={'TverskyLoss':stats.TverskyLoss,'dice_coef': stats.dice_coef,
                                                    'precision':stats.precision,'recall':stats.recall,
                                                    'hausdorff_distance':stats.hausdorff_distance})



    if loss=="DiceCoefLoss":

        model = load_model('dice.h5',custom_objects={'DiceCoefLoss':stats.DiceCoefLoss,'dice_coef': stats.dice_coef,
                                                   'precision':stats.precision,'recall':stats.recall,
                                                    'hausdorff_distance':stats.hausdorff_distance})





    # Load the input data and mask data
    input_data = np.load('../Data_Preparation/Axial/fold_3/validation_data_mri_fold_3.npy')
    mask_data = np.load('../Data_Preparation/Axial/fold_3/validation_data_mask_fold_3.npy')

    # Open a CSV file to write the results



    

        # Iterate over each sample
    all_val_results =[]

    for val_batch in range(input_data.shape[0]):
                    
            inint_dic={'hausdorff_distance':0, 'dice_coef':0, 'recall':0, 'precision':0}


            batch_results = model.test_on_batch(

                input_data[val_batch], mask_data[val_batch], return_dict=True
            )
            for name, value in batch_results.items():
                inint_dic[name]=value
            all_val_results.append(inint_dic)


    with open(f'{loss}.csv', 'w', newline='') as csvfile:
                    fieldnames = all_val_results[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    # Writing header
                    writer.writeheader()

                    # Writing data
                    writer.writerows(all_val_results)



