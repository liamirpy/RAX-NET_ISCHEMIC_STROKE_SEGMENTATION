## This is liamirpy


import numpy as np
import pandas as pd
import os 




def process_csv_files(directory):
    # List to hold dataframes
    data_frames = []

    # Iterate through all CSV files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path)
            data_frames.append((file_name, df))

    # Dictionary to store results
    results = {'file_name': []}
    # Initialize columns for mean and std
    columns = ['dice', 'presision', 'recall', 'hausdorff_distance']
    for col in columns:
        results[col + '_mean'] = []
        results[col + '_std'] = []

    # Process each dataframe
    for file_name, df in data_frames:
        results['file_name'].append(file_name)
        for col in columns:
            results[col + '_mean'].append(df[col].mean())
            results[col + '_std'].append(df[col].std())



    results['file_name'].append('All')

    for col in columns:
        results[col + '_mean'].append(np.mean(results[col + '_mean']))
        results[col + '_std'].append(np.std(results[col + '_mean']))


    
    # for file_name, df in data_frames:
    #     for col in columns:
    #         all_results=[]


    # Create a result dataframe
    result_df = pd.DataFrame(results)

    # Save the result dataframe to a new CSV file
    result_csv_path = os.path.join('.', f'{directory}_folds_Results.csv')
    result_df.to_csv(result_csv_path, index=False)

    print(f"Summary results saved to {result_csv_path}")



directory = 'Axial'  # Replace with the path to your directory
process_csv_files(directory)


directory = 'Sagittal'  # Replace with the path to your directory
process_csv_files(directory)


directory = 'Coronal'  # Replace with the path to your directory
process_csv_files(directory)






            


