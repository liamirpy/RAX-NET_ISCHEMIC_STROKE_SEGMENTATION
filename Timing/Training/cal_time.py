### Cal the time of training 



import pandas as pd
import os 
import numpy as np
import pandas as pd

# Define the models, planes, and number of folds
models = ['X_net', 'CLCI']
planes = ['axial', 'sagittal', 'coronal']
num_folds = 5

# List to store the results
results = []

# Loop through each model, plane, and fold
for model in models:
    for plane in planes:
        for fold in range(1, num_folds + 1):

            # Define the file paths for the CSV files
            val_loss_file_path = f'./{model}/{plane}/fold_{fold}/{plane}_fold_{fold}.csv'
            epoch_time_file_path = f'./{model}/{plane}/fold_{fold}/epoch_times.csv'

            # Read the val_loss CSV file
            df_val_loss = pd.read_csv(val_loss_file_path, delimiter=';')

            # Find the index of the minimum value in the 'val_loss' column
            min_val_loss_index = df_val_loss['val_loss'].idxmin()

            # Read the epoch_time CSV file
            df_epoch_time = pd.read_csv(epoch_time_file_path, delimiter=';')

            # Check for combined 'Epoch,Time' column and split if necessary
            if 'Epoch,Time' in df_epoch_time.columns:
                df_epoch_time[['Epoch', 'Time']] = df_epoch_time['Epoch,Time'].str.split(',', expand=True)

            # Extract the time value corresponding to the minimum val_loss index
            df_epoch_time['Time'] = pd.to_numeric(df_epoch_time['Time'], errors='coerce')

            # Calculate the sum of time values up to the minimum val_loss index
            total_time_value = df_epoch_time.loc[:min_val_loss_index, 'Time'].sum()


            # Append the result to the list
            results.append({
                'Model': model,
                'Plane': plane,
                'Fold': fold,
                'Time': total_time_value / 3600
            })



# For RAX net 
for model in ['RAX_NET']:
    for plane in ['axial', 'sagittal', 'coronal']:
        for fold in range(1, 6):

            # Read the CSV file
            file_path = f'./{model}/{plane}/fold_{fold}/{plane}_fold_{fold}.csv'
            df = pd.read_csv(file_path, delimiter=',')

            # Find the minimum value in the 'val_loss' column
            min_val_loss = df['val_loss'].min()

            # Find the index of the minimum value
            min_index = df['val_loss'].idxmin()

            # Sum the time values up to the min_index
            time_values = df.loc[:min_index, 'time'].tolist()
            total_time = np.sum(time_values)

            # Append these results to the same list
            results.append({
                'Model': model,
                'Plane': plane,
                'Fold': fold,
                'Time': total_time / 3600
            })

            print(f"Model: {model}, Plane: {plane}, Fold: {fold}, Min Val Loss: {min_val_loss}, Min Index: {min_index}, Total Time: {total_time}")

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# # Write the results to a CSV file
# results_df.to_csv('results_summary.csv', index=False)

# print("Results have been written to 'results_summary.csv'.")


average_times = results_df.groupby(['Model', 'Plane'])['Time'].mean().reset_index()
average_times = average_times.rename(columns={'Time': 'Average Time'})

# Merge the average times back into the results DataFrame
results_df = pd.merge(results_df, average_times, on=['Model', 'Plane'])

# Write the results to a CSV file
results_df.to_csv('training_time_summary.csv', index=False)

print("Results have been written to 'results_summary.csv'.")