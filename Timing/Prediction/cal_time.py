### Cal the time of prediction 



import pandas as pd
import numpy as np
import os

# Define the models, planes, and number of folds
models = ['X_net', 'CLCI', 'RAX_Net']
planes = ['axial', 'sagittal', 'coronal']
num_folds = 5

# List to store the results
results = []

# Loop through each model and plane
for model in models:
    for plane in planes:
        total_time = 0
        total_rows = 0

        # Loop through each fold
        for fold in range(1, num_folds + 1):
            # Define the file path for the CSV file
            file_path = f'./{model}/{plane}/result_of_{plane}_fold_{fold}.csv'

            # Check if the file exists
            if os.path.exists(file_path):
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if the 'time' column exists
                if 'time' in df.columns:
                    # Convert the 'time' column to numeric, in case it's not
                    df['time'] = pd.to_numeric(df['time'], errors='coerce')

                    # Sum the 'time' values and add to the total time
                    total_time += df['time'].sum()
                    total_rows += len(df)  # Count the total number of rows

                else:
                    print(f"Warning: 'time' column not found in {file_path}")
            else:
                print(f"Warning: File not found {file_path}")

        # Calculate the average time for this model and plane
        average_time = total_time / total_rows

        # Append the result to the list
        results.append({
            'Model': model,
            'Plane': plane,
            # 'Total Time': total_time,
            'Average Time': average_time 
        })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Write the results to a CSV file
results_df.to_csv('average_time_summary.csv', index=False)

print("Results have been written to 'average_time_summary.csv'.")
