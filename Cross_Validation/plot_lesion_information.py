import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




class plot_lesion_information:


    def lesion_information_in_one_fold_axial(self,csv_file,ranges):

        csv_file = csv_file
        df = pd.read_csv(csv_file)

        # Extract the desired column
        column_name = 'sum_voxel'  

        data = df[column_name] 


        range_counts = []
        for start, end, color in ranges:
            count = ((data >= start) & (data < end)).sum()
            range_counts.append((start, end, color, count))
    





        plt.figure(figsize=(20, 10))

        # Plot the violin plot
        x_value = 1
        sns.violinplot(x=[x_value] * len(data), y=data, color='darkcyan', fill=True,inner_kws=dict(box_width=5, whis_width=1, color=".1"))

        # Plot the scatter plot with the same x-coordinate
        for start, end, color,count in range_counts:
            mask = (data >= start) & (data < end)
            x_values = np.random.uniform(x_value - 0.1, x_value + 0.1, sum(mask))  # Add slight jitter
            plt.scatter(x_values, data[mask],color=color, label=f'{start} to {end} Pixels (N={count})',s=20,linewidths=0.5)

        # Set the labels for the axes
        plt.xlabel('Lesion Size Distribution in the Single Fold of the Axial Plane',fontsize=20)
        plt.ylabel('volume Pixels', fontsize=20)

        # Display y-axis scale values
        plt.yticks()  # This will keep the y-axis scale values

        # Add legend
        plt.legend()

        # plt.show()

        plt.grid(True)

        plt.savefig('Lesion_information_in_First_Fold_of_Axial_Plane.png', dpi=400)




    def lesion_information_in_one_fold_sagittal(self,csv_file,ranges):

        csv_file = csv_file
        df = pd.read_csv(csv_file)

        # Extract the desired column
        column_name = 'sum_voxel'  

        data = df[column_name] 


        range_counts = []
        for start, end, color in ranges:
            count = ((data >= start) & (data < end)).sum()
            range_counts.append((start, end, color, count))
    





        plt.figure(figsize=(20, 10))

        # Plot the violin plot
        x_value = 1
        sns.violinplot(x=[x_value] * len(data), y=data, color='darkcyan', fill=True,inner_kws=dict(box_width=5, whis_width=1, color=".1"))

        # Plot the scatter plot with the same x-coordinate
        for start, end, color,count in range_counts:
            mask = (data >= start) & (data < end)
            x_values = np.random.uniform(x_value - 0.1, x_value + 0.1, sum(mask))  # Add slight jitter
            plt.scatter(x_values, data[mask],color=color, label=f'{start} to {end} Pixels (N={count})',s=20,linewidths=0.5)

        # Set the labels for the axes
        plt.xlabel('Lesion Size Distribution in Single Fold of the Sagittal Plane',fontsize=20)
        plt.ylabel('volume Pixels', fontsize=20)

        # Display y-axis scale values
        plt.yticks()  # This will keep the y-axis scale values

        # Add legend
        plt.legend()

        # plt.show()

        plt.grid(True)



        plt.savefig('Lesion_information_in_First_Fold_of_Sagittal_Plane.png', dpi=400)





    def lesion_information_in_one_fold_coronal(self,csv_file,ranges):

 
        csv_file = csv_file
        df = pd.read_csv(csv_file)

        # Extract the desired column
        column_name = 'sum_voxel'  

        data = df[column_name] 


        range_counts = []
        for start, end, color in ranges:
            count = ((data >= start) & (data < end)).sum()
            range_counts.append((start, end, color, count))
    





        plt.figure(figsize=(20, 10))

        # Plot the violin plot
        x_value = 1
        sns.violinplot(x=[x_value] * len(data), y=data, color='darkcyan', fill=True,inner_kws=dict(box_width=5, whis_width=1, color=".1"))

        # Plot the scatter plot with the same x-coordinate
        for start, end, color,count in range_counts:
            mask = (data >= start) & (data < end)
            x_values = np.random.uniform(x_value - 0.1, x_value + 0.1, sum(mask))  # Add slight jitter
            plt.scatter(x_values, data[mask],color=color, label=f'{start} to {end} Pixels (N={count})',s=20,linewidths=0.5)

        # Set the labels for the axes
        plt.xlabel('Lesion Size Distribution in Single Fold of the Coronal Plane',fontsize=20)
        plt.ylabel('volume Pixels', fontsize=20)

        plt.yticks()  # This will keep the y-axis scale values

        plt.legend()


        plt.grid(True)

        plt.savefig('Lesion_information_in_First_Fold_of_Coronal_Plane.png', dpi=400)






ranges_in_axial_3D_Subject = [
    (0, 200, 'r'), 
    (200, 500, 'indigo'), 
    (500, 1000, 'gold'), 
    (1000, 2000, 'b'), 
    (2000, 3000, 'chocolate'), 
    (3000, float('inf'), 'black'), 
]





ranges_in_Sagittal_3D_Subject = [
    (0, 500, 'r'), 
    (500, 1000, 'indigo'), 
    (1000, 2000, 'gold'), 
    (2000, 3000, 'b'), 
    (3000, 5000, 'chocolate'), 
    (5000, float('inf'), 'black'), 
    
]




ranges_in_Coronal_3D_Subject = [
    (0, 200, 'r'), 
    (200, 500, 'indigo'), 
    (500, 1000, 'gold'), 
    (1000, 2000, 'b'), 
    (2000, 3000, 'chocolate'), 
    (3000, float('inf'), 'black'), 
]









axial_csv='./K_Fold_Lesion/Axial_K_Fold_CSV/Axial_Lesion_fold_03.csv'



plot=plot_lesion_information()

plot.lesion_information_in_one_fold_axial(axial_csv,ranges_in_axial_3D_Subject)







sagittal_csv='./K_Fold_Lesion/Sagittal_K_Fold_CSV/Sagittal_Lesion_fold_01.csv'



plot=plot_lesion_information()

plot.lesion_information_in_one_fold_sagittal(sagittal_csv,ranges_in_Sagittal_3D_Subject)





coronal_csv='./K_Fold_Lesion/Coronal_K_Fold_CSV/Coronal_Lesion_fold_03.csv'



plot=plot_lesion_information()

plot.lesion_information_in_one_fold_coronal(coronal_csv,ranges_in_Coronal_3D_Subject)


