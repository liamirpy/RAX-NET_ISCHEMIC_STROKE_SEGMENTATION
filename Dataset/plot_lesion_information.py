import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




class plot_lesion_information:

    def lesion_information_in_3D(self,csv_file,ranges):

        csv_file = csv_file
        df = pd.read_csv(csv_file)

        # Extract the desired column
        column_name = 'sum_voxel'  

        data = df[column_name] / 1000


        range_counts = []
        for start, end, color in ranges:
            count = ((data >= start) & (data < end)).sum()
            range_counts.append((start, end, color, count))
    





        plt.figure(figsize=(10, 10))

        # Plot the violin plot
        x_value = 0
        sns.violinplot(x=[x_value] * len(data), y=data, color='darkcyan', fill=True,inner_kws=dict(box_width=5, whis_width=1, color=".1"))

        # Plot the scatter plot with the same x-coordinate
        for start, end, color,count in range_counts:
            mask = (data >= start) & (data < end)
            x_values = np.random.uniform(x_value - 0.35, x_value + 0.35, sum(mask))  # Add slight jitter
            plt.scatter(x_values, data[mask],color=color, label=f'{start} to {end} (10^3) Voxels (N={count})',s=20,linewidths=0.5)

        # Set the labels for the axes
        plt.xlabel('Lesion Size Distribution in All 655 3D MRI Subjects',fontsize=20)
        plt.ylabel('volume (10 ^3) Voxels', fontsize=20)

        # Display y-axis scale values
        plt.yticks()  # This will keep the y-axis scale values

        # Add legend
        plt.legend()

        # plt.show()

        plt.grid(True)

        plt.savefig('Lesion_information_in_665_3D_Subjects.png', dpi=400)



    

    def lesion_information_in_axial(self,csv_file,ranges):

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
        plt.xlabel('Lesion Size Distribution in Axial Plane of All 655 3D MRI Subjects',fontsize=20)
        plt.ylabel('volume Pixels', fontsize=20)

        # Display y-axis scale values
        plt.yticks()  # This will keep the y-axis scale values

        # Add legend
        plt.legend()

        # plt.show()

        plt.grid(True)

        plt.savefig('Lesion_information_in_Axial_Plane_665_3D_Subjects.png', dpi=400)



    def lesion_information_in_sagittal(self,csv_file,ranges):

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
        plt.xlabel('Lesion Size Distribution in Sagittal Plane of All 655 3D MRI Subjects',fontsize=20)
        plt.ylabel('volume Pixels', fontsize=20)

        # Display y-axis scale values
        plt.yticks()  # This will keep the y-axis scale values

        # Add legend
        plt.legend()

        # plt.show()

        plt.grid(True)



        plt.savefig('Lesion_information_in_Sagittal_Plane_665_3D_Subjects.png', dpi=400)


    def lesion_information_in_coronal(self,csv_file,ranges):

 
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
        plt.xlabel('Lesion Size Distribution in Coronal Plane of All 655 3D MRI Subjects',fontsize=20)
        plt.ylabel('volume Pixels', fontsize=20)

        plt.yticks()  # This will keep the y-axis scale values

        plt.legend()


        plt.grid(True)

        plt.savefig('Lesion_information_in_Coronal_Plane_665_3D_Subjects.png', dpi=400)






ranges_in_3D_Subject = [
    (0, 5, 'r'), 
    (5, 30, 'indigo'), 
    (30, 60, 'gold'), 
    (60, 100, 'b'), 
    (100, 200, 'chocolate'), 
    (200, float('inf'), 'black'), 
]



ranges_in_axial_3D_Subject = [
    (0, 200, 'r'), 
    (200, 500, 'indigo'), 
    (500, 1000, 'gold'), 
    (1000, 2000, 'b'), 
    (2000, 3000, 'chocolate'), 
    (3000, float('inf'), 'black'), 
]





ranges_in_Sagittal_3D_Subject = [
    (0, 1000, 'r'), 
    (1000, 2000, 'indigo'), 
    (2000, 3000, 'gold'), 
    (3000, 5000, 'b'), 
    # (5000, 7000, 'chocolate'), 
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



lesion_3D_subject_csv='./CSV/ATLAS_Lesion_information_for_3D_MRI_Subject.csv'


plot=plot_lesion_information()

plot.lesion_information_in_3D(lesion_3D_subject_csv,ranges_in_3D_Subject)









axial_csv='./CSV/ATLAS_Axial_Lesion_information.csv'



plot=plot_lesion_information()

plot.lesion_information_in_axial(axial_csv,ranges_in_axial_3D_Subject)







sagittal_csv='./CSV/ATLAS_Sagittal_Lesion_information.csv'



plot=plot_lesion_information()

plot.lesion_information_in_sagittal(sagittal_csv,ranges_in_Sagittal_3D_Subject)





coronal_csv='./CSV/ATLAS_Coronal_Lesion_information.csv'



plot=plot_lesion_information()

plot.lesion_information_in_coronal(coronal_csv,ranges_in_Coronal_3D_Subject)


