# Moving_Converting_Data

In this section, we developed the code to use the CSV files generated in the previous section to move 80% and 20% of the data to their respective folders. Additionally, we converted the 80% of the data into 2D slices for each plane: axial, sagittal, and coronal.

In the first step, we move the 3D data based on the 80% and 20% splitting from the previous section. The folders are:

- For 80% : Train_3D_Data

- For 20% : Fusion_evaluation_3D_data

The following code achieves this:





In next step, we converted the Train_3D_Data to 2D data. The folders are:

- Axial_2D

- Sagittal_2D

- Coronal_2D 

The following code achieves this:



> [!IMPORTANT]
> Due to licensing restrictions and the size of the data, we did not include the data in this repository.
> To regenerate the data, place the data in ./Dataset/ATLAS_2. The rest of the folders will be created by running the code.
