# Data Preparation 

In previous section we categorized data based on the distribution of data in batches for the reason that we dissuced in previous section.
in This section we developed the code for read the data based on that categorized batches for training model and generate a numpy arry for all
data to load the data for training and evaluation model easily .
Also, we should mention that all data after loading normilzed between 0 to 255 as float and put it in numpy arry.


# Loading data with batch distribution 


The approach is simple: first, read the CSV file, and then read the data based on their batch number. 
Place all the data with the same batch number near each other in a numpy array.
 The numpy array has a shape of (maximum_batch, batch_size, image.shape[0], image.shape[1], image.shape[2], 1).




```
python3 load_data_with_batch_distribution.py \
	-tcsv ../Distribution_Batch/Coronal_Batches_CSV/fold_2/Train_lesion_normal_batches.csv \
	-vcsv ../Distribution_Batch/Coronal_Batches_CSV/fold_2/Validation_lesion_normal_batches.csv \	
	-dd ../Moving_Converting_Data/coronal_2D \
	-p c \	
	-fold 2 \	
	-sd ./Coronal
	
```
**-tcsv, --train_csv_directory:** The path to the Train CSV file that was generated before (includes normal and lesion data).

**-vcsv, --validation_csv_directory:** The path to the Validation CSV file.

**-dd, --data:** Data directory.

**-p, --plane:** The plane (Axial, Sagittal, or Coronal).

**-fold:** Just for creating the folder.

**-sd, --save_directory:** The directory for saving the result.

The output consists of 4 numpy arrays in .npy format: 2 for MRI data (training and validation) and their corresponding masks.

** Due to the size of data we did not includes the data here**





# Loading data without batch distribution 

same as the previous part, you just need run this code for each plane and folds



```
python3 load_data_with_out_batch_distribution.py \
	-tcsv ../Distribution_Batch/Coronal_Batches_CSV/fold_2/Train_lesion_normal_batches.csv \
	-vcsv ../Distribution_Batch/Coronal_Batches_CSV/fold_2/Validation_lesion_normal_batches.csv \	
	-dd ../Moving_Converting_Data/coronal_2D \
	-p c \	
	-fold 2 \	
	-sd ./without_batch/Coronal
	
```