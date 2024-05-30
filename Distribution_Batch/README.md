# Distribution-Batch

In the last section, we generated CSV files for K-fold cross-validation. To train the model, we train it on K-1 folds and test it on one fold. This one fold serves as the test and validation data for the model. (We cannot define validation in the K-1 fold training data because doing so would result in overlapping between training and validation sets.)

To train the model, we define the batch size, and the training data is split into batches of that size. After each batch, the parameters of the training model are updated. The common way to split data into batches is to shuffle the data first, then split and feed it to the model.

In this part, we do not shuffle the data for batch splitting; instead, we design our own batches. The approach is to split the data based on the lesion distribution of the K-1 fold of training. This means that based on the lesion distribution of the training data, we design the batches to ensure that each batch has the same distribution of input data.


# Why Distribution Batch? 
 

In our segmentation model design, our primary aim is to train the model on batches that mirror the overall distribution of lesion masks in the entire dataset.
 Instead of enforcing a uniform distribution within each batch, our strategy focuses on ensuring that the distribution of lesion masks in each batch closely resembles
 that of the entire dataset.


The specific objective is to maintain consistency in the distribution of lesion masks across batches during training.
 Each batch processed by the model contains a proportional representation of different types of lesions, preserving the overall distribution observed in the entire dataset.

The rationale behind this approach are :



1. **Preserving Data Characteristics:**
 
 By aligning the distribution of lesion masks in each batch with that of the entire dataset, we aim to preserve the
 inherent characteristics and diversity of lesion instances present in the data.
 This approach ensures that the model receives a representative sample of lesion types in each batch, facilitating comprehensive learning and generalization.

2. **Improved Optimization:**

Training the model on batches with lesion mask distributions aligned with the entire dataset enhances optimization.
By providing batches that reflect the diversity of lesion instances present in the data, 
the model's optimizer can more effectively navigate the parameter space, leading to smoother convergence and improved performance.

3. **Optimizing Batch Normalization Performance:**

Batch normalization layers rely on consistent batch statistics to normalize activations within the network. 
By maintaining a distribution of lesion masks in each batch that mirrors the overall dataset distribution,
 we enable batch normalization to accurately estimate batch statistics and effectively regulate the flow of information through the model.


# Approch


1. Read all the CSV files of folds.
2. Based on the defined bins, categorize the folds according to the bins.
3. Define the batch size.
4. Based on the number of batches and the distribution of data in the fold CSV files, split the data into the number of batches that we can generate.




```
python3 Distribution_batch.py \
    -bs 32 \
    -lcsv ../Cross_Validation/K_Fold_Lesion/Axial_K_Fold_CSV \
    -ncsv ../Cross_Validation/K_Fold_Normal/Axial_K_Fold_CSV \
    -col sum_voxel \
    -vf 1 \
    -bins [0,200,500,1000,2000,3000] \
    -aug False \
    -auglist None \
    -rdb False \
    -rsb False \
    -sd ./Axial_Batches_CSV	

```

**-bs,--batch_size:** Specifies the size of each batch. Example: 32 (The half are lesion slices with the data distribution and the rest are the normal slices)

**-lcsv, --lesion_csv_dir:** Path to the directory containing CSV files for lesion slices. The files should follow the naming convention ending in _fold_01.csv.

**-ncsv, --normal_csv_dir:** Path to the directory containing CSV files for normal slices. The files should follow the naming convention ending in _fold_01.csv.

**-col, --column:** The column in the CSV files that the operations will be based on.

**-vf, --val_fold:** Indicates which fold should be used as the validation fold. Example: 1. ( The rest folds will use for training)

**-bins, --bins:** A list of bins for data distribution. Example: [0,200,500,1000,2000,3000]. After the last one we add the inf to this list. Example :[0,200,500,1000,2000,3000,inf].

**-aug, --augment:** Boolean flag indicating whether to enable data augmentation to balance distribution. Accepts True or False. If you enable the augmentation we apply the various 
augmentation method that you define in auglist on your data to balanced the data in each bins and after that apply the batch distribution.

**-auglist, --augment_list:**  the list of augmentation methods to apply. Use None if no augmentation is needed. Otherwise you can apply any augmentation method to the augmentation method 
and the code create a column "Augmentation" and put the augmentation method for the data. The choose of the method is completely randomly from the list that you provide.
Example :['v-flip','rotation','h-flip']

**-rdb, --repeat_diff_batches:** Boolean flag indicating whether to allow repeated data across different batches. Accepts True or False.

**-rsb, --repeat_same_batch:** Boolean flag indicating whether to allow repeated data within a single batch. Accepts True or False.

**-sd, --save_dir:** Path to the directory where the output CSV files will be saved. You have 6 output :

	1- Train_lesion_batches : Desing the batches for lesion slices for the training data 

	2- Train_normal_batches : Desing the batches for normal slices for the normal data 

 	3- Train_lesion_normal_batches : Combination of lesion and normal slices (1 and 2)
	
	- The other 3 CSV file are for the validation data 


> [!CAUTION]
> We also applied the distribution batch for validation data because we aim to evaluate the model during the training progress. The evaluation should mimic the same conditions as the training data. However, for the final evaluation of the model, we predict each sample separately.

> [!NOTE]
> Based on the selected validation fold, a folder will be created with the name "fold_{validation_fold}" in the output directory after the code execution.


