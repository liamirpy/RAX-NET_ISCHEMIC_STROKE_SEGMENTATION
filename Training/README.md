# Training 


In this part, we aim to train the proposed model for axial, sagittal, and coronal planes. 
For each plane, the model will be trained, and the results will be saved in a CSV file in the directory. 
The weights will also be saved in that directory for future use.


for training the model you need to run this code:

```
cd ..

cd model 

python3 train.py \
--image_size 208 240 \
--num_classes 1 \ 
 --folder_directory ../Data_Preparation/Axial/fold_1 \
 --train_mri train_data_mri_fold_1.npy \
 --train_mask train_data_mask_fold_1.npy \
 --validation_mri validation_data_mri_fold_1.npy \
 --validation_mask validation_data_mask_fold_1.npy \
 --loss_function FocalTverskyLoss \
 --metrics_names precision recall dice_coef hausdorff_distance \
 --epochs 500 \
 --patients 50 \
 --checkpoint_path ../Training/axial/fold_1 \
 --checkpoint_name axial_fold_1
```


A complete description of this code is discussed in the [model](../Model) directory.

For ease of use, we have added bash scripts for all folds and planes in the bash_run_script directory, which consists of bash files for all folds and planes. If you navigate to that directory and run the .sh file in your terminal, the model will start training.

The results are organized as follows: for each plane, there is a directory, and within each plane directory, there are five subdirectories, one for each fold. The results are saved in each subdirectory. However, due to size constraints, the saved weights are not included in this repository.