#Model_Comparision

In this part, we reimplement other state-of-the-art methods in lesion stroke segmentation,
 specifically for brain lesion strokes in T1 images, for comparison.

We use their code as a model and train the models on our dataset to ensure the comparison is fair.
 Additionally, we train all models for all planes and folds. In the results, we compare their performance with our model across all planes and folds


The model that used for comparison are : 

- D-Unet[^1] 
- CLCI[^2] 
- X-net[^3] 

[^1]: D-UNet: a dimension-fusion U shape network for chronic stroke lesion segmentation. [Article](https://arxiv.org/pdf/1908.05104)
[^2]: CLCI-Net: Cross-Level fusion and Context Inference Networks for Lesion Segmentation of Chronic Stroke[Article](https://arxiv.org/pdf/1907.07008)
[^3]: X-Net: Brain Stroke Lesion Segmentation Basedon Depthwise Separable Convolution and Long-range Dependencies [Article](https://arxiv.org/pdf/1907.07000)


To run the code for each model, change the directory to the model directory and execute the following code. For example, to run the code for CLCI, axial plane, and first fold, use the following command:


```

cd CLCI 

python3 train.py \
--image_size 208 240 \
--num_classes 1 \ 
 --folder_directory ../Data_Preparation/without_batch/Axial/fold_1 \
 --train_mri train_data_mri_fold_1.npy \
 --train_mask train_data_mask_fold_1.npy \
 --validation_mri validation_data_mri_fold_1.npy \
 --validation_mask validation_data_mask_fold_1.npy \
 --loss_function FocalTverskyLoss \
 --metrics_names precision recall dice_coef hausdorff_distance \
 --epochs 500 \
 --patients 10 \
 --checkpoint_path ./axial/fold_1 \
 --checkpoint_name axial_fold_1
```




For ease of use, each model directory contains a folder named run_bash_script, which includes bash scripts for all planes and folds. Running these scripts will start the model training.

For all models, the results for each plane are saved in the axial, sagittal, and coronal directories. Each of these directories contains subdirectories that include all folds.

The CSV results are available in each subfolder, but the saved weights are not included in this repository due to size limitations