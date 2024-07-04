# Loss Comparison

In the previous section, we introduced a new loss function named FocalTverskyLoss,
 which combines the focal loss function and the Tversky loss function. In this section,
 we apply and train the model on a single fold of axial data using four different loss functions:
 Dice coefficient, focal loss, Tversky loss, and our new FocalTverskyLoss. We then compare the results on
 the validation data using metrics such as recall, precision, and the Dice coefficient. Our results demonstrate that the FocalTverskyLoss generally performs better overall.





To generate the results, you need to change the directory to /Model and run the following code, adjusting the loss_function parameter based on the specific loss function you want to use.




```
python3 train.py \
	--image_size 208 240 \
	--num_classes 1 \
	--folder_directory ../Data_Preparation/Axial/fold_3 \
	--train_mri train_data_mri_fold_3.npy \ 
	--train_mask train_data_mask_fold_3.npy \
	--validation_mri validation_data_mri_fold_3.npy \
	--validation_mask validation_data_mask_fold_3.npy \
	--loss_function DiceCoefLoss \
	--metrics_names precision recall  hausdorff_distance \
	--epochs 150 \	
	--patients 20 \
	--checkpoint_path ./ \	
	--checkpoint_name lo


```


We performed this process, and the results of the training progress are recorded in the epochs results, which include the training and validation metrics for each epoch. After training, the model's weights were saved with the corresponding loss function's name.

For comparison, we load the model and predict the validation data, saving the results in a CSV file by running the provided code. Note that you need to place the saved model weights in the directory, but due to their size, we did not include the model weights here.

```
python3 evaluatation_validation.py

```

However, the results are saved in the validation_results directory for each loss function.

The final comparison is presented in the table below.





| Loss_Function |      Dice      |    Precision    |     Recall     |        F1      |
| ------------- | -------------- | --------------- | ---------------|----------------|
|Dice_Coef Loss |0.7859 +- 0.0444|0.8430 +- 0.0573 |0.7406 +- 0.0648|0.7889 +- 0.0485|
| Focal Loss    |0.7326 +- 0.0419|0.7302 +- 0.0568 |0.7427 +- 0.0736|0.7364 +- 0.0552|
| Trevesky Loss |0.7764 +- 0.0430|0.7559 +- 0.0576 |0.8037 +- 0.0655|0.7791 +- 0.0513|
|FocalTrevesky  |0.7904 +- 0.0404|0.7963 +- 0.0549 |0.7894 +- 0.0627|0.7928 +- 0.0491|




