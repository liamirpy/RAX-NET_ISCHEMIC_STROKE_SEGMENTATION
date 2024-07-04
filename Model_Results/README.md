# Model_Results

"In this section, we present the results of the proposed model for axial, sagittal, and coronal views. We used the trained model to predict the validation data and evaluated the results using various metrics.

# Approach

We first load the trained model and the validation data. We predict each sample from the validation data and then evaluate the metrics based on the true values or masks.

The metrics used are Dice coefficient, recall, precision, and Hausdorff distance.

This approach is applied to all planes: axial, sagittal, and coronal. Additionally, this evaluation is performed for all folds for each plane.

# Axial

The Result of Axial: 



|      Folds    |      Dice      |       Recall    |   Precision    |             Average Score             |
| ------------- | -------------- | --------------- | ---------------|---------------------------------------|
|     Fold_1    |0.6878 +- 0.3762|0.8055 +- 0.3291 |0.7713 +- 0.3324|(0.6878 + 0.8055 + 0.7713) / 3 = 0.7549|
|     Fold_2    |0.7047 +- 0.3792|0.7674 +- 0.3606 |0.8391 +- 0.2832|(0.7047 + 0.7674 + 0.8391) / 3 = 0.7704|
|     Fold_3    |0.7282 +- 0.3558|0.7958 +- 0.3355 |0.8269 +- 0.2808|(0.7282 + 0.7958 + 0.8269) / 3 = 0.7836|
|     Fold_4    |0.7209 +- 0.3684|0.7753 +- 0.3533 |0.8515 +- 0.2678|(0.7209 + 0.7753 + 0.8515) / 3 = 0.7826|
|     Fold_5    |0.7104 +- 0.3776|0.7567 +- 0.3675 |0.8638 +- 0.2559|(0.7104 + 0.7567 + 0.8638) / 3 = 0.7770|
|   All_Folds   |0.7104 +- 0.0127|0.7801 +- 0.0164 |0.8305 +- 0.0292|                                       |


Best Fold for Axial Plane: Fold 3 (Highest Average Score: 0.7836)




# Coronal

The Result of Coronal: 



|      Folds    |      Dice      |       Recall    |   Precision    |             Average Score             |
| ------------- | -------------- | --------------- | ---------------|---------------------------------------|
|     Fold_1    |0.6579 +- 0.3990|0.7605 +- 0.3735 |0.7844 +- 0.3301|(0.6579 + 0.7605 + 0.7844) / 3 = 0.7343|
|     Fold_2    |0.6918 +- 0.3802|0.7665 +- 0.3569 |0.8161 +- 0.3033|(0.6918 + 0.7665 + 0.8161) / 3 = 0.7581|
|     Fold_3    |0.6932 +- 0.3843|0.7451 +- 0.3769 |0.8512 +- 0.2704|(0.6932 + 0.7451 + 0.8512) / 3 = 0.7632|
|     Fold_4    |0.6994 +- 0.3810|0.7620 +- 0.3627 |0.8376 +- 0.2874|(0.6994 + 0.7620 + 0.8376) / 3 = 0.7663|
|     Fold_5    |0.6953 +- 0.3861|0.7402 +- 0.3801 |0.8701 +- 0.2519|(0.6953 + 0.7402 + 0.8701) / 3 = 0.7685|
|   All_Folds   |0.6875 +- 0.0137|0.7549 +- 0.0093 |0.8319 +- 0.0269|                                       |
           

Best Fold for Coronal Plane: Fold 5 (Highest Average Score: 0.7685)



# Sagittal

The Result of Sagittal: 



|      Folds    |      Dice      |       Recall    |   Precision    |             Average Score             |
| ------------- | -------------- | --------------- | ---------------|---------------------------------------|
|     Fold_1    |0.6260 +- 0.3996|0.6931 +- 0.3889 |0.7838 +- 0.3313|(0.6260 + 0.6931 + 0.7838) / 3 = 0.7010|
|     Fold_2    |0.6058 +- 0.4159|0.6685 +- 0.4100 |0.8236 +- 0.3064|(0.6058 + 0.6685 + 0.8236) / 3 = 0.6993|
|     Fold_3    |0.6527 +- 0.3947|0.7062 +- 0.3879 |0.8305 +- 0.2841|(0.6527 + 0.7062 + 0.8305) / 3 = 0.7298|
|     Fold_4    |0.6178 +- 0.3992|0.7302 +- 0.3781 |0.7492 +- 0.3447|(0.6178 + 0.7302 + 0.7492) / 3 = 0.6991|
|     Fold_5    |0.6063 +- 0.3989|0.7297 +- 0.3738 |0.7346 +- 0.3537|(0.6063 + 0.7297 + 0.7346) / 3 = 0.6902|
|   All_Folds   |0.6217 +- 0.0157|0.7055 +- 0.0212 |0.7843 +- 0.0350|                                       |


Best Fold for Sagittal Plane: Fold 3 (Highest Average Score: 0.7298)

