# Distribution-Batch

In the last section, we generated CSV files for K-fold cross-validation. To train the model, we train it on K-1 folds and test it on one fold. This one fold serves as the test and validation data for the model. (We cannot define validation in the K-1 fold training data because doing so would result in overlapping between training and validation sets.)

To train the model, we define the batch size, and the training data is split into batches of that size. After each batch, the parameters of the training model are updated. The common way to split data into batches is to shuffle the data first, then split and feed it to the model.

In this part, we do not shuffle the data for batch splitting; instead, we design our own batches. The approach is to split the data based on the lesion distribution of the K-1 fold of training. This means that based on the lesion distribution of the training data, we design the batches to ensure that each batch has the same distribution of input data.

# Approch


