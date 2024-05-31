# Data Preparation 

In previous section we categorized data based on the distribution of data in batches for the reason that we dissuced in previous section.
in This section we developed the code for read the data based on that categorized batches for training model and generate a numpy arry for all
data to load the data for training and evaluation model easily .
Also, we should mention that all data after loading normilzed between 0 to 255 as float and put it in numpy arry.


# Approch 


The approach is simple: first, read the CSV file, and then read the data based on their batch number. 
Place all the data with the same batch number near each other in a numpy array.
 The numpy array has a shape of (maximum_batch, batch_size, image.shape[0], image.shape[1], image.shape[2], 1).

Becasue of the
