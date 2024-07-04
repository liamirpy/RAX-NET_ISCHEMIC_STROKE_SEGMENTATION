## This is liamirpy

## In this part we want to predict for evalutation data and predict the result and save the result
## We do it for each plane for all folds




## Axial 

import numpy as np

import pandas as pd

import os 
import time
import nibabel as nib
from keras.models import load_model
from statistic import Statistics
import csv 


stats = Statistics()





from keras.layers import Activation, Reshape, Lambda, dot, add, Input, BatchNormalization, ReLU, DepthwiseConv2D, Concatenate
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2


from keras import *
from keras.layers import *
import tensorflow as tf


kernel_regularizer = regularizers.l2(1e-5)
bias_regularizer = regularizers.l2(1e-5)
kernel_regularizer = None
bias_regularizer = None

def conv_lstm(input1, input2, channel=256):
    lstm_input1 = Reshape((1, input1.shape.as_list()[1], input1.shape.as_list()[2], input1.shape.as_list()[3]))(input1)
    lstm_input2 = Reshape((1, input2.shape.as_list()[1], input2.shape.as_list()[2], input1.shape.as_list()[3]))(input2)

    lstm_input = custom_concat(axis=1)([lstm_input1, lstm_input2])
    x = ConvLSTM2D(channel, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(lstm_input)
    return x

def conv_2(inputs, filter_num, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer):
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(conv_)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    return conv_

def conv_2_init(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_2(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)

def conv_2_init_regularization(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_2(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(5e-4))

def conv_1(inputs, filter_num, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer):
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    return conv_

def conv_1_init(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_1(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)

def conv_1_init_regularization(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_1(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(5e-4))

def dilate_conv(inputs, filter_num, dilation_rate):
    conv_ = Conv2D(filter_num, kernel_size=(3,3), dilation_rate=dilation_rate, padding='same', kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    return conv_

class custom_concat(Layer):

    def __init__(self, axis=-1, **kwargs):
        super(custom_concat, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.built = True
        super(custom_concat, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        self.res = tf.concat(x, self.axis)

        return self.res

    def compute_output_shape(self, input_shape):
        # return (input_shape[0][0],)+(len(input_shape),)+input_shape[0][2:]
        # print((input_shape[0][0],)+(len(input_shape),)+input_shape[0][2:])
        input_shapes = input_shape
        output_shape = list(input_shapes[0])

        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]

        return tuple(output_shape)


class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.upsampling = upsampling

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return tf.compat.v1.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                   int(inputs.shape[2] * self.upsampling[1])))



def concat_pool(conv, pool, filter_num, strides=(2, 2)):
    conv_downsample = Conv2D(filter_num, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(conv)
    conv_downsample = BatchNormalization()(conv_downsample)
    conv_downsample = Activation('relu')(conv_downsample)
    concat_pool_ = Concatenate()([conv_downsample, pool])
    return concat_pool_


















axial_fold_csv_directory='../../Distribution_Batch/Axial_Batches_CSV'
sagittal_fold_csv_directory='../../Distribution_Batch/Sagittal_Batches_CSV'
coronal_fold_csv_directory='../../Distribution_Batch/Coronal_Batches_CSV'



axial_data_directory='../../Moving_Converting_Data/axial_2D/'
sagittal_data_directory='../../Moving_Converting_Data/sagittal_2D/'
coronal_data_directory='../../Moving_Converting_Data/coronal_2D/'




axial_saved_model_folds='../../Model_Comparison/CLCI/axial'
sagittal_saved_model_folds='../../Model_Comparison/CLCI/sagittal'
coronal_saved_model_folds='../../Model_Comparison/CLCI/coronal'




number_of_folds=5


# for plane in [axial_fold_csv_directory,sagittal_fold_csv_directory,coronal_fold_csv_directory]:

for plane in [coronal_fold_csv_directory]:



    for fold in range(1,number_of_folds+1):

        results=[]

        validation_data_csv= pd.read_csv(f'{plane}/fold_{fold}/Validation_lesion_normal_batches.csv')


        if plane == axial_fold_csv_directory:


            model=load_model(f'{axial_saved_model_folds}/fold_{fold}/axial_fold_{fold}.h5',custom_objects={
                 'DiceCoefLoss':stats.DiceCoefLoss,
                 'BilinearUpsampling':BilinearUpsampling,
                 'custom_concat':custom_concat,
                'dice_coef': stats.dice_coef,
                'precision':stats.precision,
                'recall':stats.recall,
                'hausdorff_distance':stats.hausdorff_distance})
            

            single_slice_mri=train_data_mri=np.zeros((208,240,1),dtype=np.float32)

            single_slice_true_label=train_data_mri=np.zeros((208,240,1),dtype=np.float32)

            for index, row in validation_data_csv.iterrows():


                subject = row['subject_name']

                slice_number = row['slice']


                mri_nii=nib.load(f'{axial_data_directory}/{subject}/{subject}_slice_{slice_number}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()


                single_slice_mri[6:6+197,3:3+233,0]=mri_numpy

                expand_mri=np.expand_dims(single_slice_mri,axis=0)



                t0=time.time()
                prediction = model.predict(expand_mri)
                t1=time.time()


                binary_prediction = (prediction > 0.5).astype(np.float32)





                mask_nii=nib.load(f'{axial_data_directory}/{subject}/{subject}_slice_{slice_number}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()
                mask_numpy=np.where(mask_numpy > 0, 1, 0)

                single_slice_true_label[6:6+197,3:3+233,0]=mask_numpy
                expand_true_label=np.expand_dims(single_slice_true_label,axis=0)



                dice_value=stats.dice_coef(expand_true_label,binary_prediction)
                precision_value=stats.precision(expand_true_label,binary_prediction)
                recall_value=stats.recall(expand_true_label,binary_prediction)
                hddd_value=stats.hausdorff_distance(expand_true_label,binary_prediction)


                result={'subject':subject,'slice':slice_number,'dice':dice_value.numpy(),'presision':precision_value.numpy(),'recall':recall_value.numpy(),'hausdorff_distance':hddd_value.numpy(),'time':t1 - t0}
                results.append(result)


            
            lesion_coronal_distribution = f'./Axial/result_of_axial_fold_{fold}.csv'

            with open(lesion_coronal_distribution, 'w', newline='') as csvfile:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Writing header
                writer.writeheader()

                # Writing data
                writer.writerows(results)

            print(f'Data has been written to {lesion_coronal_distribution}.')

            








            




        if plane == sagittal_fold_csv_directory:


            model=load_model(f'{sagittal_saved_model_folds}/fold_{fold}/sagittal_fold_{fold}.h5'
                             ,custom_objects={'DiceCoefLoss':stats.DiceCoefLoss,
                                                'BilinearUpsampling':BilinearUpsampling,
                                                'custom_concat':custom_concat,                          
                                              'dice_coef': stats.dice_coef,'precision':stats.precision,'recall':stats.recall,'hausdorff_distance':stats.hausdorff_distance})
            

            single_slice_mri=train_data_mri=np.zeros((240,208,1),dtype=np.float32)
            single_slice_true_label=train_data_mri=np.zeros((240,208,1),dtype=np.float32)

            for index, row in validation_data_csv.iterrows():


                subject = row['subject_name']

                slice_number = row['slice']


                mri_nii=nib.load(f'{sagittal_data_directory}/{subject}/{subject}_slice_{slice_number}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()



                single_slice_mri[3:3+233,9:9+189,0]=mri_numpy

                expand_mri=np.expand_dims(single_slice_mri,axis=0)

                t0=time.time()
                prediction = model.predict(expand_mri)
                t1=time.time()

                binary_prediction = (prediction > 0.5).astype(np.float32)



                mask_nii=nib.load(f'{sagittal_data_directory}/{subject}/{subject}_slice_{slice_number}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()
                mask_numpy=np.where(mask_numpy > 0, 1, 0)



                single_slice_true_label[3:3+233,9:9+189,0]=mask_numpy
                expand_true_label=np.expand_dims(single_slice_true_label,axis=0)





                dice_value=stats.dice_coef(expand_true_label,binary_prediction)
                precision_value=stats.precision(expand_true_label,binary_prediction)
                recall_value=stats.recall(expand_true_label,binary_prediction)
                hddd_value=stats.hausdorff_distance(expand_true_label,binary_prediction)


                result={'subject':subject,'slice':slice_number,'dice':dice_value.numpy(),'presision':precision_value.numpy(),'recall':recall_value.numpy(),'hausdorff_distance':hddd_value.numpy(),'time':t1 - t0}
                results.append(result)


            lesion_coronal_distribution = f'./Sagittal/result_of_sagittal_fold_{fold}.csv'

            with open(lesion_coronal_distribution, 'w', newline='') as csvfile:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Writing header
                writer.writeheader()

                # Writing data
                writer.writerows(results)

            print(f'Data has been written to {lesion_coronal_distribution}.')

            

            
            



        if plane == coronal_fold_csv_directory:


            model=load_model(f'{coronal_saved_model_folds}/fold_{fold}/coronal_fold_{fold}.h5',
                             custom_objects={  
                                             'BilinearUpsampling':BilinearUpsampling,
                                            'custom_concat':custom_concat,                                                                
                                            'DiceCoefLoss':stats.DiceCoefLoss,'dice_coef': stats.dice_coef,'precision':stats.precision,'recall':stats.recall,'hausdorff_distance':stats.hausdorff_distance})
            


            single_slice_mri=train_data_mri=np.zeros((208,208,1),dtype=np.float32)
            single_slice_true_label=train_data_mri=np.zeros((208,208,1),dtype=np.float32)

            for index, row in validation_data_csv.iterrows():


                subject = row['subject_name']

                slice_number = row['slice']


                mri_nii=nib.load(f'{coronal_data_directory}/{subject}/{subject}_slice_{slice_number}_T1.nii.gz')

                mri_numpy=mri_nii.get_fdata()


                single_slice_mri[5:5+197,9:9+189,0]=mri_numpy

                expand_mri=np.expand_dims(single_slice_mri,axis=0)

                t0=time.time()
                prediction = model.predict(expand_mri)
                t1=time.time()
                binary_prediction = (prediction > 0.5).astype(np.float32)


                mask_nii=nib.load(f'{coronal_data_directory}/{subject}/{subject}_slice_{slice_number}_mask.nii.gz')

                mask_numpy=mask_nii.get_fdata()
                mask_numpy=np.where(mask_numpy > 0, 1, 0)

                single_slice_true_label[5:5+197,9:9+189,0]=mask_numpy
                expand_true_label=np.expand_dims(single_slice_true_label,axis=0)



                dice_value=stats.dice_coef(expand_true_label,binary_prediction)
                precision_value=stats.precision(expand_true_label,binary_prediction)
                recall_value=stats.recall(expand_true_label,binary_prediction)
                hddd_value=stats.hausdorff_distance(expand_true_label,binary_prediction)


                result={'subject':subject,'slice':slice_number,'dice':dice_value.numpy(),'presision':precision_value.numpy(),'recall':recall_value.numpy(),'hausdorff_distance':hddd_value.numpy(),'time':t1 - t0}
                results.append(result)



            lesion_coronal_distribution = f'./Coronal/result_of_coronal_fold_{fold}.csv'

            with open(lesion_coronal_distribution, 'w', newline='') as csvfile:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Writing header
                writer.writeheader()

                # Writing data
                writer.writerows(results)

            print(f'Data has been written to {lesion_coronal_distribution}.')

            





