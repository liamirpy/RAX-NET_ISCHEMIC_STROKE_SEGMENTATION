
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from tensorflow.keras.layers import Flatten,Dense,Conv2D
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
# import cv2
import os




import tensorflow as tf
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)


csv_name='log_VURX_attention_fl_90_30_70_lr_3.csv'




from tensorflow import keras

from tensorflow.keras import layers

img_size=(208,240)
num_classes=1
num_classes=1
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from tensorflow.keras.layers import Flatten,Dense,Conv2D
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
# import cv2
import os










csv_name='log_VURX_attention_fl_90_30_70_lr_3.csv'




from tensorflow import keras

from tensorflow.keras import layers

img_size=(208,240)
num_classes=1
num_classes=1
from tensorflow import keras


from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from tensorflow.keras.layers import Flatten,Dense,Conv2D
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
# import cv2
import os










csv_name='log_VURX_attention_fl_90_30_70_lr_3.csv'




from tensorflow import keras

from tensorflow.keras import layers

img_size=(208,240)
num_classes=1
num_classes=1
from tensorflow import keras


def VURX_attention(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    conv1=inputs

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    conv2=x



    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        if filters == 128 or filters==256:
          x = layers.Activation("relu")(x)
          x = layers.Conv2D(filters, 3, padding="same")(x)
          x = layers.BatchNormalization()(x)

          x = layers.Activation("relu")(x)
          x = layers.SeparableConv2D(filters, 3, padding="same")(x)
          x = layers.BatchNormalization()(x)

        #   x=keras.layers.Dropout(0.5)(x)





        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)



        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual


        if filters==64:
          conv3=x

        if filters==128:
          conv4=x

        previous_block_activation = x  # Set aside next residual



    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)


        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        # x=keras.layers.Dropout(0.5)(x)



        if filters == 256 or filters==128:
          x = layers.Activation("relu")(x)
          x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
          x = layers.BatchNormalization()(x)

          x = layers.Activation("relu")(x)
          x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
          x = layers.BatchNormalization()(x)




        x = layers.UpSampling2D(2)(x)

        if filters == 32:

          residual = layers.UpSampling2D(2)(previous_block_activation)
          residual = layers.Conv2D(filters, 1, padding="same")(residual)
          # residual = layers.BatchNormalization()(residual)
          # residual = layers.Activation("relu")(residual)
          # residual = layers.DepthwiseConv2D(filters, 1, padding="same")(residual)

          conv1=layers.Conv2D(filters, 1, padding="same")(conv1)
          # conv1 = layers.BatchNormalization()(conv1)
          # conv1 = layers.Activation("relu")(conv1)


          # conv1 = layers.DepthwiseConv2D(filters, 1, padding="same")(conv1)

          first_residual=layers.add([residual,conv1])
          first_residual = layers.Activation("relu")(first_residual)


          x1=layers.Conv2D(filters, 1, padding="same")(first_residual)
          x1 = layers.BatchNormalization()(x1)


          x = layers.add([x, x1]) 

        if filters == 64:

          residual = layers.UpSampling2D(2)(previous_block_activation)
          residual = layers.Conv2D(filters, 1, padding="same")(residual)
          # residual = layers.BatchNormalization()(residual)
          # residual = layers.Activation("relu")(residual)
          # residual = layers.DepthwiseConv2D(filters, 1, padding="same")(residual)

          conv2=layers.Conv2D(filters, 1, padding="same")(conv2)
          # conv2 = layers.BatchNormalization()(conv2)
          # conv2 = layers.Activation("relu")(conv2)


          # conv1 = layers.DepthwiseConv2D(filters, 1, padding="same")(conv1)

          first_residual=layers.add([residual,conv2])

          first_residual = layers.Activation("relu")(first_residual)

          x1=layers.Conv2D(filters, 1, padding="same")(first_residual)
          x1 = layers.BatchNormalization()(x1)
          # x1 = layers.Activation("relu")(x1)


          x = layers.add([x, x1]) 


        if filters == 128:
       #   # x=layers.Concatenate(axis=3)([x,conv2])
          residual = layers.UpSampling2D(2)(previous_block_activation)
          residual = layers.Conv2D(filters, 1, padding="same")(residual)
          # residual = layers.BatchNormalization()(residual)
          # residual = layers.Activation("relu")(residual)
          # residual = layers.DepthwiseConv2D(filters, 1, padding="same")(residual)

          conv3=layers.Conv2D(filters, 1, padding="same")(conv3)
          # conv3 = layers.BatchNormalization()(conv3)
          # conv3 = layers.Activation("relu")(conv3)


          # conv1 = layers.DepthwiseConv2D(filters, 1, padding="same")(conv1)

          first_residual=layers.add([residual,conv3])
          first_residual = layers.Activation("relu")(first_residual)

          x1=layers.Conv2D(filters, 1, padding="same")(first_residual)
          x1 = layers.BatchNormalization()(x1)
          # x1 = layers.Activation("relu")(x1)


          x = layers.add([x, x1]) 
        if filters == 256:

          residual = layers.UpSampling2D(2)(previous_block_activation)
          residual = layers.Conv2D(filters, 1, padding="same")(residual)
          # residual = layers.BatchNormalization()(residual)
          # residual = layers.Activation("relu")(residual)
          # residual = layers.DepthwiseConv2D(filters, 1, padding="same")(residual)

          conv4=layers.Conv2D(filters, 1, padding="same")(conv4)
          # conv4 = layers.BatchNormalization()(conv4)
          # conv4 = layers.Activation("relu")(conv4)


          # conv1 = layers.DepthwiseConv2D(filters, 1, padding="same")(conv1)

          first_residual=layers.add([residual,conv4])
          first_residual = layers.Activation("relu")(first_residual)

          x1=layers.Conv2D(filters, 1, padding="same")(first_residual)
          x1 = layers.BatchNormalization()(x1)
          # x1 = layers.Activation("relu")(x1)


          x = layers.add([x, x1]) 




        previous_block_activation = x  # Set aside next residual


    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

model = VURX_attention(img_size, num_classes)
model.summary()




def hausdorff_distance(y_true, y_pred):
    """
    Custom Hausdorff distance loss function for Keras.
    
    Parameters:
    y_true : tensor
        Ground truth values. Shape (batch_size, num_points, 2) for 2D points.
    y_pred : tensor
        Predicted values. Shape (batch_size, num_points, 2) for 2D points.
    
    Returns:
    loss : tensor
        Computed Hausdorff distance.
    """
    def compute_hausdorff(a, b):
        # Compute pairwise distances
        d_matrix = tf.norm(tf.expand_dims(a, axis=-2) - tf.expand_dims(b, axis=-3), axis=-1)
        
        # Directed distances
        d_ab = tf.reduce_max(tf.reduce_min(d_matrix, axis=-1), axis=-1)
        d_ba = tf.reduce_max(tf.reduce_min(d_matrix, axis=-2), axis=-1)
        
        # Hausdorff distance
        hausdorff = tf.maximum(d_ab, d_ba)
        
        return hausdorff
    
    # Compute batch-wise Hausdorff distances
    hd_batch = tf.map_fn(lambda x: compute_hausdorff(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
    
    # Return the mean Hausdorff distance for the batch
    return tf.reduce_mean(hd_batch)






lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100000,
    decay_rate=0.9)











csv_logger = CSVLogger(csv_name,append=True , separator=';')





def TP(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_pred_f = K.round(y_pred_f)

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    return true_positives


def FP(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_pred_f = K.round(y_pred_f)

    y_pred_f01 = K.round(K.clip(y_pred_f, 0, 1))
    tp_f01 = K.round(K.clip(y_true_f * y_pred_f, 0, 1))
    false_positives = K.sum(K.round(K.clip(y_pred_f01 - tp_f01, 0, 1)))
    return false_positives


def TN(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_pred_f = K.round(y_pred_f)

    y_pred_f = K.round(y_pred_f)

    y_pred_f01 = K.round(K.clip(y_pred_f, 0, 1))
    all_one = K.ones_like(y_pred_f01)
    y_pred_f_1 = -1 * (y_pred_f01 - all_one)
    y_true_f_1 = -1 * (y_true_f - all_one)
    true_negatives = K.sum(K.round(K.clip(y_true_f_1 + y_pred_f_1, 0, 1)))
    return true_negatives


def FN(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # y_pred_f = K.round(y_pred_f)

    # y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    tp_f01 = K.round(K.clip(y_true_f * y_pred_f, 0, 1))
    false_negatives = K.sum(K.round(K.clip(y_true_f - tp_f01, 0, 1)))
    return false_negatives


def rec(y_true, y_pred):
    tp = TP(y_true, y_pred)
    fn = FN(y_true, y_pred)
    return( tp +1 )/ (tp +1 + fn) 


def pre(y_true, y_pred):
    tp = TP(y_true, y_pred)
    fp = FP(y_true, y_pred)
    return (tp + 1) / (tp +1 + fp) 







def dice_loss(y_true, y_pred):

    # y_pred = K.cast(y_pred > 0.5, 'float32')  # Ensure y_true is float32

    y_true = K.cast(y_true , 'float32')  # Ensure y_true is float32


    # y_pred = y_pred > 0.5 

    # y_pred = K.cast(y_pred , 'float32')  # Ensure y_true is float32

    y_truef = K.flatten(y_true)

    y_predf = K.flatten(y_pred)

    # y_predf = K.round(y_predf)



    floss=keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=True,
    alpha=0.80,
    gamma=2,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    
    )
    fl= floss(y_true, y_pred)


    # y_predf=K.round(y_predf)

    intersection = K.sum(y_truef * y_predf)



    tp = TP(y_true, y_pred)
    fp = FP(y_pred, y_pred) 
    fn = FN(y_true, y_pred) 

    # Calculate Tversky coefficient
    tversky = (tp +1) / (tp + 0.3 * fp + 0.7 * fn + 1)

    # Calculate Tversky loss
    # tversky_loss = 1 - tversky



    return 1 + (((2. * intersection + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))  ) * 0 - tversky + fl

   
    # return 1 + (((2. * intersection + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))  ) * 0 - tversky + fl

    # return 1 - (((2. * intersection + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))  ) 

    # return fl - K.log(((2. * intersection + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth)))





smooth=1
def dice_coef(y_true, y_pred):
    # y_true = K.cast(y_true , 'float32')  # Ensure y_true is float32
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    y_predf = K.round(y_predf)


    And=K.sum(y_truef* y_predf)
    
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))




op=keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=0.004,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)


# # 
# model.compile(optimizer=op, loss=dice_loss, metrics=['accuracy',dice_coef,pre,rec,hausdorff_distance])
# # model.compile(optimizer=keras.optimizers.Adam(learning_rate=10e-6), loss=dice_loss, metrics=['accuracy',dice_coef])


# checkpointer=tf.keras.callbacks.ModelCheckpoint('urd_axial_batch_16.h5',verbose=1,save_best_only=True)
# callbacks=[checkpointer,tf.keras.callbacks.EarlyStopping(patience=1000,monitor='val_loss'),
#            tf.keras.callbacks.TensorBoard(log_dir='callback_unet')]


# results=model.fit(train_data_mri,train_data_mask,validation_data=(validation_data_mri,validation_data_mask),shuffle=False,validation_steps=1,batch_size=32,epochs=1000,callbacks=[callbacks,csv_logger])






train_data_mri = np.load('../Load_Data_with_Distribution_Batch/train_data_mri_fold_1.npy').astype(np.float32)
# train_data_mri= np.floor(train_data_mri).astype(np.float32)
# 
train_data_mask=np.load('../Load_Data_with_Distribution_Batch/train_data_mask_fold_1.npy').astype(np.float32)




validation_data_mri=np.load('../Load_Data_with_Distribution_Batch/validation_data_mri_fold_1.npy').astype(np.float32)

# validation_data_mri= np.floor(validation_data_mri).astype(np.float32)

validation_data_mask=np.load('../Load_Data_with_Distribution_Batch/validation_data_mask_fold_1.npy').astype(np.float32)


model.compile(optimizer=op, loss=dice_loss, metrics=['accuracy',dice_coef])
# # model.compile(optimizer=keras.optimizers.Adam(learning_rate=10e-6), loss=dice_loss, metrics=['accuracy',dice_coef])


checkpointer=tf.keras.callbacks.ModelCheckpoint('urd_axial_batch_16_b.h5',verbose=1,save_best_only=True)
callbacks=[checkpointer,tf.keras.callbacks.EarlyStopping(patience=1000,monitor='val_loss'),
           tf.keras.callbacks.TensorBoard(log_dir='callback_unet')]




import time

number_of_epoch=10



for epoch in range(number_of_epoch):


    for batch in range(train_data_mask.shape[0]):










        if batch <  train_data_mask.shape[0] -1 :
    
            loss,acc,dice=model.train_on_batch( train_data_mri[batch,:,:,:,:],train_data_mask[batch,:,:,:,:] , sample_weight=None, class_weight=None, return_dict=False)


            percent = (batch + 1) / train_data_mask.shape[0]
    # Create the progress bar
            bar_length = 40
            bar = '#' * int(bar_length * percent) + '-' * (bar_length - int(bar_length * percent))
            # Print the progress bar
            print(f'\r{" " * (bar_length + 10)}', end='')  # Clear the line
            print(f'\r[{bar}] {int(percent * 100)}%', end='', flush=True)  # Print the new progress

            
            # Simulate a delay
            time.sleep(0.1)







        if batch == train_data_mask.shape[0] -1 :
                    
            loss,acc,dice=model.train_on_batch( train_data_mri[batch,:,:,:,:],train_data_mask[batch,:,:,:,:] , sample_weight=None, class_weight=None, return_dict=False)

            all_val_loss=[]
            all_val_acc=[]
            all_val_dice=[]


            for val_batch in range(validation_data_mask.shape[0]):
               

                val_loss,val_acc,val_dice=model.test_on_batch( validation_data_mri[val_batch,:,:,:,:], validation_data_mask[val_batch,:,:,:,:], sample_weight=None, return_dict=False)

                all_val_loss.append(val_loss)
                all_val_acc.append(val_acc)
                all_val_dice.append(val_dice)


            print('Loss:', loss , 'Accuracy:', acc , 'Dice:', dice , 'Val_Loss:', np.mean(all_val_loss) , 'Val_Accuracy:', np.mean(all_val_acc) , 'Val_Dice:' , np.mean(all_val_dice) )






