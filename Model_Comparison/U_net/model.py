from tensorflow.keras import layers, models, Input

import tensorflow as tf
import numpy as np
import random





def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)





from keras.layers import Activation, Reshape, Lambda, dot, add, Input, BatchNormalization, ReLU, DepthwiseConv2D, Concatenate
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def attention_block(x, g, inter_channels):
    """
    Attention Block
    :param x: Input feature map from encoder
    :param g: Input feature map from decoder
    :param inter_channels: Number of filters for intermediate convolution layers
    :return: Attention weighted feature map
    """
    theta_x = layers.Conv2D(inter_channels, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g = layers.Conv2D(inter_channels, (1, 1), padding='same')(g)
    
    add_xg = layers.Add()([theta_x, phi_g])
    add_xg = layers.Activation('relu')(add_xg)
    
    psi = layers.Conv2D(1, (1, 1), padding='same')(add_xg)
    psi = layers.Activation('sigmoid')(psi)
    
    upsample_psi = layers.UpSampling2D(size=(2, 2))(psi)
    upsample_psi = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': x.shape[3]})(upsample_psi)
    
    y = layers.Multiply()([upsample_psi, x])
    
    return y

def attention_unet(input_shape):
    inputs = layers.Input(input_shape+ (1,))
    
    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    
    # Decoder
    up6 = layers.Conv2D(512, (2, 2), activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv5))
    att6 = attention_block(conv4, up6, 256)
    merge6 = layers.Concatenate(axis=3)([att6, up6])
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2D(256, (2, 2), activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
    att7 = attention_block(conv3, up7, 128)
    merge7 = layers.Concatenate(axis=3)([att7, up7])
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    
    up8 = layers.Conv2D(128, (2, 2), activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv7))
    att8 = attention_block(conv2, up8, 64)
    merge8 = layers.Concatenate(axis=3)([att8, up8])
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    
    up9 = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv8))
    att9 = attention_block(conv1, up9, 32)
    merge9 = layers.Concatenate(axis=3)([att9, up9])
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    
    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    model = models.Model(inputs=[inputs], outputs=[conv10])
    
    return model

# Example usage

