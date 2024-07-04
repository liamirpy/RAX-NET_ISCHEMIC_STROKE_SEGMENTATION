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


def conv2d_bn_relu(input, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def fsm(x):
    channel_num = x.shape[-1]

    res = x

    x = conv2d_bn_relu(x, filters=int(channel_num // 8), kernel_size=(3, 3))

    # x = non_local_block(x, compression=2, mode='dot')

    ip = x
    ip_shape = K.int_shape(ip)
    batchsize, dim1, dim2, channels = ip_shape
    intermediate_dim = channels // 2
    rank = 4
    if intermediate_dim < 1:
        intermediate_dim = 1

    # theta path
    theta = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    theta = Reshape((-1, intermediate_dim))(theta)

    # phi path
    phi = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    phi = Reshape((-1, intermediate_dim))(phi)

    # dot
    f = dot([theta, phi], axes=2)
    size = K.int_shape(f)
    # scale the values to make it size invariant
    f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    # g path
    g = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    g = Reshape((-1, intermediate_dim))(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])
    y = Reshape((dim1, dim2, intermediate_dim))(y)
    y = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-5))(y)
    y = add([ip, y])

    x = y
    x = conv2d_bn_relu(x, filters=int(channel_num), kernel_size=(3, 3))
    print(x)

    x = add([x, res])
    return x


def create_fsm_model(input_shape=(224, 192, 1)):
    input_img = Input(shape=input_shape)
    x = fsm(input_img)
    model = Model(input_img, x)
    model.summary()
    return model





from keras.models import Sequential
from keras.layers import Reshape
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Concatenate, ReLU, DepthwiseConv2D, add,BatchNormalization
from keras.layers import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.optimizers import Adam , SGD
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.regularizers import l2



def conv2d_bn_relu(input, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def depth_conv_bn_relu(input, filters, kernel_size, strides=(1, 1), padding='same', dilation_rate=(1, 1),
                   initializer='he_normal', regularizer=l2(1e-5)):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding,
                        depthwise_initializer=initializer, use_bias=False, depthwise_regularizer=regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding=padding,
               kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def x_block(x, channels):
    res = conv2d_bn_relu(x, filters=channels, kernel_size=(1, 1))
    x = depth_conv_bn_relu(x, filters=channels, kernel_size=(3, 3))
    x = depth_conv_bn_relu(x, filters=channels, kernel_size=(3, 3))
    x = depth_conv_bn_relu(x, filters=channels, kernel_size=(3, 3))
    x = add([x, res])
    return x


def xception_unet(input_shape=(208, 208), pretrained_weights_file=None):
    input = Input(input_shape+ (1,))

    # stage_1
    x = x_block(input, channels=64)
    stage_1 = x

    # stage_2
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = x_block(x, channels=128)
    stage_2 = x

    # stage_3
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = x_block(x, channels=256)
    stage_3 = x

    # stage_4
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = x_block(x, channels=512)
    stage_4 = x

    # stage_5
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = x_block(x, channels=1024)
    x = fsm(x)

    # stage_4
    x = UpSampling2D(size=(2,2))(x)
    x = conv2d_bn_relu(x, filters=512, kernel_size=3)
    x = Concatenate()([stage_4, x])
    x = x_block(x, channels=512)

    # stage_3
    x = UpSampling2D(size=(2,2))(x)
    x = conv2d_bn_relu(x, filters=256, kernel_size=3)
    x = Concatenate()([stage_3, x])
    x = x_block(x, channels=256)

    # stage_2
    x = UpSampling2D(size=(2, 2))(x)
    x = conv2d_bn_relu(x, filters=128, kernel_size=3)
    x = Concatenate()([stage_2, x])
    x = x_block(x, channels=128)

    # stage_1
    x = UpSampling2D(size=(2, 2))(x)
    x = conv2d_bn_relu(x, filters=64, kernel_size=3)
    x = Concatenate()([stage_1, x])
    x = x_block(x, channels=64)

    # output
    x = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(x)

    # create model
    model = Model(input, x)
    model.summary()
    print('Create X-Net with N block, input shape = {}, output shape = {}'.format(input_shape, model.output.shape))

    # load weights
    if pretrained_weights_file is not None:
        model.load_weights(pretrained_weights_file, by_name=True, skip_mismatch=True)

    return model

