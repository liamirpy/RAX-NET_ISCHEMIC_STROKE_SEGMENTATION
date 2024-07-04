from tensorflow.keras import layers, models, Input

import tensorflow as tf
import numpy as np
import random





def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)
class RAXNet:
    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes


    def build_model(self):
        """
        Build the RAX_NET model.
        
        Returns:
        - model: Keras Model, the constructed neural network model
        """
        
        # Input layer with specified image size and a single channel
        input_layer = Input(shape=self.image_size + (1,))
        
        # Initial convolution layer
        initial_conv = input_layer

        # First Convolution Block
        x = layers.Conv2D(32, 3, strides=2, padding="same")(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        conv_2 = x  # Save the output for residual connections
        
        previous_block_output = x  # Initialize the previous block output for residual connections
        
        # Downsampling Blocks
        for num_filters in [64, 128, 256]:
            # Convolution layers
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(num_filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(num_filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            
            # Additional layers for higher number of filters
            # if num_filters in [128, 256]:
            #     x = layers.Activation("relu")(x)
            #     x = layers.Conv2D(num_filters, 3, padding="same")(x)
            #     x = layers.BatchNormalization()(x)
                
            #     x = layers.Activation("relu")(x)
            #     x = layers.SeparableConv2D(num_filters, 3, padding="same")(x)
            #     x = layers.BatchNormalization()(x)
            
            # Downsampling with max pooling
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
            
            # Residual connection
            # residual = layers.Conv2D(num_filters, 1, strides=2, padding="same")(previous_block_output)
            # x = layers.add([x, residual])  
            
            # Save outputs for upsampling residuals
            # if num_filters == 64:
            #     conv_3 = x
            # elif num_filters == 128:
            #     conv_4 = x
            
            previous_block_output = x  # Update previous block output for the next iteration

        # Upsampling Blocks
        for num_filters in [256, 128, 64, 32]:
            # Transpose convolution layers
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(num_filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(num_filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            
            # Additional layers for higher number of filters
            # if num_filters in [256, 128]:
            #     x = layers.Activation("relu")(x)
            #     x = layers.Conv2DTranspose(num_filters, 3, padding="same")(x)
            #     x = layers.BatchNormalization()(x)
                
            #     x = layers.Activation("relu")(x)
            #     x = layers.Conv2DTranspose(num_filters, 3, padding="same")(x)
            #     x = layers.BatchNormalization()(x)
            
            # Upsampling
            x = layers.UpSampling2D(2)(x)
            
            # Adding residual connections during upsampling
            # if num_filters == 32:
            #     x = self.add_upsampling_residual(x, previous_block_output, initial_conv, num_filters)
            
            # elif num_filters == 64:
            #     x = self.add_upsampling_residual(x, previous_block_output, conv_2, num_filters)
            
            # elif num_filters == 128:
            #     x = self.add_upsampling_residual(x, previous_block_output, conv_3, num_filters)
            
            # elif num_filters == 256:
            #     x = self.add_upsampling_residual(x, previous_block_output, conv_4, num_filters)
            
            # previous_block_output = x  # Update previous block output for the next iteration
        
        # Output layer 
        output_layer = layers.Conv2D(self.num_classes, 1, activation="sigmoid", padding="same")(x)
        
        # Define the model
        model = models.Model(input_layer, output_layer)
        return model

# # Usage example
# image_size = (128, 128)  # Example image size
# num_classes = 10  # Example number of classes

# rax_net = RAXNet(image_size, num_classes)
# model = rax_net.build_model()
# model.summary()