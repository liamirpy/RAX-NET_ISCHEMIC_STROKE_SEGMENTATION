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


    def add_upsampling_residual(self, x, prev_output, downsample_layer, num_filters):
        """
        Add upsampling residual connection.

        Parameters:
        - x: current layer
        - prev_output: output from the previous block for residual connection
        - downsample_layer: layer from the downsampling part for residual connection
        - num_filters: number of filters for the convolution layers
        
        Returns:
        - x: updated layer with residual connection
        """
        # Upsample the previous block output
        upsampled_residual = layers.UpSampling2D(2)(prev_output)
        
        # Apply a 1x1 convolution to adjust the number of filters
        upsampled_residual = layers.Conv2D(num_filters, 1, padding="same")(upsampled_residual)
        
        # Apply a 1x1 convolution to the downsample layer from the downsampling part to match the dimensions
        downsample_conv = layers.Conv2D(num_filters, 1, padding="same")(downsample_layer)
        
        # Add the upsampled residual and the downsample convolution
        residual_combined = layers.add([upsampled_residual, downsample_conv])
        
        # Apply activation and further convolution and batch normalization
        residual_combined = layers.Activation("relu")(residual_combined)
        residual_conv = layers.Conv2D(num_filters, 1, padding="same")(residual_combined)
        residual_conv = layers.BatchNormalization()(residual_conv)
        
        # Add the result to the current layer
        x = layers.add([x, residual_conv])
        return x

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
            if num_filters in [128, 256]:
                x = layers.Activation("relu")(x)
                x = layers.Conv2D(num_filters, 3, padding="same")(x)
                x = layers.BatchNormalization()(x)
                
                x = layers.Activation("relu")(x)
                x = layers.SeparableConv2D(num_filters, 3, padding="same")(x)
                x = layers.BatchNormalization()(x)
            
            # Downsampling with max pooling
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
            
            # Residual connection
            residual = layers.Conv2D(num_filters, 1, strides=2, padding="same")(previous_block_output)
            x = layers.add([x, residual])  
            
            # Save outputs for upsampling residuals
            if num_filters == 64:
                conv_3 = x
            elif num_filters == 128:
                conv_4 = x
            
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
            if num_filters in [256, 128]:
                x = layers.Activation("relu")(x)
                x = layers.Conv2DTranspose(num_filters, 3, padding="same")(x)
                x = layers.BatchNormalization()(x)
                
                x = layers.Activation("relu")(x)
                x = layers.Conv2DTranspose(num_filters, 3, padding="same")(x)
                x = layers.BatchNormalization()(x)
            
            # Upsampling
            x = layers.UpSampling2D(2)(x)
            
            # Adding residual connections during upsampling
            if num_filters == 32:
                x = self.add_upsampling_residual(x, previous_block_output, initial_conv, num_filters)
            
            elif num_filters == 64:
                x = self.add_upsampling_residual(x, previous_block_output, conv_2, num_filters)
            
            elif num_filters == 128:
                x = self.add_upsampling_residual(x, previous_block_output, conv_3, num_filters)
            
            elif num_filters == 256:
                x = self.add_upsampling_residual(x, previous_block_output, conv_4, num_filters)
            
            previous_block_output = x  # Update previous block output for the next iteration
        
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