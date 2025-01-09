#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains functions for building 2d CNN Keras model.
Inputs are spectrogram images.
Parameters:
- sensor_lst: list with string names of each sensor
- input_dimensions: list with corresponding dimensions (shape of datapoints)
- num_classes: number of classes to predict probabilities for
- feat_shape: shape of higher feature array input
"""

from keras.models import Model
from keras.layers import Input, concatenate, Dense, Dropout, \
    Flatten, Conv2D, MaxPooling2D, Add, ReLU    
from keras.regularizers import l2

def build_conv_block(input_tensor, input_shape, units, conv_reg):
    """ 2 layered Convolution block with Max Pooling and residual connections."""
    #1st
    x = Conv2D(units, kernel_size=3, activation='relu', padding='same', 
               kernel_regularizer=l2(conv_reg), input_shape=input_shape)(input_tensor)
    fx = Conv2D(units, kernel_size=3, activation=None, padding='same', 
                kernel_regularizer=l2(conv_reg))(x)

    x = Add()([x,fx])
    x = ReLU()(x)
    
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    
    #2nd
    x = Conv2D(units * 2, kernel_size=3, activation='relu', padding='same', 
               kernel_regularizer=l2(conv_reg))(x)
    fx = Conv2D(units * 2, kernel_size=3, activation=None, padding='same', 
                kernel_regularizer=l2(conv_reg))(x)
    
    x = Add()([x,fx])
    x = ReLU()(x)
    
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    
    # Flatten
    x = Flatten()(x)

    return x

###############################################################################

# !!! 
# 2D CNN for spectrograms, with residual connections
# No batchnorm as it disables training process
def build_2DCNN(sensor_lst, input_dimensions, num_classes, units=64, 
                conv_reg=0.00001, print_summary=True):
    """Builds a multi-input 2D CNN including residual connections.
    
    Parameters:
    sensor_lst -- List of sensor names (strings) corresponding to the inputs.
    feat_shape -- Feat shape (tuple shaped (n,)).
    num_classes -- Number of output classes for the classification task (int).
    units -- Number of units used in the first Dense layer (int). Defaults to 512.
    conv_reg -- L2 regularization strength for Conv layers (float). 
        Defaults to 0.00001.
    print_summary -- Boolean to print keras model summary. Defaults to True.

    Returns: keras.Model
    """
    # input dimensions is list with arrays expected to be in shape (num frequency bins, num time segments, 3)     
    assert len(sensor_lst)==len(input_dimensions)
    inputs = [Input(shape=input_dim, name=sensor) for input_dim, sensor in zip(input_dimensions, sensor_lst)]

    # Convolutional layers
    conv_blocks = [build_conv_block(inp, inp_dim, units, conv_reg) for inp, inp_dim in zip(inputs, input_dimensions)]
    
    # Concatenate across sensors
    merged = concatenate(conv_blocks, axis=-1) if len(conv_blocks) > 1 else conv_blocks[0]

    dense_layer = Dense(units * 4, activation='relu', kernel_regularizer=l2(0.001))(merged)
    dense_layer = Dropout(0.1)(dense_layer)
    output = Dense(num_classes, activation='softmax')(dense_layer)
    
    # Create model
    model = Model(inputs=inputs, outputs=output)
    if print_summary: model.summary()
    return model

# Model with additional higher features
def build_CNN_FF(sensor_lst, input_dimensions, feat_shape, num_classes, units=64, 
                 units_feat=512, conv_reg=0.00001, print_summary=True):
    """Builds a multi-input 2D CNN (for spectrograms), including residual 
    connections, combined with a Feedforward NN of additional feature inputs.
    
    Parameters:
    sensor_lst -- List of sensor names (strings) corresponding to the inputs.
    input_dimensions -- List of input shapes (tuples) for each sensor.
    feat_shape -- Feat shape (tuple shaped (n,)).
    num_classes -- Number of output classes for the classification task (int).
    units -- Number of units used in first Convolution layer (int). Defaults to 64.
    units_feat -- Number of units used in the first Dense layer (int) for features. 
        Defaults to 512.
    conv_reg -- L2 regularization strength for Conv layers (float). 
        Defaults to 0.00001.
    print_summary -- Boolean to print keras model summary. Defaults to True.
    
    Returns: keras.Model
    """
    
    # input dimensions is list with arrays expected to be in shape (num frequency bins, num time segments, 3)     
    assert len(sensor_lst)==len(input_dimensions)
    inputs = [Input(shape=input_dim, name=sensor) for input_dim, sensor in zip(input_dimensions, sensor_lst)]
    inputs_feat = [Input(shape=feat_shape, name=sensor+'_feats') for sensor in sensor_lst]
        
    # Convolutional layers
    conv_blocks = [build_conv_block(inp, inp_dim, units, conv_reg) for inp, inp_dim in zip(inputs, input_dimensions)]
    
    blocks = []
    for i in range(len(inputs_feat)):
        # Feed Forward
        dense = Dense(units_feat, activation='relu')(inputs_feat[i])
        dense = Dense(units_feat // 2, activation='relu')(dense)
        
        # Concatenate both inputs
        x = concatenate([conv_blocks[i], dense], axis=-1)
        blocks.append(x)

    # Concatenate across sensors
    merged = concatenate(blocks, axis=-1) if len(blocks) > 1 else blocks[0]

    dense_layer = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(merged)
    dense_layer = Dropout(0.15)(dense_layer)
    output = Dense(num_classes, activation='softmax')(dense_layer)
    
    # Create model
    model = Model(inputs + inputs_feat, outputs=output)
    if print_summary: model.summary()
    return model