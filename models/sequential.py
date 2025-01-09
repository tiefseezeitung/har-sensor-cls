#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for building Keras models with sequential input.
Inputs are sequences: either 3-dimensional time-series data or 1-dimensional feature arrays.
Parameters:
- sensor_lst: list with string names of each sensor
- input_dimensions: list with corresponding data shapes
- num_classes: number of classes to predict probabilities for
- feat_shape: shape of higher feature array input
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, concatenate, \
    Dense, Dropout, Attention, BatchNormalization, Activation, \
    Conv1D, MaxPooling1D, MultiHeadAttention, GlobalAveragePooling1D, \
    LayerNormalization, Add, Layer
    #SimpleRNN, LSTM, Bidirectional, GRU
from keras import ops
from keras.regularizers import l2

###############################################################################
# Layers and block components

class ReduceMeanLayer(Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1)

def build_simple_lstm_block(input_tensor, units, rate_do, rate_rdo, reg):
    """Build simple LSTM block with self-attention."""
    x = LSTM(units, dropout=rate_do, recurrent_dropout=rate_rdo, 
             kernel_regularizer=l2(reg), return_sequences=True)(input_tensor)
    # Self attention and pooling
    x = Attention(use_scale=True)([x, x])
    x = GlobalAveragePooling1D()(x)
    return x

def build_2llstm_block(input_tensor, units, rate_do, rate_rdo, reg):
    """Build a two-layer LSTM block with batch normalization and self-attention."""
    x = LSTM(units, dropout=rate_do, recurrent_dropout=rate_rdo, 
             kernel_regularizer=l2(reg), return_sequences=True)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = LSTM(units // 2, dropout=rate_do, recurrent_dropout=rate_rdo, 
             kernel_regularizer=l2(reg), return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Self-attention and pooling
    x = Attention(use_scale=True)([x, x])
    x = GlobalAveragePooling1D()(x)
    return x

def build_conv1d_block(input_tensor, units):
    x = Conv1D(units, kernel_size=3, activation='relu', padding='same')(input_tensor)
    x = MaxPooling1D(pool_size=(2), padding='same')(x)
    x = Conv1D(units // 2, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(2), padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    return x

###############################################################################
###############################################################################
# Models: Inputs are 3-dimensional time-series arrays

# LSTM with Self Attention
def build_LSTM_SA(sensor_lst, input_dimensions, num_classes, units=256, reg=0.0001, print_summary=True):
    """ Builds a multi-input, LSTM model with attention and 
    classification head. Returns a keras model.

    Parameters:
    sensor_lst -- list of sensor names (strings).
    input_dimensions -- list of input shapes (tuples) for each sensor.
    num_classes -- Number of output classes for classification (int).
    units -- Number of units in the LSTM layer (int).
    reg -- L2 regularization strength for LSTM layers (float). Defaults to 0.0001.
    print_summary -- Boolean to print keras model summary. Defaults to True.
    
    Returns: keras.Model
    """
    assert len(sensor_lst)==len(input_dimensions)
    
    inputs = [Input(shape=input_dim, name=sensor) for input_dim, sensor in zip(input_dimensions, sensor_lst)]

    rate_do = 0.15
    rate_rdo = 0.15
    
    # LSTM
    lstm_blocks = [build_simple_lstm_block(inp, units, rate_do, rate_rdo, reg) for inp in inputs]
    
    # Concatenate across sensors
    merged = concatenate(lstm_blocks, axis=-1) if len(lstm_blocks) > 1 else lstm_blocks[0]
     
    # Fully connected layers
    dense_layer = Dense(units * 2, activation='relu', kernel_regularizer=l2(0.001))(merged)
    dense_layer = Dropout(0.1)(dense_layer)
    output = Dense(num_classes, activation='softmax')(dense_layer)
    
    model = Model(inputs=inputs, outputs=output)
    if print_summary: model.summary()
    return model

# Best recurrent model for sequence input of HA24 study dataset
# !!! 2 LSTM blocks including BN and ReLU
def build_2layer_LSTM(sensor_lst, input_dimensions, num_classes, units=256, reg=0.0001, print_summary=True):
    """ Builds a multi-input, two-layer LSTM model with attention and 
    classification head.
    
    Parameters:
    sensor_lst -- List of sensor names (strings).
    input_dimensions -- List of input shapes (tuples) for each sensor.
    num_classes -- Number of output classes for classification (int).
    units -- Number of units in the first LSTM layer (int). Defaults to 256 units.
    reg -- L2 regularization strength for LSTM layers (float). Defaults to 0.0001.
    print_summary -- Boolean to print keras model summary. Defaults to True.
    
    Returns: keras.Model
    """
    assert len(sensor_lst)==len(input_dimensions)
    
    inputs = [Input(shape=input_dim, name=sensor) for input_dim, sensor in zip(input_dimensions, sensor_lst)]

    rate_do = 0.15
    rate_rdo = 0.15
    
    # 2 Layer LSTM
    lstm_blocks = [build_2llstm_block(inp, units, rate_do, rate_rdo, reg) for inp in inputs]
    
    # Concatenate across sensors
    merged = concatenate(lstm_blocks, axis=-1) if len(lstm_blocks) > 1 else lstm_blocks[0]
    
    # Fully connected layers
    dense_layer = Dense(units * 2, activation='relu', kernel_regularizer=l2(0.001))(merged)
    dense_layer = Dropout(0.1)(dense_layer)
    output = Dense(num_classes, activation='softmax')(dense_layer)
    
    model = Model(inputs=inputs, outputs=output)
    if print_summary: model.summary()
    return model

# 1d CNN for sequences
# results comparable to simple LSTM with Self-Attention but much smaller model
def build_1DCNN(sensor_lst, input_dimensions, num_classes, units=128):
    """Builds a multi-input 2 layered 1D CNN with classification head.
    
    Parameters:
        sensor_lst -- List of sensor names (strings) corresponding to the inputs.
        input_dimensions -- List of input shapes (tuples) for each sensor.
        num_classes -- Number of output classes for the classification task (int).
        units -- Number of units for first 1D Convolution (int).
    
    Returns: keras.Model
    """
    assert len(sensor_lst)==len(input_dimensions)
    inputs = [Input(shape=input_dim, name=sensor) for input_dim, sensor in zip(input_dimensions, sensor_lst)]

    # Convolutional layers
    conv_blocks = [build_conv1d_block(inp, units) for inp in inputs]

    # Concatenate across sensors
    merged = concatenate(conv_blocks, axis=-1) if len(conv_blocks) > 1 else conv_blocks[0]
     
    # Fully connected layers
    dense_layer = Dense(units * 2, activation='relu')(merged)
    dense_layer = Dropout(0.2)(dense_layer)
    output = Dense(num_classes, activation='softmax')(dense_layer)
    
    # Create model
    model = Model(inputs=inputs, outputs=output)
    model.summary()
    return model

def build_transformer(sensor_lst, input_dimensions, num_classes, num_heads=8, 
                      emb_dim=128, dropout=0.01, num_transformer_blocks=1, print_summary=False):
    """Builds a multi-input Transformer classification model with positional embeddings 
    and residual connections.

    Parameters:
    sensor_lst -- List of sensor names (strings) corresponding to the inputs.
    input_dimensions -- List of input shapes (tuples) for each sensor.
    num_classes -- Number of output classes for the classification task (int).
    num_heads -- Number of attention heads in the Multi-Head Attention layer (int). 
        Defaults to 8.
    emb_dim -- Dimension of the dense projection and positional embeddings (int). 
        Defaults to 128.
    dropout -- Dropout rate applied in Transformer MLP block (float).
        Defaults to 0.01.
    num_transformer_blocks -- Number of Transformer encoder blocks per sensor (int). 
        Defaults to 1.
    print_summary -- Prints the keras model summary (boolean). Defaults to False.

    Returns: keras.Model
    """    
    assert len(sensor_lst)==len(input_dimensions)
    inputs = [Input(shape=input_dim, name=sensor) for input_dim, sensor in zip(input_dimensions, sensor_lst)]

    mlp_units = [emb_dim * 2, emb_dim]
    tf_blocks = []
    for i in range(len(inputs)):
        
        projection = Dense(emb_dim)(inputs[i])
        # Create and add positional embedding to the projected tensor
        positions = ops.expand_dims(
            ops.arange(start=0, stop=inputs[i].shape[1], step=1), axis=0)
        position_embedding = Embedding(input_dim=inputs[i].shape[1], output_dim=emb_dim)(positions)
        # Add embeddings to projected tensor
        encoded_x = projection + position_embedding
          
        # Encoder
        for _ in range(num_transformer_blocks):
            # Normalization, Multi-head self-attention, add Residual
            x = LayerNormalization(epsilon=1e-6)(encoded_x)
            x = MultiHeadAttention(num_heads=num_heads, key_dim=input_dimensions[i][-1] // num_heads)(x, x)
            x = Dropout(dropout)(x)
            res = Add()([x, encoded_x])  # Residual connection
            
            # Feedforward layer, add Residual
            x = LayerNormalization(epsilon=1e-6)(res)
            for units in mlp_units:
                x = Dense(units, activation='relu')(x) 
                x = Dropout(dropout)(x)
            x = Add()([x, res]) # Residual connection
            encoded_x = x
        
        x = GlobalAveragePooling1D()(encoded_x)
        tf_blocks.append(x)
        
    # Concatenate across sensors
    merged = concatenate(tf_blocks, axis=-1) if len(tf_blocks) > 1 else tf_blocks[0]
    # Fully connected layers for classification
    dense_layer = Dense(256, activation="relu")(merged)
    dense_layer = Dropout(0.1)(dense_layer)
    output = Dense(num_classes, activation="softmax")(dense_layer)
    
    model = Model(inputs=inputs, outputs=output)
    if print_summary: model.summary()
    return model

###############################################################################
###############################################################################
# Models: Inputs are one dimensional feature arrays

# !!!
def build_FF_Feats(sensor_lst, feat_shape, num_classes, units=512, print_summary=True):
    """Builds a multi-input feedforward NN, receiving arrays containing extracted features.
    
    Parameters:
    sensor_lst -- List of sensor names (strings) corresponding to the inputs.
    feat_shape -- Feat shape (tuple shaped (n,)).
    num_classes -- Number of output classes for the classification task (int).
    units -- Number of units used in the first Dense layer (int). Defaults to 512.
    print_summary -- Boolean to print keras model summary. Defaults to True.

    Returns: keras.Model
    """
    inputs_feat = [Input(shape=feat_shape, name=sensor+'_feats') for sensor in sensor_lst]

    dense_blocks = []
    for i in range(len(inputs_feat)):
        x = Dense(units, activation='relu')(inputs_feat[i])
        x = Dense(units // 2, activation='relu')(x)
        dense_blocks.append(x)

    # Concatenate across sensors
    merged = concatenate(dense_blocks, axis=-1) if len(dense_blocks) > 1 else dense_blocks[0]

    dense_layer = Dense(256, activation='relu')(merged)
    dense_layer = Dropout(0.1)(dense_layer)
    output = Dense(num_classes, activation='softmax')(dense_layer)
    
    # Define model with multiple inputs
    model = Model(inputs=inputs_feat, outputs=output)
    if print_summary: model.summary()
    return model
