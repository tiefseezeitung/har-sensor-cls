#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains Vision Transformer Model and function for building Keras model.
Implemented after https://keras.io/examples/vision/image_classification_with_vision_transformer/
Inputs are spectrogram images.
Parameters:
- sensor_lst: list with string names of each sensor
- input_dimensions: list with corresponding dimensions (shape of data samples)
- num_classes: number of classes to predict probabilities for
- feat_shape: shape of higher feature array input
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, concatenate, Dense, Dropout, \
    MultiHeadAttention, LayerNormalization, Add, Layer, Resizing
from keras import ops

# Class for Patches adjusted from https://keras.io/examples/vision/image_classification_with_vision_transformer/
class Patches(Layer):
    def __init__(self, patch_shape, **kwargs):
        super().__init__(**kwargs) 
        self.patch_shape = patch_shape
        self.patch_size_h = patch_shape[0]
        self.patch_size_w = patch_shape[1]

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size_h
        num_patches_w = width // self.patch_size_w
        
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size_h, self.patch_size_w, 1],
            strides=[1, self.patch_size_h, self.patch_size_w, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size_h * self.patch_size_w * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_shape": self.patch_shape})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Class for PatchEncoder adjusted from https://keras.io/examples/vision/image_classification_with_vision_transformer/
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim)
        
    def build(self, input_shape):
       self.projection.build(input_shape)  # Projection Dense layer build
       self.position_embedding.build((self.num_patches, self.projection_dim))
       super().build(input_shape)
    
    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class AddClassToken(Layer):
    def __init__(self, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.cls_token = self.add_weight(shape=(1, 1, projection_dim), 
                                         initializer='random_normal', 
                                         trainable=True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_token_broadcasted = tf.broadcast_to(self.cls_token, [batch_size, 1, tf.shape(inputs)[-1]])
        return tf.concat([cls_token_broadcasted, inputs], axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"projection_dim": self.projection_dim})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ExtractClassToken(Layer):
    def call(self, inputs):
        return inputs[:, 0]

def build_transformer_block(input_tensor, projection_dim, num_heads, dropout_mlp, num_blocks, transformer_units):
    """Creates a sequence of Transformer blocks."""
    
    encoded = input_tensor
    for _ in range(num_blocks):
        # Normalization
        x = LayerNormalization(epsilon=1e-2)(encoded)
        # Multi-head self-attention
        x = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x, x)
        # Add residual connection
        res = Add()([x, encoded])  # Residual connection
        
        # Normalization
        x = LayerNormalization(epsilon=1e-2)(res)
        # Multi Layer Perceptron / Feedforward layers
        for units in transformer_units:
            x = Dense(units, activation='gelu')(x) #keras.activations.gel
            x = Dropout(dropout_mlp)(x)
        # Add residual connection
        encoded = Add()([x, res])
        
    return encoded

###############################################################################

# Best overall model
# !!! Vision Transformer, includes Class Token
def build_ViT1(sensor_lst, input_dimensions, num_classes, num_heads=8, 
                projection_dim=128, dropout=0.1, num_transformer_blocks=1, print_summary=True):
    """Vision Transformer https://arxiv.org/abs/2010.11929
    
    Parameters:
    sensor_lst -- List of sensor names (strings) corresponding to the inputs.
    input_dimensions -- List of input shapes (tuples) for each sensor.
    num_classes -- Number of output classes for the classification task (int).
    num_heads -- Number of attention heads in the multi-head self-attention 
        operation (int). Defaults to 8.
    projection_dim -- Dimensionality of the projection space used before and 
        inside the Transformer blocks (int). Defaults to 128.
    dropout -- Dropout rate applied to the feedforward layers in the encoder and 
        after extracting the class token (float). Defaults to 0.1.
    num_transformer_blocks -- Number of Transformer blocks to stack for each 
        sensor's input. Defaults to 1.
    print_summary -- Boolean for printing keras model summary. Defaults to True.

    Returns: keras.Model
    """

    assert len(sensor_lst) == len(input_dimensions)
    
    inputs = [Input(shape=input_dim, name=sensor) for input_dim, sensor in zip(input_dimensions, sensor_lst)]
    blocks = []
    
    for i in range(len(inputs)):

        image_shape = (32,48)
        patch_shape = (8,8) 
        num_patches = (image_shape[0] // patch_shape[0]) * (image_shape[1] // patch_shape[1])
        
        # Resize image and divide into patches
        resized_images = Resizing(image_shape[0], image_shape[1])(inputs[i])
        patches = Patches(patch_shape)(resized_images)

        # Add positional embeddings
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
        # Add extra class token / embedding
        encoded_patches = AddClassToken(projection_dim)(encoded_patches)

        # Number of units for MLP in Transformer encoder
        transformer_units = [projection_dim * 2, projection_dim]
        
        # Transformer encoder block
        encoded_patches = build_transformer_block(encoded_patches, projection_dim, 
                                                  num_heads=num_heads, dropout_mlp=dropout, 
                                                  num_blocks=num_transformer_blocks, 
                                                  transformer_units=transformer_units)

        # Extract only the features from the first (class) token
        x = ExtractClassToken()(encoded_patches)
        x = Dropout(dropout)(x)

        blocks.append(x)
        
    # Concatenate across sensors
    merged = concatenate(blocks, axis=-1) if len(blocks) > 1 else blocks[0]

    # Fully connected layers for classification
    dense_layer = Dense(projection_dim * 2, activation="relu")(merged)
    dense_layer = Dropout(0.1)(dense_layer)
    output = Dense(num_classes, activation="softmax")(dense_layer)
    
    model = Model(inputs=inputs, outputs=output)
    if print_summary: model.summary()
    return model
