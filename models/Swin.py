#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains Swin Transformer Model and function for building Keras model.
'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows'
by by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
(https://arxiv.org/abs/2103.14030)

Implemented after https://keras.io/examples/vision/swin_transformers/
Parameters:
- sensor_lst: list with string names of each sensor
- input_dimensions: list with corresponding dimensions (shape of datapoints)
- num_classes: number of classes to predict probabilities for
"""

import tensorflow as tf
import numpy as np
import keras
from keras import layers
from keras.models import Model
from keras.layers import Input, Embedding, concatenate, \
    Dense, Dropout,  Activation, GlobalAveragePooling1D, \
    Layer, LayerNormalization, Resizing
from keras import ops


# window_partition adjusted from https://keras.io/examples/vision/swin_transformers/
def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = ops.reshape(
        x,
        (
            -1,
            patch_num_y,
            window_size,
            patch_num_x,
            window_size,
            channels,
        ),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = ops.reshape(x, (-1, window_size, window_size, channels))
    return windows

# window_reverse adjusted from https://keras.io/examples/vision/swin_transformers/
def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = ops.reshape(
        windows,
        (
            -1,
            patch_num_y,
            patch_num_x,
            window_size,
            window_size,
            channels,
        ),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (-1, height, width, channels))
    return x


# WindowAttention class from https://keras.io/examples/vision/swin_transformers/
class WindowAttention(Layer):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = Dropout(dropout_rate)
        self.proj = Dense(dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        
        self.relative_position_index = keras.Variable(
            initializer=relative_position_index,
            shape=relative_position_index.shape,
            dtype="int",
            trainable=False,
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = ops.reshape(x_qkv, (-1, size, 3, self.num_heads, head_dim))
        x_qkv = ops.transpose(x_qkv, (2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = ops.transpose(k, (0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = ops.reshape(self.relative_position_index, (-1,))
        relative_position_bias = ops.take(
            self.relative_position_bias_table,
            relative_position_index_flat,
            axis=0,
        )
        relative_position_bias = ops.reshape(
            relative_position_bias,
            (num_window_elements, num_window_elements, -1),
        )
        relative_position_bias = ops.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + ops.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            mask_float = ops.cast(
                ops.expand_dims(ops.expand_dims(mask, axis=1), axis=0),
                "float32",
            )
            attn = ops.reshape(attn, (-1, nW, self.num_heads, size, size)) + mask_float
            attn = ops.reshape(attn, (-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = ops.transpose(x_qkv, (0, 2, 1, 3))
        x_qkv = ops.reshape(x_qkv, (-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv
    
        def get_config(self):
            config = super().get_config()
            config.update({"dim": self.dim,
                           "window_size": self.window_size,
                           "num_heads": self.num_heads})
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

# SwinTransformer class from https://keras.io/examples/vision/swin_transformers/
class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = Dropout(dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                Dense(num_mlp),
                Activation(keras.activations.gelu),
                Dropout(dropout_rate),
                Dense(dim),
                Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = ops.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = ops.reshape(
                mask_windows, [-1, self.window_size * self.window_size]
            )
            attn_mask = ops.expand_dims(mask_windows, axis=1) - ops.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = ops.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = ops.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = keras.Variable(
                initializer=attn_mask,
                shape=attn_mask.shape,
                dtype=attn_mask.dtype,
                trainable=False,
            )

    def call(self, x, training=False):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = ops.reshape(x, (-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = ops.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = ops.reshape(
            x_windows, (-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = ops.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, channels),
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = ops.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x
        

        x = ops.reshape(x, (-1, height * width, channels))
        x = self.drop_path(x, training=training)

        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim,
                       "num_patch": self.num_patch,
                       "num_heads": self.num_heads,
                       "shift_size": self.shift_size,
                       "window_size": self.window_size,
                       "num_mlp": self.num_mlp})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Patches class adjusted from https://keras.io/examples/vision/image_classification_with_vision_transformer/
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

# PatchEncoder class adjusted from https://keras.io/examples/vision/image_classification_with_vision_transformer/
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim)
        
    def build(self, input_shape):
       # This method is called the first time the layer is used
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
    
# PatchMerging class from https://keras.io/examples/vision/swin_transformers/
class PatchMerging(Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = None

    def build(self, input_shape):
        # Initialize the linear transformation layer with the correct input shape
        self.linear_trans = Dense(2 * self.embed_dim, use_bias=False)
        super().build(input_shape) 
        
    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.shape
        x = ops.reshape(x, (-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = ops.concatenate((x0, x1, x2, x3), axis=-1)
        x = ops.reshape(x, (-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_patch": self.num_patch, "embed_dim": self.embed_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# !!! Swin Transformer
def build_Swin(sensor_lst, input_dimensions, num_classes, num_heads=16, 
               projection_dim=128, window_size=4, num_mlp = 256, num_transformer_blocks=1,
               dropout_rate=0.01, qkv_bias = True, shift_size = 2, print_summary=False):
    """ 
    Builds a multi-input Swin Transformer for classification.
    
    Patch Merging is not applied between the two Transformer blocks to allow small 
    image sizes to be processed (after halfing the image, shifting is not possible 
    when the image size is to small). If num_transformer_block > 1, the build 
    of the model will throw an error, if the image is to small.
    
    Parameters:
    sensor_lst -- List of sensor names (strings) corresponding to the inputs.
    input_dimensions -- List of input shapes (tuples) for each sensor.
    num_classes -- Number of output classes for the classification task (int).
    num_heads -- Number of heads for Multi-Head Attention (int). Defaults to 16.
    projection_dim -- Dimension for projecting patches for embedding (int).
        Defaults to 128.
    window_size -- Size for patches for conducting Multi-Headed Attention.
        Defaults to 4.
    num_mlp -- Number of units for the MLP in the encoder. Defaults to 256.
    num_transformer_blocks -- Number of transformer block iteration, each 
        consisting of a non-shifting and shifting Block (int). Defaults to 1.
    dropout_rate -- Dropout rate before and after MLP layer. Defaults to 0.01.
    qkv_bias -- If True creates bias for query value key (bool). Defaults to True.
    shift_size -- Size of shifting window (int). Defaults to 2.
    print_summary -- Calls model.summary() from keras. Defaults to False
    
    Returns: keras.Model
    """
    inputs = [Input(shape=input_dim, name=sensor) for input_dim, sensor in zip(input_dimensions, sensor_lst)]

    blocks = []
    
    for i in range(len(sensor_lst)):

        # Initial image reshaping
        image_shape = (64, 64)
        patch_shape = (16, 16) 
        
        num_patches = (image_shape[0] // patch_shape[0]) * (image_shape[1] // patch_shape[1])
        num_patches_x = (image_shape[0] // patch_shape[0]) 
        num_patches_y = (image_shape[1] // patch_shape[1])
        
        # Resize image and divide into patches
        resized_images = Resizing(image_shape[0], image_shape[1])(inputs[i])
        patches = Patches(patch_shape)(resized_images)

        # Add positional embeddings
        x = PatchEncoder(num_patches, projection_dim)(patches)
        
        block_projection_dim = projection_dim
        
        for j in range(num_transformer_blocks):

            # Normal Windows
            x = SwinTransformer(
                dim=block_projection_dim,
                num_patch=(num_patches_x, num_patches_y),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                num_mlp=num_mlp,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
            )(x)
            # Shifted Windows
            x = SwinTransformer(
                dim=block_projection_dim,
                num_patch=(num_patches_x, num_patches_y),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                num_mlp=num_mlp,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
            )(x)
            
            # Downsample if not last iteration
            if j < num_transformer_blocks - 1: 
                x = PatchMerging((num_patches_x, num_patches_y), embed_dim=projection_dim)(x)
                block_projection_dim *= 2
                num_patches_x //= 2
                num_patches_y //= 2
               
        x = GlobalAveragePooling1D()(x)
        blocks.append(x)
            
    # Concatenate across sensor outputs
    merged = concatenate(blocks, axis=-1) if len(blocks) > 1 else blocks[0]
    
    dense_layer = Dense(2 * projection_dim, activation="relu")(merged)
    dense_layer = Dropout(0.1)(dense_layer)
    output = Dense(num_classes, activation="softmax")(dense_layer)
    
    model = Model(inputs=inputs, outputs=output)
    if print_summary: model.summary()
    return model

def build_Swin_orig(sensor_lst, input_dimensions, num_classes, num_heads=16, 
               projection_dim=128, window_size=4, num_mlp = 256, num_transformer_blocks=1,
               dropout_rate=0.01, qkv_bias = True, shift_size = 2, print_summary=False):
    """ 
    Builds a multi-input Swin Transformer for classification.

    Performs Patch Merging between the Window and Shifted Window Transformer blocks 
    as in the original paper (https://arxiv.org/abs/2103.14030).
    
    Parameters:
    sensor_lst -- List of sensor names (strings) corresponding to the inputs.
    input_dimensions -- List of input shapes (tuples) for each sensor.
    num_classes -- Number of output classes for the classification task (int).
    num_heads -- Number of heads for Multi-Head Attention (int). Defaults to 16.
    projection_dim -- Dimension for projecting patches for embedding (int).
        Defaults to 128.
    window_size -- Size for patches for conducting Multi-Headed Attention.
        Defaults to 4.
    num_mlp -- Number of units for the MLP in the encoder. Defaults to 256.
    num_transformer_blocks -- Number of transformer block iteration, each 
        consisting of a non-shifting and shifting Block (int). Defaults to 1.
    dropout_rate -- Dropout rate before and after MLP layer. Defaults to 0.01.
    qkv_bias -- If True creates bias for query value key (bool). Defaults to True.
    shift_size -- Size of shifting window (int). Defaults to 2.
    print_summary -- Calls model.summary() from keras. Defaults to False
    
    Returns: keras.Model
    """
    
    inputs = [Input(shape=input_dim, name=sensor) for input_dim, sensor in zip(input_dimensions, sensor_lst)]
    blocks = []
    
    for i in range(len(sensor_lst)):
        # Initial image reshaping
        image_shape = (128, 128)
        patch_shape = (16, 16) 
        
        num_patches = (image_shape[0] // patch_shape[0]) * (image_shape[1] // patch_shape[1])
        num_patches_x = (image_shape[0] // patch_shape[0]) 
        num_patches_y = (image_shape[1] // patch_shape[1])
        
        # Resize image and divide into patches
        resized_images = Resizing(image_shape[0], image_shape[1])(inputs[i])
        patches = Patches(patch_shape)(resized_images)

        # Add positional embeddings
        x = PatchEncoder(num_patches, projection_dim)(patches)
        
        block_projection_dim = projection_dim
        
        for j in range(num_transformer_blocks):

            # Normal Windows
            x = SwinTransformer(
                dim=block_projection_dim,
                num_patch=(num_patches_x, num_patches_y),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                num_mlp=num_mlp,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
            )(x)
            
            # Patch Merge
            x = PatchMerging((num_patches_x, num_patches_y), embed_dim=projection_dim)(x)
            block_projection_dim *= 2
            num_patches_x //= 2
            num_patches_y //= 2

            # Shifted Windows
            x = SwinTransformer(
                dim=block_projection_dim,
                num_patch=(num_patches_x, num_patches_y),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                num_mlp=num_mlp,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
            )(x)
            
            # Downsample if not last iteration
            if j < num_transformer_blocks - 1: 
                x = PatchMerging((num_patches_x, num_patches_y), embed_dim=projection_dim)(x)
                block_projection_dim *= 2
                num_patches_x //= 2
                num_patches_y //= 2
                
        x = GlobalAveragePooling1D()(x)
        blocks.append(x)
            
    # Concatenate across sensor outputs
    merged = concatenate(blocks, axis=-1) if len(blocks) > 1 else blocks[0]
    
    dense_layer = Dense(2 * projection_dim, activation="relu")(merged)
    dense_layer = Dropout(0.1)(dense_layer)
    output = Dense(num_classes, activation="softmax")(dense_layer)
    
    model = Model(inputs=inputs, outputs=output)
    if print_summary: model.summary()
    return model