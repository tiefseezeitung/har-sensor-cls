#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements the CSWin Transformer and is written after the Pytorch 
implementation of Microsoft:
'https://github.com/microsoft/CSWin-Transformer/tree/main'
by Microsoft, model was proposed in
'CSWin Transformer: A General Vision Transformer Backbone with 
Cross-Shaped Windows' (2021) by {Xiaoyi Dong, Jianmin Bao, Dongdong Chen, 
Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen and Baining Guo}
"""
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.layers import Input, concatenate, Dense, Dropout, \
    LayerNormalization, Conv2D, Resizing, Reshape, \
    GlobalAveragePooling1D
from keras.regularizers import l2
import math


def img2windows(img, H_sp, W_sp):
    """
    Splits the input tensor `img` into a series of non-overlapping windows.
    img -- tensor shaped (B, C, H, W)
    H_sp -- height of each window.
    W_sp -- width of each window.
    """

    # Extract dynamic dimensions
    B = tf.shape(img)[0]
    C = tf.shape(img)[1]
    H = tf.shape(img)[2]
    W = tf.shape(img)[3]

    # Reshape to create windows: (B, C, H//H_sp, H_sp, W//W_sp, W_sp)
    img_reshape = tf.reshape(img, (B, C, H // H_sp, H_sp, W // W_sp, W_sp))

    # Reorder dimensions -> shape (B, H//H_sp, W//W_sp, H_sp, W_sp, C)
    img_perm = tf.transpose(img_reshape, perm=[0, 2, 4, 3, 5, 1])
    
    num_windows = (H // H_sp) * (W // W_sp)
    # Flatten to shape (-1, H_sp * W_sp, C)
    img_windows = tf.reshape(img_perm, (B*num_windows, H_sp * W_sp, C))

    return img_windows

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Reconstructs the image from windowed splits.
    img_splits_hw -- tensor shaped (B', H'*W', C)
    H_sp, W_sp -- height and width of each window
    H, W -- original height and width of the (convolved) image
    """
    # Calculate the original batch size, channel size
    B = tf.shape(img_splits_hw)[0] // ((H * W) // (H_sp * W_sp))
    C = tf.shape(img_splits_hw)[-1]
    
    # Reshape to (B, H // H_sp, W // W_sp, H_sp, W_sp, C)
    img = tf.reshape(img_splits_hw, (B, H // H_sp, W // W_sp, H_sp, W_sp, C))

    # Reorder dimensions to prepare for desired output shape (B, H, W, C)
    img = tf.transpose(img, perm=[0, 1, 3, 2, 4, 5])

    # Flatten back to (B, H, W, C)
    img = tf.reshape(img, (B, H, W, C))

    return img

class LePEAttention(tf.keras.layers.Layer):
    """
    Horizontal/vertical Attention + Locally-enhanced Positional Encoding (LePE) Layer
    """
    def __init__(self, dim, image_shape, idx, split_size=7, num_heads=8, dim_out=None,
                 attn_drop_rate=0.01, qk_scale=None, **kwargs):
        super(LePEAttention, self).__init__(**kwargs)
        self.dim = dim
        self.image_shape = image_shape
        self.resolution_h, self.resolution_w = image_shape
        self.split_size = split_size
        self.num_heads = num_heads
        # declared but not used in original implementation:
        #self.dim_out = dim_out or dim
        self.attn_drop_rate = attn_drop_rate
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        if idx == -1:
            self.H_sp, self.W_sp = self.resolution_h, self.resolution_w
        elif idx == 0:
            self.H_sp, self.W_sp = self.resolution_h, self.split_size
        elif idx == 1:
            self.W_sp, self.H_sp = self.resolution_w, self.split_size
        else:
            raise ValueError("ERROR MODE", idx)

        self.get_v = Conv2D(dim, kernel_size=3, padding="same", groups=1)
        self.attn_drop = Dropout(attn_drop_rate)
        
    def build(self, input_shape):
        self.get_v.build((None, self.H_sp, self.W_sp, input_shape[-1]))
        #self.get_v.build((input_shape[0] * (input_shape[1] // (self.H_sp * self.W_sp)), self.H_sp, self.W_sp, input_shape[-1]))
        self.attn_drop.build((None, self.num_heads, self.H_sp * self.W_sp, self.H_sp * self.W_sp))

        super().build(input_shape)
        
    def im2cswin(self, x):
        B, _, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        H, W = self.resolution_h, self.resolution_w
        
        # Reshape to [B, C, H, W]
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, [B, C, H, W])

        x = img2windows(x, self.H_sp, self.W_sp)
        num_patches = (H // self.H_sp) * (W // self.W_sp)
        x = tf.reshape(x, (B * num_patches, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        # Output shape x: (B', num_heads, H'*W', C // num_heads)
        return x 

    def get_lepe(self, x):
        """ x shaped (B, N, C) """
        B, _, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        H, W = self.resolution_h, self.resolution_w

        # Reshape to [B, C, H, W]
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, [B, C, H, W])
    
        # Splitting dimensions
        H_sp, W_sp = self.H_sp, self.W_sp
        num_patches = (H // H_sp) * (W // W_sp)
        x = tf.reshape(x, [B, C, H // H_sp, H_sp, W // W_sp, W_sp])

        # Permute and reshape to [-1 (B*H//Hsp*W//Wsp), C, H_sp, W_sp]
        # -> (B', C, H', W')
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 4, 1, 3, 5]), [B * num_patches, C, H_sp, W_sp])
        
        x_prep = tf.transpose(x, perm=[0, 2, 3, 1])
        # Apply the convolution and reshape results to match output shape
        lepe = self.get_v(x_prep)  
        # Reshape back
        lepe = tf.reshape(lepe, [B * num_patches, H_sp * W_sp, self.num_heads, C // self.num_heads])
        lepe = tf.transpose(lepe, perm=[0, 2, 1, 3])

        # Reshape x to match required shape and permute
        x = tf.reshape(x, [B * num_patches, self.num_heads, C // self.num_heads, H_sp * W_sp])
        x = tf.transpose(x, perm=[0, 1, 3, 2])

        return x, lepe
       
    def call(self, qkv):     
        q, k, v = qkv[0], qkv[1], qkv[2]
        B, _, C = tf.shape(q)[0], tf.shape(q)[1], tf.shape(q)[2]
        H, W = self.resolution_h, self.resolution_w

        q = self.im2cswin(q) 
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v)

        q *= self.scale # scale
        # Transpose for matrix multiplication
        k = tf.transpose(k, perm=[0, 1, 3, 2])
        attn = q @ k

        # Normalize -> attention weights
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        
        x = attn @ v
        
        # Add Positional Encoding
        x += lepe
        
        num_patches = (H // self.H_sp) * (W // self.W_sp)
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (B*num_patches, self.H_sp * self.W_sp, C))

        x = windows2img(x, self.H_sp, self.W_sp, H, W)

        x = tf.reshape(x, (B, H*W, C))

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim,
                       "image_shape": self.image_shape,
                       "idx": self.idx,
                       "num_heads": self.num_heads,
                       "split_size": self.split_size,
                       "attn_drop_rate": self.attn_drop_rate
                       })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# MLP Block
class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, drop_rate=0.0, **kwargs):
        super(MLPBlock, self).__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.drop_rate = drop_rate
        # Layers
        self.fc1 = Dense(self.hidden_features, activation="gelu")
        self.fc2 = Dense(in_features)
        self.drop1 = Dropout(drop_rate)
        self.drop2 = Dropout(drop_rate)

    def build(self, input_shape):
        self.fc1.build(input_shape)
        self.fc2.build((None, input_shape[1], self.hidden_features))
        self.drop1.build((None, input_shape[1], self.hidden_features))
        self.drop2.build(input_shape)

        super().build(input_shape)
        
    def call(self, x):
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"in_features": self.in_features,
                       "hidden_features": self.hidden_features,
                       "drop_rate": self.drop_rate
                       })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# CSWin Block
class CSWinBlock(tf.keras.layers.Layer):
    def __init__(self, dim, image_shape, num_heads, split_size=7, mlp_ratio=2, 
                 attn_drop_rate=0.01, drop_rate=0.01, last_stage=False, qkv_bias=True, 
                 qk_scale=None, **kwargs):
        super(CSWinBlock, self).__init__(**kwargs)
        self.dim = dim
        self.image_shape = image_shape
        self.resolution_h, self.resolution_w = image_shape
        self.num_heads = num_heads
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.attn_drop_rate=attn_drop_rate
        self.last_stage = last_stage
        self.qkv_bias = qkv_bias
        
        self.qkv = Dense(dim * 3, use_bias=qkv_bias)
        self.norm1 = LayerNormalization(epsilon=1e-3)
        
        # Determine branch_num based on last_stage flag
        self.branch_num = 1 if (last_stage or self.resolution_h == split_size \
                            or self.resolution_w == split_size) else 2
        
        self.proj = Dense(dim)
        # declared but not used in original implementation:
        #self.proj_drop = Dropout(drop) 
        
        # Initialize attention heads based on branch_num
        if self.branch_num == 1:
            # Last stage
            self.attns = [LePEAttention(dim, image_shape, idx=-1, 
                                        split_size=split_size, num_heads=num_heads,
                                        qk_scale=qk_scale, attn_drop_rate=attn_drop_rate)]
        else:
            # Attention for horizontal and vertical direction, each half the numbr of attentiion heads´
            self.attns = [LePEAttention(dim // 2, image_shape, idx, split_size=split_size, 
                                        num_heads=num_heads // 2, qk_scale=qk_scale, 
                                        attn_drop_rate=attn_drop_rate) for idx in range(2)]
        
        self.drop_rate = drop_rate
        self.dropout_p = Dropout(drop_rate) # instead of Droppath in original implementation
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(in_features=dim, hidden_features=mlp_hidden_dim, drop_rate=drop_rate)
        self.norm2 = LayerNormalization(epsilon=1e-3)

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.qkv.build(input_shape)
        if self.branch_num == 2:
            self.attns[0].build((3, None, input_shape[1], input_shape[-1] // 2))
            self.attns[1].build((3, None, input_shape[1], input_shape[-1] // 2))
        else:             
            self.attns[0].build((3, *input_shape))
        self.proj.build(input_shape)
        self.dropout_p.build(input_shape)
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)
        super().build(input_shape)
        
    def call(self, x):
        """ Receives x shaped (B, L, C). L is H*W. """
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        img = self.norm1(x)

        qkv = self.qkv(img)        
        qkv = tf.reshape(qkv, (B, L, 3, C)) 
        
        qkv = tf.transpose(qkv, perm=[2, 0, 1, 3])
        
        if self.branch_num == 2:

            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            atten_x = tf.concat([x1, x2], axis=2)
        else:
            atten_x = self.attns[0](qkv)
        
        atten_x = self.proj(atten_x)

        x = x + self.dropout_p(atten_x)
        x = x + self.mlp(self.norm2(x))

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim,
                       "image_shape": self.image_shape,
                       "num_heads": self.num_heads,
                       "split_size": self.split_size,
                       "mlp_ratio": self.mlp_ratio,
                       "attn_drop_rate": self.attn_drop_rate,
                       "drop_rate": self.drop_rate,
                       "last_stage": self.last_stage,
                       "qkv_bias": self.qkv_bias
                       })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class PatchMergingCS(layers.Layer):
    def __init__(self, dim, dim_out, image_shape, **kwargs):
        super(PatchMergingCS, self).__init__(**kwargs)
        self.dim = dim
        self.dim_out = dim_out
        self.conv = Conv2D(dim_out, kernel_size=3, strides=2, padding='same')
        self.norm = LayerNormalization(epsilon=1e-3)
        self.image_shape = image_shape
        
    def build(self, input_shape):
        self.conv.build((None, *self.image_shape, input_shape[-1]))
        self.norm.build((None, self.image_shape[0] // 2 * self.image_shape[1] // 2, self.dim_out))
        super().build(input_shape)
       
    def call(self, x):
        B, _, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        H, W = self.image_shape
        # Reshape to 2D image
        x = tf.reshape(x, [B, H, W, C])
        # Apply convolution
        x = self.conv(x)
        # Extract feature dimension, and new image dimensions
        C_new = tf.shape(x)[3]
        H_new, W_new = tf.shape(x)[1], tf.shape(x)[2]
        # Flatten dimensions 
        x = tf.reshape(x, [B, H_new * W_new, C_new])
        # Layer Normalization
        x = self.norm(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, 
                       "dim_out": self.dim_out,
                       "image_shape": self.image_shape
                       })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
###############################################################################
# !!! Build model
def build_CSWin(sensor_lst, input_dimensions, num_classes, num_heads=8, 
                projection_dim=96, split_size=4, attn_drop=0.0, drop=0.1,
                num_transformer_blocks=2, mlp_ratio=4, print_summary=True):
    """
    Builds multi-input CSWin Transformer for Classification.
    
    Parameters: 
    sensor_lst -- List of sensor names (strings) corresponding to the inputs.
    input_dimensions -- List of input shapes (tuples) for each sensor.
    num_classes -- Number of output classes for the classification task (int).
    num_heads -- Number of attention heads in the multi-head self-attention 
        mechanism (int or list of int). If a list, the number of heads is 
        specified per block. Defalts to 8.
    projection_dim -- Dimensionality of the projection space used for the 
        initial convolutional embedding and the Encoder block (int). Defaults to 96.
    split_size -- Size of the cross-shaped window for splitting input in CSWin 
        self-attention (int). Defaults to 4.
    attn_drop -- Dropout rate applied to the attention scores. Defaults to 0.0.
    drop -- Dropout rate applied to the feedforward layers in the CSWin block (float). 
        Defaults to 0.1
    num_transformer_blocks -- Number of Transformer block iterations for each 
        sensor's input. Defaults to 2.
    mlp_ratio -- Expansion ratio for the first feedforward layer (MLP) inside the 
        Transformer blocks (int). Defaults to 4.
    print_summary -- Boolean for printing keras model summary. Defaults to True.

    Returns: keras.Model
    """
    inputs = [Input(shape=input_dim, name=sensor) for input_dim, sensor in zip(input_dimensions, sensor_lst)]

    blocks = []
    for i in range(len(sensor_lst)):
        
        orig_image_shape = (32, 48) # Shape for resizing (not actually original shape)

        # Resize image and convolve into patches
        resized_images = Resizing(orig_image_shape[0], orig_image_shape[1])(inputs[i])
        encoded = Conv2D(filters=projection_dim, kernel_size=3, strides=2, padding="same")(resized_images)

        # Flatten height and width dimensions
        patches_h = encoded.shape[1]
        patches_w = encoded.shape[2]
        image_shape = (patches_h, patches_w)
        flatten_dim = patches_h * patches_w
        encoded = Reshape((flatten_dim, projection_dim))(encoded)
 
        # Last dimension / feature dimension
        block_projection_dim = projection_dim
        
        # Adjust num_heads to a list matched to the number of blocks
        if isinstance(num_heads, list):
            if len(num_heads) != num_transformer_blocks:
                raise ValueError("List for num_heads must have length matching \
                                 number of blocks, instead has length ", len(num_heads))
            else: heads = num_heads
        else: # scalar/int expected
            heads = [num_heads for j in range(num_transformer_blocks)]

        # Encoder
        x = encoded
        for j in range(num_transformer_blocks):
            last_stage = j == (num_transformer_blocks - 1)

            # Applying CSWinBlock with LePEAttention for Self-Attention
            x = CSWinBlock(dim=block_projection_dim, image_shape=image_shape, 
                           num_heads=heads[j], split_size=split_size, mlp_ratio=mlp_ratio,
                           attn_drop_rate=attn_drop, drop_rate=drop,
                           last_stage=last_stage)(x)

            # Patch Merging: downsample image and increase feature dimension
            if j < num_transformer_blocks - 1:  # No merging in the last block/stage
                x = PatchMergingCS(dim=block_projection_dim, 
                                                dim_out=block_projection_dim * 2, 
                                                image_shape=image_shape)(x)
                # Update block_projection_dim for subsequent blocks
                block_projection_dim *= 2
                # Calculate new image dimension considering a 2 strided 3x3 Conv with 
                H_new = math.ceil(image_shape[0] / 2)
                W_new = math.ceil(image_shape[1] / 2)
                image_shape = (H_new, W_new)
        
        x = GlobalAveragePooling1D()(x)
        blocks.append(x)

    # Concatenate across sensor outputs
    merged = concatenate(blocks, axis=-1) if len(blocks) > 1 else blocks[0]

    # Fully connected layers for classification
    dense_layer = Dense(2 * projection_dim, activation="relu", kernel_regularizer=l2(0.001))(merged)
    dense_layer = Dropout(0.1)(dense_layer)
    output = Dense(num_classes, activation="softmax")(dense_layer)
    
    model = Model(inputs=inputs, outputs=output)
    if print_summary: model.summary()
    return model


