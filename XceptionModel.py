#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:21:50 2019

@author: Purnendu Mishra

"""

import tensorflow as tf

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, SeparableConv2D, Concatenate, Add
from keras.layers import BatchNormalization, GlobalAveragePooling2D, Activation
from keras.initializers import he_normal
from keras.regularizers import l2

from keras import backend as K
from keras.utils import plot_model
from utility import sep_conv_bn_relu, conv_bn_relu, _bn_relu

if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3


def _shortcut(input_, residual, name=None):
    """Add a shorcut between input and residula block and merges then with Add
    """
    
    input_shape = K.int_shape(input_)
    residual_shape = K.int_shape(residual)
    
    stride_width = int(round(input_shape[ROW_AXIS] /residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS]/residual_shape[COL_AXIS]))
    
    is_equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]
    
    shorcut = input_
    
    if stride_width > 1 or stride_height > 1 or not is_equal_channels:
        shorcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                         kernel_size=(1,1),
                         strides=(stride_width, stride_height),
                         padding='same',
                         kernel_initializer=he_normal(),
                         kernel_regularizer=l2(1e-4),
                         name='shortcut_{}'.format(name))(input_)
        
    return Add(name='Add_{}'.format(name))([shorcut, residual])

def basic_block(filters, init_strides=(1,1), name=None):
    def f(input_):
        residual = sep_conv_bn_relu(filters=filters,strides=init_strides, name=name)(input_)
        return residual
    return f


def _residual_block(block_function,filters, repetition=3, increase_stride=False, name=None):
   def f(input_):
       residual = input_
       for i in range(repetition):
           strides = (1,1)
           if i==(repetition - 1) and increase_stride:
               strides=(2,2)
               
           residual = block_function(filters=filters, init_strides=strides, name='{0}_{1:02d}'.format(name, i+1))(residual)
       return _shortcut(input_, residual, name=name)
   return f



def entry_residual_block(block_function):
    def f(input_):
        x = _residual_block(block_function, filters=128, repetition=3, increase_stride=True, name='entry_block-A')(input_)
        x = _residual_block(block_function, filters=256, increase_stride=True, name='entry_block-B')(x)
        x = _residual_block(block_function, filters=728, increase_stride=True, name='entry_block-C')(x)
        return x
    return f

def middle_residual_block(block_function, reptition=16):
    def f(input_):
        for i in range(reptition):
            input_ = _residual_block(block_function, filters=728, increase_stride=False, name='middle_block_{:02d}'.format(i+1))(input_)
        return input_
    return f

def exit_block(block_function, final_block_stride=(1,1), rate=2):
    def f(input_):
        x = block_function(filters=728, name='exit_res_01')(input_)
        x = block_function(filters=1024, name='exit_res_02')(x)
        x = block_function(filters=1024,init_strides=final_block_stride, name='exit_res_03')(x)
        x = _shortcut(input_, x, name='exit_block')
        
        if final_block_stride[0] == 1:
            x = sep_conv_bn_relu(filters=1536, dilation_rate=2, name='exit_A')(x)
            x = sep_conv_bn_relu(filters=1536, dilation_rate=2, name='exit_B')(x)
            x = sep_conv_bn_relu(filters=2048, dilation_rate=2, name='exit_c')(x)
        else:
            x = block_function(filters=1536, name='exit_A')(x)
            x = block_function(filters=1536, name='exit_B')(x)
            x = block_function(filters=2048, name='exit_C')(x)
        return x
    return f
    

def Xception(input_shape=(None,None,3)):
    input_ = Input(shape=input_shape, name='input_layer')
    x = conv_bn_relu(filters=32, kernel_size=(3,3),strides=(2,2), name='input_A')(input_)
    x = conv_bn_relu(filters=64, kernel_size=(3,3),name='input_B')(x)
    
    # The entry block
    x = entry_residual_block(block_function=basic_block)(x)
    x = middle_residual_block(block_function=basic_block, reptition=16)(x)
    x = exit_block(block_function=basic_block, final_block_stride=(1,1), rate=2)(x)
    
    return Model(inputs = input_, outputs=x)


if __name__ == '__main__':
    model = Xception(input_shape=(512,512,3))
    plot_model(model=model, to_file='Xception_deepLabV3.png', show_shapes=True, show_layer_names=True)
    model.summary()