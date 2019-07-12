#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:32:40 2019

@author: Purnendu Mishra
"""
import tensorflow as tf
from keras.models import Model

from keras.utils import plot_model

from keras.layers import Input
from keras import backend as K
import numpy as np

from resnet_common import ResNet50, ResNet101, ResNet50V2, ResNet101V2

if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CH_AXIS  = 3


def ResNet_50(input_shape = (None,None,3), OS = 16):
    input_tensor = Input(shape = input_shape, name='input')
    backbone_model = ResNet50(input_tensor = input_tensor, include_top = False, weights='imagenet')

    config = backbone_model.get_config()
    
    for i, layer in enumerate(backbone_model.layers):
        keys = config['layers'][i]['config']
        
        if 'activation' in keys['name']:
            keys['name'] = 'activation_{:02d}'.format(i)
        
        dims = layer.output_shape
        
        output_stride_0 = int(np.ceil(input_shape[ROW_AXIS - 1]) /  dims[ROW_AXIS])
        output_stride_1 = int(np.ceil(input_shape[COL_AXIS - 1]) / dims[COL_AXIS])
        
        if OS == 16:
            if output_stride_0 >= 32 and output_stride_1 >= 32:
                if 'padding' in keys:    
                    if keys['padding'] == 'valid':
                        keys['padding'] = 'same'
    
                if 'strides' in keys:
                    if keys['strides'] == (2,2):
                        keys['strides'] = (1,1)
                if 'dilation_rate' in keys:
                    keys['dilation_rate'] = 2
                    
        elif OS == 8:
            if output_stride_0 >= 16 and output_stride_1 >= 16:

                if 'strides' in keys:
                    if keys['strides'] == (2,2):
                        keys['strides'] = (1,1)
                if 'dilation_rate' in keys:
                    keys['dilation_rate'] = 4
                
                if 'padding' in keys:    
                    if keys['padding'] == 'valid':
                        keys['padding'] = 'same'
    
    return Model.from_config(config)

def ResNet_50V2(input_shape = (None,None,3), OS = 16):
    input_tensor = Input(shape = input_shape, name='input')
    backbone_model = ResNet50V2(input_tensor = input_tensor, include_top = False, weights='imagenet')

    config = backbone_model.get_config()
    
    for i, layer in enumerate(backbone_model.layers):
        keys = config['layers'][i]['config']
        
        if 'activation' in keys['name']:
            keys['name'] = 'activation_{:02d}'.format(i)
        
        dims = layer.output_shape
        
        output_stride_0 = int(np.ceil(input_shape[ROW_AXIS - 1]) /  dims[ROW_AXIS])
        output_stride_1 = int(np.ceil(input_shape[COL_AXIS - 1]) / dims[COL_AXIS])
        
        if OS == 16:
            if output_stride_0 >= 32 and output_stride_1 >= 32:
                
                if 'padding' in keys:    
                    if isinstance(keys['padding'], tuple):
                        keys['padding'] = 0
                    elif isinstance(keys['padding'], str):
                        keys['padding'] = 'same'
                    
    
                if 'strides' in keys:
                    if keys['strides'] == (2,2):
                        keys['strides'] = (1,1)
                if 'dilation_rate' in keys:
                    keys['dilation_rate'] = 2
                    
        elif OS == 8:
            if output_stride_0 >= 16 and output_stride_1 >= 16:

                if 'strides' in keys:
                    if keys['strides'] == (2,2):
                        keys['strides'] = (1,1)
                if 'dilation_rate' in keys:
                    keys['dilation_rate'] = 4
                
                if 'padding' in keys:    
                    if isinstance(keys['padding'], tuple):
                        keys['padding'] = 0
                    elif isinstance(keys['padding'], str):
                        keys['padding'] = 'same'
    
    return Model.from_config(config)


def ResNet_101(input_shape = (None, None, 3), OS = 16):
    input_tensor = Input(shape = input_shape, name='input')
    backbone_model = ResNet101(input_tensor = input_tensor, include_top = False, weights='imagenet')
    
    config = backbone_model.get_config()
    
    for i, layers in enumerate(backbone_model.layers):
        
        keys = config['layers'][i]['config']
        dims = layers.output_shape
        
#        print(i, keys)
        
        if 'activation' in keys['name']:
            keys['name'] = 'activation_{:02d}'.format(i)

        
        output_stride_0 = int(np.ceil(input_shape[ROW_AXIS - 1])/ dims[ROW_AXIS])
        output_stride_1 = int(np.ceil(input_shape[COL_AXIS - 1])/ dims[COL_AXIS])
        
        if OS == 16:
            if output_stride_0 >= 32 and output_stride_1 >= 32:
                if 'padding' in keys:    
                    if keys['padding'] == 'valid':
                        keys['padding'] = 'same'
    
                if 'strides' in keys:
                    if keys['strides'] == (2,2):
                        keys['strides'] = (1,1)
                if 'dilation_rate' in keys:
                    keys['dilation_rate'] = 2
                    
        elif OS == 8:
            if output_stride_0 >= 16 and output_stride_1 >= 16:

                if 'strides' in keys:
                    if keys['strides'] == (2,2):
                        keys['strides'] = (1,1)
                if 'dilation_rate' in keys:
                    keys['dilation_rate'] = 4
                
                if 'padding' in keys:    
                    if keys['padding'] == 'valid':
                        keys['padding'] = 'same'
        
 
    return Model.from_config(config)


def ResNet_101V2(input_shape = (None, None, 3), OS = 16):
    input_tensor = Input(shape = input_shape, name='input')
    backbone_model = ResNet101V2(input_tensor = input_tensor, include_top = False, weights='imagenet')
    
    config = backbone_model.get_config()
    
    for i, layers in enumerate(backbone_model.layers):
        
        keys = config['layers'][i]['config']
        dims = layers.output_shape
        
#        print(i, keys)
        
        if 'activation' in keys['name']:
            keys['name'] = 'activation_{:02d}'.format(i)

        
        output_stride_0 = int(np.ceil(input_shape[ROW_AXIS - 1])/ dims[ROW_AXIS])
        output_stride_1 = int(np.ceil(input_shape[COL_AXIS - 1])/ dims[COL_AXIS])
        
        if OS == 16:
            if output_stride_0 >= 32 and output_stride_1 >= 32:
                if 'padding' in keys:    
                    if keys['padding'] == 'valid':
                        keys['padding'] = 'same'
    
                if 'strides' in keys:
                    if keys['strides'] == (2,2):
                        keys['strides'] = (1,1)
                if 'dilation_rate' in keys:
                    keys['dilation_rate'] = 2
                    
        elif OS == 8:
            if output_stride_0 >= 16 and output_stride_1 >= 16:

                if 'strides' in keys:
                    if keys['strides'] == (2,2):
                        keys['strides'] = (1,1)
                if 'dilation_rate' in keys:
                    keys['dilation_rate'] = 4
                
                if 'padding' in keys:    
                    if keys['padding'] == 'valid':
                        keys['padding'] = 'same'
        
 
    return Model.from_config(config)


if __name__ == "__main__":
    with tf.device('/gpu:0'):
        output_stride = 16
        model = ResNet_50V2(input_shape = (512, 512, 3), OS = output_stride)
        plot_model(model, to_file = 'model_images/Resnet_101_OS-{}.png'.format(output_stride), 
                   show_shapes=True, show_layer_names=True)
        model.summary()