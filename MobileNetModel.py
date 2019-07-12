#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:01:47 2019

@author: Purnendu Mishra
"""

from keras.models import Model, Sequential
from keras.layers import Input
from keras.applications import MobileNetV2
from keras.utils import plot_model
from keras import backend as K
import numpy as np

if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CH_AXIS  = 3

def MobileNetV2_modified(input_shape=(None,None,3), OS = 16):
    input_tensor = Input(shape = input_shape)
    backbone_model = MobileNetV2(input_tensor =  input_tensor, include_top=False, weights='imagenet')
    
    
    # To configure the model as used in DeepLab model
    
    config = backbone_model.get_config()
    for i, layers in enumerate(backbone_model.layers):
        
        dims = layers.output_shape
        
        
        output_stride_0 = int(np.ceil(input_shape[ROW_AXIS - 1] / dims[ROW_AXIS]))
        output_stride_1 = int(np.ceil(input_shape[COL_AXIS - 1] / dims[COL_AXIS]))
        
        keys = config['layers'][i]['config']
#       
        if OS == 16:
            if keys['name'] == 'block_13_pad':
                keys['padding'] = 0
            
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
#            if keys['name']  == 'block_6_depthwise':
#                keys['padding'] = 'same'
            if keys['name'] == 'block_6_pad' or keys['name'] == 'block_13_pad':
                keys['padding'] = 0
                   
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


if __name__ == '__main__':
    input_shape = (512,512,3)
    output_stride = 16
    model = MobileNetV2_modified(input_shape=(512,512,3), OS=output_stride)
    model.summary()
#    plot_model(model, to_file='Reconfigured_Mobilenetv2.png', show_shapes=True)
        
    

