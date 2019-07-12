#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:40:59 2019

@author: Purnendu Mishra
"""

from keras.models import Model
from keras.applications import VGG16
from keras.layers import Input
from keras import backend as K
import numpy as np

if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CH_AXIS  = 3


def VGG16Net(input_shape=(None,None,3), OS = 16):
    input_tensor = Input(shape = input_shape)
    backbone_model = VGG16(input_tensor =  input_tensor, include_top=False, weights='imagenet')
    
    config = backbone_model.get_config()
    for i, layer in enumerate(backbone_model.layers):
        dims = layer.output_shape
            
        output_stride_0 = int(np.ceil(input_shape[ROW_AXIS - 1]/dims[ROW_AXIS]))
        output_stride_1 = int(np.ceil(input_shape[COL_AXIS - 1]/dims[COL_AXIS]))
        
        if OS == 16:
            # Make pooling stride of last pooling layer equal to One
            keys = config['layers'][-1]['config']
            keys['strides'] = (1,1)
            keys['padding'] = 'same'
        
        elif OS == 8:
            keys = config['layers'][i]['config']
            
            if output_stride_0 >= 16 or output_stride_1 >= 16:
                
                if 'strides' in keys:
                    keys['strides'] = 1
                    
                if 'dilation_rate' in keys:
                    keys['dilation_rate'] = 4
                    
                if keys['padding'] == 'valid':
                    keys['padding'] = 'same'
                    
                
                
    return Model.from_config(config)


if __name__ == '__main__':
    model = VGG16Net(input_shape=(512,512,3), OS = 8)
    model.summary()
