#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:46:53 2019

@author: Purnendu Mishra

DeepLab V3+ model 
"""
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, SeparableConv2D, UpSampling2D, Activation
from keras.layers import BatchNormalization, GlobalAveragePooling2D, Reshape, Concatenate
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.layers import Input

from keras import backend as K

# IMPORT BACKBONE MODELS

from ResNetModel import ResNet_50, ResNet_101
from VGGModel import VGG16Net
from XceptionModel import Xception 
from MobileNetModel import MobileNetV2_modified

from utility import sep_conv_bn_relu, conv_bn_relu
from keras.utils import plot_model


if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3

def ASPP():
    ''' Function to to perfrom atrous pyramid pooling
    '''
    def f(tensor):
        # Get shape of the final feature layer of the backbone network
        h, w, c = K.int_shape(tensor)[1:]
        
        global_avg_pool = GlobalAveragePooling2D(name='global_avg_pool')(tensor)
        
        # Get the number of output channels from the previous layer
        c = K.int_shape(global_avg_pool)[-1]
    
        image_level_feature = Reshape((1,1,c), name='reshape')(global_avg_pool)
        image_level_feature = conv_bn_relu(filters=256, kernel_size=(1,1),name='bottleneck_GAP')(image_level_feature)
        image_level_feature = UpSampling2D(size=(h,w),interpolation='bilinear')(image_level_feature)
        
        aspp_conv_01 = conv_bn_relu(filters=256, kernel_size=(1,1),dilation_rate=1, name = 'conv01_r1')(tensor)
        aspp_conv_02 = sep_conv_bn_relu(filters=256, kernel_size=(3,3),dilation_rate=6, name = 'atrous01_r6')(tensor)
        aspp_conv_03 = sep_conv_bn_relu(filters=256, kernel_size=(3,3),dilation_rate=12, name = 'atrous02_r12')(tensor)
        aspp_conv_04 = sep_conv_bn_relu(filters=256, kernel_size=(3,3),dilation_rate=18, name = 'atrous03_r18')(tensor)
        
        x = Concatenate()([aspp_conv_01, aspp_conv_02, aspp_conv_03, aspp_conv_04, image_level_feature ])
        
        return x
    return f

    
def DeepLabV3plus(backbone='vgg16',OS=16, shape=(None,None,3), num_classes=2):
    
    if backbone == 'vgg16':
        backbone_model = VGG16Net(input_shape = shape, OS = OS)
        high_level_feature_layer = backbone_model.layers[-1].output
        low_level_feature_layer  = backbone_model.get_layer('block3_conv3').output

            
            
    elif backbone == 'xception':
        backbone_model = Xception(input_shape=shape)
        if OS == 16:
           low_level_feature_layer  = backbone_model.get_layer('Add_entry_block-A').output
           high_level_feature_layer = backbone_model.layers[-1].output
           
    elif backbone == 'mobilenetv2':
        
        backbone_model = MobileNetV2_modified(input_shape=shape, OS = OS)
        
        high_level_feature_layer = backbone_model.layers[-1].output
        low_level_feature_layer =  backbone_model.get_layer('block_3_expand_relu').output
        
    elif backbone == 'resnet101':
        backbone_model = ResNet_101(input_shape = shape, OS = OS)
        backbone_model.load_weights(resnet101_weight_path, by_name=True)
        
        high_level_feature_layer = backbone_model.layers[-1].output
        low_level_feature_layer  = backbone_model.get_layer('conv2_block3_out').output
    
    elif backbone == 'resnet50':
        backbone_model = ResNet_50(input_shape = shape, OS = OS)
        
        high_level_feature_layer = backbone_model.layers[-1].output
        low_level_feature_layer  = backbone_model.get_layer('conv2_block3_out').output
        
        
    else:
        raise ValueError('Implementation for backbone "{}" is not present. Please choose from "vgg16, xception, mobilenetv2,' \
                         ' resnet50 and resnet101"'.format(backbone))
    
    
    x = ASPP()(high_level_feature_layer)
    
    x = conv_bn_relu(filters=256, kernel_size=(1,1), name='feature_reduce_high_level')(x)
    
    if OS == 16:
        x = UpSampling2D(size=(4,4), interpolation='bilinear', name='upsample_4x_first')(x)
    elif OS == 8:
        x = UpSampling2D(size=(2,2), interpolation='bilinear', name='upsample_2x_first')(x)
        
    low_level_features = conv_bn_relu(filters=48, kernel_size=(1,1), name='bottleneck_low_level_features')(low_level_feature_layer)
    
    x = Concatenate()([x, low_level_features])
    x = sep_conv_bn_relu(filters=256, kernel_size=(3,3), name='sep_conv_last')(x)
    
    x = UpSampling2D(size=(4,4), interpolation='bilinear', name='upsample_4x_second')(x)

    x = Conv2D(filters=num_classes, kernel_size=(1,1), padding='same', name='logit')(x)
    x = Activation('softmax')(x)
    
    return Model(inputs=backbone_model.input, outputs=x)


  
if __name__ == '__main__':
    
    with tf.device('/gpu:0'):
        backbone = 'resnet50'
        output_stride = 16
        
        model = DeepLabV3plus(backbone = backbone, 
                              shape=(512,512,3), 
                              num_classes=2, 
                              OS = output_stride)
        
        plot_model(model=model, to_file='model_images/DeepLabV3plus_{}.png'.format(backbone), 
                   show_shapes=True, 
                   show_layer_names=True)
        
        model.summary()
