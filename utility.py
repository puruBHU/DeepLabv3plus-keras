#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:10:11 2019

@author: Purnendu Mishra
"""
from keras.layers import Conv2D, SeparableConv2D, Activation
from keras.layers import BatchNormalization
from keras.initializers import he_normal
from keras.regularizers import l2

def _bn_relu(input_):
    norm = BatchNormalization()(input_)
    return Activation('relu')(norm)

def conv_bn_relu(**params):
    filters     = params['filters']
    kernel_size = params['kernel_size']
    strides     = params.setdefault('strides',(1,1))
    padding     = params.setdefault('padding','same')
    dilation_rate = params.setdefault('dilation_rate', 1)
    kernel_initializer = params.setdefault('kernel_initializer', he_normal())
    kernel_regularizer = params.setdefault('kernel_regularizer', l2(1e-3))
    activation         = params.setdefault('activation','relu')
    name               = params.setdefault('name',None)

    if not name == None:
        conv_name = 'conv_{}'.format(name)
        bn_name   = 'BN_{}'.format(name) 
        act_name  = 'Act_{}_{}'.format(name, activation)

    def f(input_):
        conv = Conv2D(filters       = filters,
                      kernel_size   = kernel_size,
                      strides       = strides,
                      padding       = padding,
                      dilation_rate = dilation_rate,
                      kernel_initializer = kernel_initializer,
                      kernel_regularizer = kernel_regularizer,
                      name=conv_name)(input_)

        batch_norm = BatchNormalization(name=bn_name)(conv)

        return Activation(activation,name=act_name)(batch_norm)
    return f


def sep_conv_bn_relu(**params):
    filters     = params['filters']
    kernel_size = params.setdefault('kernel_size',(3,3))
    strides     = params.setdefault('strides',(1,1))
    padding     = params.setdefault('padding','same')
    dilation_rate = params.setdefault('dilation_rate', 1)
    depthwise_initializer = params.setdefault('depthwise_initializer', he_normal())
    pointwise_initializer = params.setdefault('pointwise_initializer', he_normal())

    depthwise_regularizer = params.setdefault('depthwise_regularizer', l2(1e-3))
    pointwise_regularizer = params.setdefault('pointwise_regularizer', l2(1e-3))
    activation         = params.setdefault('activation','relu')
    name               = params.setdefault('name',None)

    if not name == None:
        conv_name = 'conv_{}'.format(name)
        bn_name   = 'BN_{}'.format(name) 
        act_name  = 'Act_{}_{}'.format(name, activation)

    def f(input_):
        conv = SeparableConv2D(filters       = filters,
                              kernel_size   = kernel_size,
                              strides       = strides,
                              padding       = padding,
                              dilation_rate = dilation_rate,
                              depthwise_initializer=depthwise_initializer,
                              pointwise_initializer=pointwise_initializer,
                              depthwise_regularizer=depthwise_regularizer,
                              pointwise_regularizer=pointwise_regularizer,
                              name=conv_name
                              )(input_)

        batch_norm = BatchNormalization(name=bn_name)(conv)

        return Activation(activation, name=act_name)(batch_norm)
    return f
