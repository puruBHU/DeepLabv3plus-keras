#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:59:48 2019

@author: Purnendu Mishra
"""

from keras import backend as K

def pixel_accuracy(y_true, y_pred):
    
    output = K.cast(K.argmax(y_pred, axis=-1),dtype='float32')
    
    target = K.squeeze(y_true, axis=-1)
    target = K.cast(target, dtype='float32')
   
    
    equal_output = K.cast(K.equal(target, output), dtype='float32')
    total_equal  = K.sum(equal_output)
    
    not_equal_output = K.cast(K.not_equal(target, output), dtype='float32')
    not_total_equal  = K.sum(not_equal_output)
    

    accuracy = (total_equal / (total_equal +  not_total_equal))
    return accuracy

def pix_acc_th(threshold = 0.95):
    ''' Calculates pixel-wise accuracy for skin segmentation
        To be used when model has only one output channel
        Args:
            Threshold: If pixel value of prediction is greater than threshold,
                        assign it value of 1.
    '''
    def pixel_accuracy(y_true, y_pred):
        
        target = K.squeeze(y_true, axis=-1)
        target = K.cast(target, dtype='float32')
        
        output = K.squeeze(y_pred, axis=-1)
        
        # Pixel value greater than greater tha threshold should be One
        is_greater = K.greater_equal(output, threshold)
        predicted_value = K.cast(is_greater, dtype='float32')
        
        equal_output = K.cast(K.equal(predicted_value, target), dtype='float32')
        total_equal  = K.sum(equal_output)
        
        not_equal_output = K.cast(K.not_equal(predicted_value, target), dtype='float32')
        total_not_equal = K.sum(not_equal_output)
        
        accuracy = (total_equal / (total_equal +  total_not_equal))
        
        return accuracy
    return pixel_accuracy

def mean_iou(threshold, smooth=1.0):
    def MeanIOU(y_true, y_pred):
        target = K.squeeze(y_true, axis=-1)
        target = K.cast(target, dtype='float32')
        
        output = K.squeeze(y_pred, axis=-1)
        output = K.cast(K.greater_equal(output, threshold), dtype='float32')
        
        intersection = K.sum(target * output)
        
        target_true = K.sum(K.cast(K.equal(target, 1.0), dtype='float32'))
        output_true = K.sum(K.cast(K.equal(output, 1.0), dtype='float32'))
        
        union = target_true + output_true -  intersection
    
        IoU = (intersection + smooth) / (union + smooth) 
       
        return IoU
    return MeanIOU

def MeanIOU(y_true, y_pred):
    smooth = 1
    
    target = K.squeeze(y_true, axis=-1)
    target = K.cast(target, dtype='float32')
    
    output = K.cast(K.argmax(y_pred, axis=-1),dtype='float32')

    intersection = K.sum(target * output)
    
    target_true = K.sum(K.cast(K.equal(target, 1.0), dtype='float32'))
    output_true = K.sum(K.cast(K.equal(output, 1.0), dtype='float32'))
    
    union = target_true + output_true -  intersection
    
    IoU = (intersection + smooth) / (union + smooth) 
    return IoU



def CustomLoss(y_true, y_pred):
    target = K.squeeze(y_true, axis=-1)
    target = K.cast(target, dtype='float32')
    
    output = K.squeeze(y_pred, axis=-1)
    output = K.cast(output, dtype='float32')
    
    x  = (target - output)
    
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    
    return K.mean(_logcosh(x), axis=(0,1,2))


