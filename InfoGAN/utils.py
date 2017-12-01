#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: utils.py
@time: 2017/11/24 14:23
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
class Con2d(object):
    def __init__(self,filters,
                 kernel_size,
                 strides,
                 padding='SAME',
                 bias_initializer=tf.random_uniform_initializer(),
                 activation=tf.nn.relu6,
                 use_bias=True,
                 name='con2d'):
        self.filters = filters
        self.kernel_size= kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        # self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        # self.trainable = trainable
        self.name = name

    def __call__(self,input):
        input_dims = tf.shape(input)[-1]
        kernel = [input_dims,self.kernel_size[0],self.kernel_size[1],self.filters]
        strides = [1,self.strides,self.strides,1]
        cov = tf.nn.conv2d(input=input,filter=kernel,strides=strides,padding=self.padding,name=self.name)
        if self.use_bias:
            bias = tf.get_variable(name=self.name+'/bias',shape=self.filters,dtype=tf.float32,initializer=self.bias_initializer)
            cov = tf.nn.bias_add(cov,bias)
        cov = self.activation(cov)
        return cov






if __name__ == '__main__':
    pass