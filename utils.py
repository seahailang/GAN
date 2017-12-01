#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: utils.py
@time: 2017/11/30 13:25
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime

def conv_layer(tensor,filters,k_size,strides,bias=True,activate=tf.nn.relu,name=''):
    filter_shape = [k_size[0],k_size[1], tensor.shape[-1], filters]
    weights = tf.get_variable(name='weight'+name,
                              shape=filter_shape,
                              dtype=tf.float32,
                              initializer=tf.random_normal_initializer(-0.06, 0.06))
    strides = [1, strides, strides, 1]
    conv = tf.nn.conv2d(tensor, filter=weights, strides=strides, padding="SAME")
    if bias:
        bias = tf.get_variable('bias'+name,
                               shape=filters,
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        conv = activate(tf.nn.bias_add(conv, bias))
    else:
        conv = activate(conv)
    return conv

def linear_layer(tensor,units,activate=tf.nn.relu,weights_initializer=None,name=''):
    if not weights_initializer:
        weights_initializer = tf.random_uniform_initializer(-0.006, 0.006)
    weights = tf.get_variable(name=name+'weights',
                              shape=[tensor.shape[-1],units],
                              dtype=tf.float32,
                              initializer=weights_initializer)
    # if not bias_initializer:
    #     bias_initializer = tf.random_uniform_initializer(-0.06, 0.06)
    bias = tf.get_variable(name=name+'bias',
                           shape=[units],
                           dtype=tf.float32,
                           initializer=tf.zeros_initializer())
    tensor = activate((tf.nn.bias_add(tf.matmul(tensor, weights), bias)))
    return tensor

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    pass