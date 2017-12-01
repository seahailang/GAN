#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: discriminator.py
@time: 2017/11/24 14:07
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.path.dirname(__file__))
import tensorflow as tf
# from tensorflow import keras
from datetime import datetime
from utils import *

FLAGS = tf.app.flags.FLAGS


class Discriminator(object):
    def __init__(self):
        self.conv_params = []
    def build_graph(self,tensor):
        for conv_param in self.conv_params:
            tensor = self._build_cnn(self,tensor,**conv_param)
        tensor = tf.reshape(FLAGS)
        for full_param in self.full_params:
            tensor = self._build_full_connected(tensor,**full_param)
    def __call__(self):
        pass
    def disc(self,input):
        pass
    def _build_cnn(self,input,scope=None,name='CNN_layer',
                   filters=6,kernel_w=3,kernel_l=3,strides=1,padding='SAME'):
        with tf.name_scope(scope) as scope:
            kernel_size = [kernel_w,kernel_l]
            cov = Conv2d(filters=filters,
                         kernel_size=kernel_size,
                         strides= strides,
                         padding=padding,
                         activation=tf.nn.relu6,
                         name=name)
            output = cov(input)
        return output
    def _build_full_connected(self,input,score=None):


if __name__ == '__main__':
    pass