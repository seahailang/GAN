#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: mnist_utils.py
@time: 2017/12/13 10:48
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
path = os.path.dirname(__file__)

FLAGS = tf.app.flags.FLAGS
class MnistModel(object):
    def __init__(self):
        self.image_w = 28
        self.image_h = 28
        self.channel = 1
        self.cat_num = 10
        self.example_shape=28*28*1

mnist = input_data.read_data_sets(path+"/MNIST_data/", one_hot=True)

if __name__ == '__main__':
    pass