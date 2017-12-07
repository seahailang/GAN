#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: DG.py
@time: 2017/12/4 19:10
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from config import FLAGS
import utils

class Discriminator(object):
    def __init__(self):
        # self.batch_size = FLAGS.batch_size
        self.batch_size = int(FLAGS.batch_size/2)
        self.image_w = FLAGS.image_w
        self.image_h = FLAGS.image_h
        self.channel = FLAGS.channel
        self.linear_units = [128,500]

    def input(self):
        inputs = tf.placeholder(dtype=tf.float32,shape=(self.batch_size,self.image_w*self.image_h),name='inputs')
        image = tf.reshape(inputs,shape=(self.batch_size,self.image_w,self.image_h,self.channel))
        tf.summary.image('input_image',image)
        return inputs
    def build_graph(self,tensor):
        with tf.variable_scope('D_linear1',reuse=tf.AUTO_REUSE):
            tensor = utils.linear_layer(tensor,self.linear_units[0])
        # with tf.variable_scope('D_linear2',reuse=tf.AUTO_REUSE):
        #     tensor = utils.linear_layer(tensor,self.linear_units[1])
        # with tf.variable_scope('D_soft_max',reuse=tf.AUTO_REUSE):
        #     logits = utils.linear_layer(tensor,1,activate=tf.nn.sigmoid)
        # tensor = tf.reshape(tensor,[tensor.shape[0],FLAGS.image_w,FLAGS.image_h,1])
        # with tf.variable_scope('conv1'):
        #     tensor = utils.conv_layer(tensor,64,[3,3],1)
        # with tf.variable_scope('conv2'):
        #     tensor = utils.conv_layer(tensor,64,[3,3],1)
        # tensor = tf.reshape(tensor,[tensor.shape[0],-1])
        with tf.variable_scope('linear'):
            logits = utils.linear_layer(tensor,1,lambda x:x)
        return logits
    def __call__(self,inputs):
        with tf.variable_scope('Discriminator',reuse=tf.AUTO_REUSE):
            result = self.build_graph(inputs)
        return result



class Generator(object):
    def __init__(self):
        self.batch_size = int(FLAGS.batch_size/2)
        self.image_w = FLAGS.image_w
        self.image_h = FLAGS.image_h
        self.channel = FLAGS.channel
        self.linear_units = [128, 500,500]
    def build_graph(self,tensor):
        with tf.variable_scope('G_linear1'):
            tensor = utils.linear_layer(tensor,self.linear_units[0])
            # tensor = tf.reshape(tensor,[self.batch_size,7,7,1])
        # with tf.variable_scope('G_deconv_1'):
        #     tensor = utils.de_conv_layer(tensor,32,[3,3],2)
        # with tf.variable_scope('G_deconv_2'):
        #     tensor = utils.de_conv_layer(tensor,1,[3,3],2,activate=tf.nn.sigmoid)

        # with tf.variable_scope('G_linear2'):
        #     tensor = utils.linear_layer(tensor,self.linear_units[1])
        with tf.variable_scope('G_linear3'):
            tensor = utils.linear_layer(tensor,self.image_w*self.image_h*self.channel,activate=tf.nn.sigmoid)
        image = tf.reshape(tensor, shape=(self.batch_size, self.image_w, self.image_h, self.channel))
        tf.summary.image('output_image', image)
        tensor = tf.reshape(tensor,[tensor.shape[0],-1])
        return tensor
    def __call__(self,z):
        with tf.variable_scope('Generator'):
            result = self.build_graph(z)
        return result

class Sampler(object):
    def __init__(self,seed):
        tf.set_random_seed(seed)
        self.batch_size = int(FLAGS.batch_size/2)
        self.latent_dims = FLAGS.latent_dims
    def sample(self):
        z = tf.random_uniform(shape=[self.batch_size,self.latent_dims],minval=-1,maxval=1)
        return z

if __name__ == '__main__':
    pass