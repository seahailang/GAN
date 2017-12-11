#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: GAN_utils.py
@time: 2017/12/11 10:12
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import abc

FLAGS = tf.app.flags.FLAGS

class BaseGAN(object):
    def __init__(self,generator,discriminator,sampler):
        self.batch_size=FLAGS.batch_size
        self.generator = generator
        self.discriminator = discriminator
        self.sampler = sampler

    def input(self):
        return self.discriminator.input()

    def sample(self):
        return self.sampler.sample()

    def build_graph(self,true_examples,fake_examples):
        fake_logits = self.discriminator(fake_examples)
        true_logits = self.discriminator(true_examples)
        return true_logits,fake_logits

    def compute_G_loss(self,logits,label):
        return tf.reduce_mean(-tf.log(1e-6+tf.sigmoid(logits)))
    def compute_real_D_loss(self,logits,label):
        return tf.reduce_mean(-tf.log(1e-6+tf.sigmoid(logits)))
    def compute_fake_D_loss(self,logits,label):
        return tf.reduce_mean(-tf.log(1e-6+1-tf.sigmoid(logits)))


    def train_op(self,g_loss,d_loss):
        d_opt = tf.train.AdamOptimizer()
        g_opt = tf.train.AdamOptimizer()
        var_list = tf.trainable_variables()
        d_var = []
        g_var = []
        for var in var_list:
            if var.name.startswith('D'):
                d_var.append(var)
            if var.name.startswith('G'):
                g_var.append(var)
        d_grads_and_vars = d_opt.compute_gradients(d_loss,d_var)
        g_grads_and_vars = g_opt.compute_gradients(g_loss,g_var)
        for g,v in g_grads_and_vars:
            tf.summary.histogram(v.name+'_grad',g)
            tf.summary.scalar(v.name+'_gradients',tf.reduce_sum(tf.abs(g)))
        for g,v in d_grads_and_vars:
            tf.summary.histogram(v.name+'_grad',g)
            tf.summary.scalar(v.name+'_gradients',tf.reduce_sum(tf.abs(g)))
        d_apply_op = d_opt.apply_gradients(d_grads_and_vars)
        g_apply_op = g_opt.apply_gradients(g_grads_and_vars)
        return d_apply_op,g_apply_op

class BaseDiscriminator(object):
    def __init__(self,batch_size):
        self.batch_size = batch_size

    @abc.abstractmethod
    def input(self):
        return

    @abc.abstractmethod
    def build_graph(self,tensor):
        return

    def __call__(self,inputs):
        with tf.variable_scope('Discriminator',reuse=tf.AUTO_REUSE):
            result = self.build_graph(inputs)
        return result

class BaseGenerator(object):
    def __init__(self,batch_size):
        self.batch_size = batch_size
    @abc.abstractmethod
    def build_graph(self,tensor):
        return
    def __call__(self,z):
        with tf.variable_scope('Generator'):
            result = self.build_graph(z)
        return result

class BaseSampler(object):
    def __init__(self,seed,batch_size,latent_dims):
        tf.set_random_seed(seed)
        self.batch_size = batch_size
        self.latent_dims = latent_dims

    @abc.abstractmethod
    def sample(self):
        return




if __name__ == '__main__':
    pass