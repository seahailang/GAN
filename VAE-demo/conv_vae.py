#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: vae.py
@time: 2017/11/29 11:08
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from utils import linear_layer,conv_layer

from datetime import datetime
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_w',28,
                            '''image width''')
tf.app.flags.DEFINE_integer('image_h',28,
                            '''image height''')
tf.app.flags.DEFINE_integer('channel',1,
                            '''image channel''')
tf.app.flags.DEFINE_integer('batch_size',64,
                            '''batch size''')

tf.app.flags.DEFINE_integer('variation_dim',2,
                            'para dim')
tf.app.flags.DEFINE_integer('linear_units',500,'')

tf.app.flags.DEFINE_float('learning_rate',1e-3,'')
tf.app.flags.DEFINE_integer('decay_steps',10000,'')
tf.app.flags.DEFINE_float('decay_rate',0.8,'')
tf.app.flags.DEFINE_string('ckpt','./ckpt0/','')
tf.app.flags.DEFINE_integer('max_steps','1000000','')
tf.app.flags.DEFINE_integer('regular_step',50000,'')


class Encoder(object):
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.image_w = FLAGS.image_w #28
        self.image_h = FLAGS.image_h #28
        self.channel = FLAGS.channel #1
        self.filters=[64,16]
        self.kernel = [(3,3),(3,3)]
        self.variation_dim = FLAGS.variation_dim
        self.linear_units = 500
    def input(self):
        input = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.image_w*self.image_h*self.channel],name='encoder_input')
        return input

    def builde_encode(self,input):
        # output shape 64*28*28*64
        input = tf.reshape(input,shape=[self.batch_size,self.image_w,self.image_h,self.channel])
        tf.summary.image('input_image', input)
        with tf.variable_scope('conv_layer1'):
            tensor = conv_layer(input,filters=self.filters[0],k_size=self.kernel[0],strides=1)

        # output shape = 64*14*14*64
        with tf.variable_scope('max_pool_layer_1'):
            ksize = [1, 5, 5, 1]
            strides = [1, 2, 2, 1]
            tensor = tf.nn.max_pool(tensor, ksize=ksize, strides=strides, padding='SAME')
        # output shape = 64*14*14*64
        with tf.variable_scope('conv_layer2'):
            tensor = conv_layer(tensor,self.filters[1],self.kernel[1],strides=1)
        # output shape = 64*7*7*64
        with tf.variable_scope('max_pool_layer_1'):
            ksize = [1,5,5,1]
            strides = [1,2,2,1]
            tensor = tf.nn.max_pool(tensor,ksize=ksize,strides=strides,padding='SAME')
            # output_size = tf.shape(pool)[1,2]
        tensor = tf.reshape(tensor, [self.batch_size, -1])

        with tf.variable_scope('linear'):
            tensor = linear_layer(tensor,self.linear_units)

        with tf.variable_scope('latent'):
            tensor = linear_layer(tensor,self.variation_dim,lambda x:x)
            tensor2 = linear_layer(tensor,self.variation_dim,lambda x:x,name='2')
        return tensor, tensor2

    def __call__(self,input):
        return self.builde_encode(input)





class Decoder(object):
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.sampler = Sampler(1234)
        self.e_size=[FLAGS.batch_size,7,7,16]
        self.in_channel= 16
        self.filters = [64,1]
        self.kernel = [(3,3),(3,3)]
        self.linear_units = 500
        self.deconv_size = [[self.batch_size,14,14,self.filters[0]],[self.batch_size,28,28,self.filters[1]]]

        self.variation_dim = FLAGS.variation_dim
    def sample(self,variation_paras):
        eps = tf.random_normal((self.batch_size, FLAGS.variation_dim), 0, 1,
                               dtype=tf.float32)
        # eps = self.sampler.sample()
        mu, delta = variation_paras
        # delta = tf.matmul(tf.reshape(delta,-1),epsilon)
        z = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(delta)), eps))
        return z
    def build_decoder(self,z):

        # output_shape = 64*14*14*6
        tensor= z
        # with tf.variable_scope('decoder_linear'):
        #     tensor = linear_layer(tensor,self.linear_units)

        with tf.variable_scope('decoder_linear_1'):
            tensor = linear_layer(tensor,self.e_size[1]*self.e_size[2]*self.e_size[3])

        tensor = tf.reshape(tensor,self.e_size)
        with tf.variable_scope('deconv_layer1'):
            filter_shape = [self.kernel[0][0], self.kernel[0][1], self.filters[0], self.in_channel]
            weights = tf.get_variable(name='weight',
                                      shape=filter_shape,
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(-0.06, 0.06))
            strides = [1,2,2,1]
            tensor = tf.nn.conv2d_transpose(tensor,weights,output_shape=self.deconv_size[0],strides=strides)
            bias = tf.get_variable(name='bias',shape=[self.deconv_size[0][-1]])
            tensor = tf.nn.relu(tf.nn.bias_add(tensor,bias))
        with tf.variable_scope('deconv_layer2'):
            filter_shape = [self.kernel[1][0], self.kernel[1][1], self.filters[1], self.filters[0]]
            weights = tf.get_variable(name='weight',
                                      shape=filter_shape,
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(-0.06, 0.06))
            strides = [1, 2, 2, 1]
            tensor = tf.nn.conv2d_transpose(tensor, weights, output_shape=self.deconv_size[1], strides=strides)
            bias = tf.get_variable(name='bias', shape=[self.deconv_size[1][-1]])
            image = tf.nn.sigmoid(tf.nn.bias_add(tensor,bias))
            # print(image.shape)

        return image

    def __call__(self,paras):
        return self.build_decoder(paras)



class Sampler(object):
    def __init__(self,seed=1234):
        self.seed = seed
        self.size = FLAGS.variation_dim
        np.random.seed(self.seed)
        self.batch_size = FLAGS.batch_size
    def sample(self):
        mu = np.array([0]*self.size)
        conv = np.zeros(shape=(self.size,self.size))
        for i in range(self.size):
            conv[i,i] = 1.0
        return np.random.multivariate_normal(mu,conv,self.batch_size)




class VAE(object):
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.decay_rate = FLAGS.decay_rate
        self.decay_steps = FLAGS.decay_steps
        self.learning_rate = FLAGS.learning_rate
        self.image_w = FLAGS.image_w
        self.image_h = FLAGS.image_h
        self.image_size = self.image_w * self.image_w
        self.encoder = Encoder()
        self.decoder = Decoder()


    def build_graph(self):
        input = self.encoder.input()
        # image = tf.reshape(input,[self.batch_size,self.image_w,self.image_h,1])
        # tf.summary.image('input_image', image)

        paras = self.encoder(input)
        z =self.decoder.sample(paras)
        output = self.decoder(z)
        print(output.shape)
        tf.summary.image('output_image',output)
        return input,paras,output

    def _compute_loss(self,input,paras,output):
        mu,log_var = paras

        # KL = tf.reduce_mean(tf.reduce_sum(tf.ones(shape=mu.shape)+log_var-tf.square(mu)-tf.exp(log_var),axis = 1))
        latent_loss = tf.reduce_mean(tf.constant(-0.5)*tf.reduce_sum(tf.ones(shape=mu.shape)+log_var-tf.square(mu)-tf.exp(log_var),axis = 1))
        tf.add_to_collection(tf.GraphKeys.LOSSES,latent_loss)
        tf.summary.scalar('KL_loss',latent_loss)
        # flatten_input = tf.reshape(input,shape=[self.batch_size,-1])
        flatten_output = tf.reshape(output,shape=[self.batch_size,-1])
        # recon_loss = -tf.reduce_sum(input*tf.log(1e-10+flatten_output)+(1-input)*tf.log(1e-10+1-flatten_output))/self.batch_size
        recon_loss = tf.losses.log_loss(input,flatten_output)*self.image_size
        # global_step = tf.train.get_or_create_global_step()
        # theta = tf.tanh(tf.multiply(0.001,tf.cast(global_step-FLAGS.regular_step,tf.float32)))
        # theta = tf.cond(global_step>FLAGS.regular_step,lambda :theta,lambda :0.0)
        # tf.summary.scalar('theta',theta)
        # recon_loss = tf.losses.mean_squared_error(input, flatten_output)
        # recon_loss = tf.losses.sigmoid_cross_entropy(input,flatten_output)
        tf.add_to_collection(tf.GraphKeys.LOSSES,recon_loss)
        tf.summary.scalar('Re_loss',recon_loss)
        loss = latent_loss+recon_loss
        # loss = recon_loss
        tf.summary.scalar('loss',loss)
        return loss

    def train_op(self,loss):
        global_step = tf.train.get_or_create_global_step()
        # loss = tf.add_n(tf.get_collection('t'))
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   decay_rate=self.decay_rate,
                                                   decay_steps=self.decay_steps,
                                                   global_step=global_step)

        tf.summary.scalar('computed_loss',loss)
        tf.summary.scalar('lr',learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = opt.compute_gradients(loss=loss)

        gradsl = []
        for grads,vars in grads_and_vars:
            gradsl.append(grads)
            print(vars)


        # tf.summary.histogram('grad',grads_and_vars)
        apply_op = opt.apply_gradients(grads_and_vars,global_step=global_step)
        return apply_op



def train(vae,dataset):
    input, paras,output = vae.build_graph()
    print(output.shape)
    loss = vae._compute_loss(input,paras,output)
    # loss =tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
    train_op = vae.train_op(loss)
    global_step = tf.train.get_or_create_global_step()
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.ckpt)
    writer.add_graph(graph=tf.get_default_graph())
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    if not os.path.exists(FLAGS.ckpt):
        os.mkdir(FLAGS.ckpt)


    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:

            sess.run(init_op)

        for i in range(FLAGS.max_steps):
            feed_dict = {input: dataset.next_batch(vae.batch_size)[0]}
            o,p,l,g_step,grads,summary_str = sess.run([output,paras,loss,global_step,train_op,summary_op],feed_dict=feed_dict)
            writer.add_summary(summary_str, global_step=g_step)
            if g_step%100==0:
                # summary_str = sess.run(summary_op)

                print(g_step,l)
                print(p[1][0][:5],p[0][0][:5])
                # print(grads)
                saver.save(sess,save_path=FLAGS.ckpt+'model',global_step=g_step)








if __name__ == '__main__':
    vae = VAE()
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    train(vae,mnist.train)