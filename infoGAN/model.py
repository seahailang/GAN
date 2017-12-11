#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model.py
@time: 2017/12/11 10:02
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utils
import gan_utils
import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('./')
from config import FLAGS

class info_G(gan_utils.BaseGenerator):
    def __init__(self,batch_size=FLAGS.batch_size):
        super(info_G,self).__init__(batch_size)
        self.example_shape = FLAGS.image_w*FLAGS.image_h*FLAGS.channel
        self.image_w = FLAGS.image_w
        self.image_h = FLAGS.image_h
        self.channel = FLAGS.channel

    def build_graph(self,tensor):
        with tf.variable_scope('linear1'):
            tensor = utils.linear_layer(tensor,256)
        with tf.variable_scope('linear2'):
            tensor = utils.linear_layer(tensor,self.example_shape,activate=tf.nn.sigmoid)
        image = tf.reshape(tensor,shape=[self.batch_size,self.image_w,self.image_h,self.channel])
        tf.summary.image('out_image',image)
        return tensor

class info_D(gan_utils.BaseDiscriminator):
    def __init__(self,batch_size=FLAGS.batch_size):
        super(info_D,self).__init__(batch_size)
        self.example_shape = FLAGS.image_w * FLAGS.image_h * FLAGS.channel
        self.image_w = FLAGS.image_w
        self.image_h = FLAGS.image_h
        self.channel = FLAGS.channel
        self.cat_num = FLAGS.cat_num

    def input(self):
        inputs = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.example_shape])
        image = tf.reshape(inputs,shape=[self.batch_size,self.image_w,self.image_h,self.channel])
        tf.summary.image('out_image', image)
        labels = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.cat_num])
        return inputs,labels

    def build_graph(self,tensor):
        with tf.variable_scope('linear1'):
            tensor = utils.linear_layer(tensor,128)
        with tf.variable_scope('linear'):
            tensor = utils.linear_layer(tensor,1,activate=lambda x:x)
        return tensor

class info_Q(object):
    def __init__(self):
        self.cat_num = FLAGS.cat_num
    def build_graph(self,tensor):
        with tf.variable_scope('linear1'):
            tensor = utils.linear_layer(tensor,128)
        with tf.variable_scope('linear2'):
            tensor = utils.linear_layer(tensor,self.cat_num,activate=lambda x:x)
        return tensor
    def __call__(self,example):
        with tf.variable_scope('Q',reuse=tf.AUTO_REUSE):
            cat_logitss = self.build_graph(example)
        return cat_logitss


class info_S(gan_utils.BaseSampler):
    def __int__(self,seed=1234,batch_size=FLAGS.batch_size,latent_dims=FLAGS.latent_dims):
        super(info_S,self).__init__(seed=seed,batch_size= batch_size,latent_dims=latent_dims)
        self.cat_num = 10

    def sample(self):
        z = tf.random_uniform(shape=[self.batch_size,self.latent_dims],minval=-1.0,maxval=1.0)
        logitss = tf.tile([[0.1]*10],[self.batch_size,1])
        cat = tf.multinomial(logitss,num_samples=1)
        c = tf.one_hot(cat,depth=10,axis=1)
        c = tf.reshape(c,shape=[self.batch_size,-1])
        latent = tf.concat([z,c],axis=-1)
        return latent,c

class info_GAN(gan_utils.BaseGAN):
    def __init__(self,generator,discriminator,sampler,q_net):
        super(info_GAN,self).__init__(generator=generator,discriminator=discriminator,sampler=sampler)
        self.q_net = q_net
    def compute_G_loss(self,logitss,label):
        return tf.reduce_mean(-tf.log(1e-6+tf.sigmoid(logitss)))
    def compute_real_D_loss(self,logitss,label):
        return tf.reduce_mean(-tf.log(1e-6+tf.sigmoid(logitss)))
    def compute_fake_D_loss(self,logitss,label):
        return tf.reduce_mean(-tf.log(1e-6+1-tf.sigmoid(logitss)))
    def compute_Q_loss(self,cat_logitss,cat_labels):
        return tf.losses.softmax_cross_entropy(onehot_labels=cat_labels,logitss=cat_logitss)
    def q_train_op(self,q_loss):
        opt = tf.train.AdamOptimizer()
        varlist=[]
        for var  in tf.global_variables():
            if var.name.startswith('Q') or var.name.startswith('G'):
                varlist.append(var)
        grads_and_vars = opt.compute_gradients(q_loss,varlist)
        op = opt.apply_gradients(grads_and_vars)
        return op

def train(gan,datasets):
    real_example,real_cat = gan.input()
    latent,fake_cat = gan.sample()
    fake_example = gan.generator(latent)
    real_logits = gan.discriminator(real_example)
    real_cat_logits = gan.q_net(real_example)
    fake_logits= gan.discriminator(fake_example)
    fake_cat_logits = gan.q_net(fake_example)

    predict = tf.less_equal((fake_logits), 0.5)
    # p_label = tf.argmax(false_labels,1,output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(predict, tf.float32))
    tf.summary.scalar('acc', acc)

    g_loss = gan.compute_G_loss(fake_logits,tf.ones(shape=[gan.batch_size,1]))
    # d_loss = gan.compute_D_loss(real_logits,tf.ones(shape=[gan.batch_size,1]))+\
    #     gan.compute_D_loss(fake_logits,tf.zeros(shape=[gan.batch_size,1]))
    fake_d_loss = gan.compute_fake_D_loss(fake_logits,tf.zeros_like(fake_logits))

    real_d_loss = gan.compute_real_D_loss(real_logits,tf.ones_like(real_logits))
    d_loss = fake_d_loss+real_d_loss
    q_loss = gan.compute_Q_loss(fake_cat_logits,fake_cat)+gan.compute_Q_loss(real_cat_logits,real_cat)
    d_op,g_op = gan.train_op(g_loss=g_loss,d_loss=d_loss)
    q_op = gan.q_train_op(q_loss)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.ckpt,graph=tf.get_default_graph())
    saver = tf.train.Saver()

    global_step = tf.train.get_or_create_global_step()
    global_step = tf.assign_add(global_step, 1)
    init_op = tf.global_variables_initializer()
    train_op = [d_op, g_op,q_op]
    with tf.Session() as sess:
        for step  in range(FLAGS.max_steps):
            utils.load_or_initial_model(sess, FLAGS.ckpt, saver, init_op)
            a = 0
            G_step = 0
            D_step = 0
            i = 0
            if not os.path.exists('out/'):
                os.makedirs('out/')
            for step in range(FLAGS.max_steps):
                data =  datasets.next_batch(gan.batch_size)
                feed_dict = {real_example:data[0],real_cat:data[1]}
                a, loss_d, loss_g, _, g_step, summary_str = \
                    sess.run([acc, d_loss, g_loss, train_op, global_step, summary_op], feed_dict=feed_dict)
                D_step += 1
                G_step += 1
                if g_step % 1000 == 0:
                    print(g_step, loss_d, loss_g, a, D_step, G_step)
                    samples = sess.run(fake_example)
                    samples = samples[:16]
                    writer.add_summary(summary_str, global_step=g_step)
                    # saver.save(sess, FLAGS.ckpt, global_step=g_step)
                    fig = utils.plot(samples)
                    plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)




def main():
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    discriminator = info_D()
    generator = info_G()
    q_net = info_Q()
    sampler = info_S(seed=1234,batch_size=FLAGS.batch_size,latent_dims=FLAGS.latent_dims)
    gan = info_GAN(generator=generator, discriminator=discriminator, sampler=sampler,q_net=q_net)
    train(gan, mnist.train)





if __name__ == '__main__':
    main()