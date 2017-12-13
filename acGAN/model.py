#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model.py
@time: 2017/12/12 10:44
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import gan_utils
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data
sys.path.append('./')

from config import FLAGS

class Generator(gan_utils.BaseGenerator):
    def __init__(self,batch_size=FLAGS.batch_size):
        super(Generator,self).__init__(batch_size)
        self.example_shape = FLAGS.image_w*FLAGS.image_h*FLAGS.channel
        self.image_w = FLAGS.image_w
        self.image_h = FLAGS.image_h
        self.channel = FLAGS.channel
    def build_graph(self,tensor):
        with tf.variable_scope('linear1'):
            tensor = utils.linear_layer(tensor,128)
        with tf.variable_scope('linear2'):
            tensor = utils.linear_layer(tensor,self.example_shape,activate=tf.nn.sigmoid)
        image = tf.reshape(tensor,shape=[self.batch_size,self.image_w,self.image_h,self.channel])
        tf.summary.image('out_image',image)
        return tensor

class ac_D(gan_utils.BaseDiscriminator):
    def __init__(self,batch_size=FLAGS.batch_size):
        super(ac_D,self).__init__(batch_size)
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
            tensor = utils.linear_layer(tensor,1,activate=tf.nn.sigmoid)
            # in wiseodd's github, the ac_layer did't use sigmoid as a activate function,
            # but in my experiment, if no sigmoid function,the network can't work, I don't know why
            ac_tensor = utils.linear_layer(tensor,10,activate=tf.nn.sigmoid,name='ac')
        return tensor,ac_tensor

class ac_S(gan_utils.BaseSampler):
    def __int__(self,seed=1234,batch_size=FLAGS.batch_size,latent_dims=FLAGS.latent_dims):
        super(ac_S,self).__init__(seed=seed,batch_size= batch_size,latent_dims=latent_dims)
        self.cat_num = 10

    def sample(self):
        # the ac_gan using the label from true example to generate examples,I think we can sample one too.
        z = tf.random_uniform(shape=[self.batch_size,self.latent_dims],minval=-1.0,maxval=1.0)
        logits = tf.tile([[0.1]*10],[self.batch_size,1])
        cat = tf.multinomial(logits,num_samples=1)
        c = tf.one_hot(cat,depth=10,axis=1)
        c = tf.reshape(c,shape=[self.batch_size,-1])
        latent = tf.concat([z,c],axis=-1)
        return latent,c

class ac_GAN(gan_utils.BaseGAN):
    def compute_G_loss(self,logits,label):
        return -tf.reduce_mean(tf.log(logits + 1e-8))
        # return tf.reduce_mean(-tf.log(1e-6+tf.sigmoid(logits)))

    def compute_real_D_loss(self,logits,label):
        return -tf.reduce_mean(tf.log(logits + 1e-8))
    def compute_fake_D_loss(self,logits,label):
        return -tf.reduce_mean(tf.log(1. - logits + 1e-8))
    def compute_AC_loss(self,cat_logits,cat_labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cat_logits, labels=cat_labels))

        # return tf.losses.softmax_cross_entropy(onehot_labels=cat_labels, logits=cat_logits)
def train(gan,datasets):
    real_example, real_cat = gan.input()
    # z = gan.sample()
    # c = tf.cast(real_cat,tf.float32)
    # latent = tf.concat([z,c],axis=1)
    latent,fake_cat = gan.sample()
    fake_example = gan.generator(latent)
    real_logits,real_cat_logits = gan.discriminator(real_example)
    fake_logits,fake_cat_logits = gan.discriminator(fake_example)

    predict = tf.less_equal((fake_logits), 0.5)
    acc = tf.reduce_mean(tf.cast(predict, tf.float32))
    tf.summary.scalar('acc', acc)

    g_loss = gan.compute_G_loss(fake_logits, tf.ones(shape=[gan.batch_size, 1]))
    fake_d_loss = gan.compute_fake_D_loss(fake_logits, tf.zeros_like(fake_logits))

    real_d_loss = gan.compute_real_D_loss(real_logits, tf.ones_like(real_logits))

    real_cat_loss = gan.compute_AC_loss(real_cat_logits,real_cat)
    fake_cat_loss = gan.compute_AC_loss(fake_cat_logits,fake_cat)

    # I think the fake_cat_loss didn't contribute to discriminator network
    # and if we ignore the fake_cat_loss, the network may work better
    d_loss = fake_d_loss+real_d_loss+real_cat_loss
    # d_loss = fake_d_loss + real_d_loss + real_cat_loss + fake_cat_loss
    g_loss = g_loss+ fake_cat_loss+real_cat_loss
    d_op,g_op = gan.train_op(d_loss=d_loss,g_loss=g_loss)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.ckpt, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    global_step = tf.train.get_or_create_global_step()
    global_step = tf.assign_add(global_step, 1)
    init_op = tf.global_variables_initializer()
    train_op = [d_op, g_op]
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
                # a, loss_d, loss_g, _, g_step, summary_str = \
                #     sess.run([acc, d_loss, g_loss, train_op[1], global_step, summary_op], feed_dict=feed_dict)
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
    tf.set_random_seed(1234)
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    discriminator = ac_D()
    generator = Generator()
    sampler = ac_S(seed=1234,batch_size=FLAGS.batch_size,latent_dims=FLAGS.latent_dims)
    gan = ac_GAN(generator=generator, discriminator=discriminator, sampler=sampler)
    train(gan, mnist.train)


if __name__ == '__main__':
    main()