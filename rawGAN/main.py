#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: main.py
@time: 2017/11/24 13:55
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import utils

sys.path.append('./')
import DG

from rawGAN.config import FLAGS


class Model(object):
    def __init__(self,generator,discriminator,sampler):
        self.generator = generator
        self.discriminator = discriminator
        self.sampler = sampler
        self.learning_rate = FLAGS.learning_rate

    def input(self):
        return self.discriminator.input()
    def sample(self):
        return self.sampler.sample()
    def build_graph(self,true_examples,false_examples):
        # z = self.sampler.sample()
        # false_examples = self.generator(z)
        # true_examples = self.discriminator.input()
        # train_examples = tf.concat([false_examples,true_examples],0)
        false_logits = self.discriminator(false_examples)
        true_logits = self.discriminator(true_examples)
        # logits = tf.concat([false_logits,true_logits],0)
        # true_labels = tf.ones(shape=[true_examples.shape[0],1],dtype=tf.int32)
        # false_labels = tf.zeros(shape=[true_examples.shape[0],1],dtype=tf.int32)
        # labels = tf.concat([false_labels,true_labels],0)
        return true_logits,false_logits
    def compute_G_loss(self,logits,labels):
        # return -tf.reduce_mean(tf.log((logits)))
        return tf.reduce_mean(tf.losses.sigmoid_cross_entropy(1-labels,logits))
    def compute_D_loss(self,logits,labels):
        # return -tf.reduce_mean(tf.log((1-logits)))
        # return -tf.reduce_mean((1-tf.cast(labels,tf.float32))*tf.log(1-logits))
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))
    def train_op(self,g_loss,d_loss):

        learning_rate = FLAGS.learning_rate
        d_opt = tf.train.AdamOptimizer()
        g_opt = tf.train.AdamOptimizer()
        var_list = tf.trainable_variables()
        d_var = []
        g_var = []
        for var in var_list:
            if var.name.startswith('D'):
                d_var.append(var)
            else:
                g_var.append(var)
        # g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

        # g_grads_and_vars = opt.compute_gradients(g_loss)
        return d_apply_op,g_apply_op

def train(gan,datasets):
    true_examples = gan.input()
    z = gan.sample()
    # n=tf.ones(shape=z.shape)
    # tf.summary.scalar('z',z)
    false_examples = gan.generator(z)
    true_logits, false_logits = gan.build_graph(true_examples=true_examples,
                                                                  false_examples=false_examples)
    predict = tf.less_equal((false_logits),0.5)
    # p_label = tf.argmax(false_labels,1,output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(predict,tf.float32))
    tf.summary.scalar('acc',acc)
    true_d_loss = gan.compute_D_loss(logits=true_logits, labels=tf.ones(shape=true_logits.shape))
    false_d_loss = gan.compute_D_loss(logits=false_logits, labels=tf.zeros(shape=false_logits.shape))
    # d_loss = true_d_loss+false_d_loss
    d_loss = tf.reduce_mean(false_logits)-tf.reduce_mean(true_logits)
    tf.summary.scalar('true_d_loss',true_d_loss)
    tf.summary.scalar('false_d_loss',false_d_loss)
    # g_loss = gan.compute_D_loss(logits=false_logits, labels=tf.ones(shape=false_logits.shape))
    g_loss = -2*tf.reduce_mean(false_logits)
    tf.summary.scalar('g_loss', g_loss)

    # true_d_op = gan.train_op(true_d_loss,'D')
    # false_d_op = gan.train_op(d_loss,'D')
    d_op,g_op = gan.train_op(d_loss=d_loss,g_loss=g_loss)
    global_step = tf.train.get_or_create_global_step()
    global_step = tf.assign_add(global_step,1)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.ckpt)
    writer.add_graph(graph=tf.get_default_graph())
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    train_op = [d_op,g_op]
    with tf.Session() as sess:
        utils.load_or_initial_model(sess,FLAGS.ckpt,saver,init_op)
        a=0
        G_step=0
        D_step=0
        i=0
        if not os.path.exists('out/'):
            os.makedirs('out/')
        for step in range(FLAGS.max_steps):
            feed_dict = {true_examples:datasets.next_batch(gan.sampler.batch_size)[0]}
            # for i in range(10):
            #     _z,loss_d,loss_g,_,g_step,summary_str = \
            #         sess.run([z,d_loss,g_loss,train_op[0],global_step,summary_op],feed_dict=feed_dict)
            # for i in range(10):
            #     _z, loss_d, loss_g, _, g_step, summary_str = \
            #         sess.run([z, d_loss, g_loss, train_op[1], global_step, summary_op], feed_dict=feed_dict)
            # for i in range(5):
            #     sess.run(g_op,feed_dict=feed_dict)
            # for i in range(5):
            #     sess.run(d_op,feed_dict=feed_dict)
            # for i in range(10):
            #     sess.run(d_op,feed_dict=feed_dict)
            if False:
                pass
            # if a<0.5:
            #     a,loss_d, loss_g, _, g_step, summary_str = \
            #         sess.run([acc, d_loss, g_loss, train_op[0], global_step, summary_op], feed_dict=feed_dict)
            #     D_step+=1
            # elif a>0.8:
            #     # a, loss_d, loss_g, _, g_step, summary_str = \
            #     #     sess.run([acc, d_loss, g_loss, train_op, global_step, summary_op], feed_dict=feed_dict)
            #     a, loss_d, loss_g, _, g_step, summary_str = \
            #         sess.run([acc, d_loss, g_loss, train_op[1], global_step, summary_op], feed_dict=feed_dict)
            #     G_step+=1
            #     D_step+=0
            else:
                a, loss_d, loss_g, _, g_step, summary_str = \
                    sess.run([acc, d_loss, g_loss, train_op, global_step, summary_op], feed_dict=feed_dict)
                D_step+=1
                G_step+=1
            if g_step % 1000 == 0:
                print(g_step, loss_d, loss_g, a,D_step,G_step)
                samples = sess.run(false_examples)
                samples=samples[:16]
                writer.add_summary(summary_str, global_step=g_step)
                # saver.save(sess, FLAGS.ckpt, global_step=g_step)
                fig = plot(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)



def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig





def main():
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    discriminator = DG.Discriminator()
    generator = DG.Generator()
    sampler = DG.Sampler(seed=1234)
    gan = Model(generator=generator,discriminator=discriminator,sampler=sampler)
    train(gan,mnist.train)




if __name__ == '__main__':
    main()