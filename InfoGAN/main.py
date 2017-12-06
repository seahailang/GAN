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

import tensorflow as tf
import utils
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('./')
from datetime import datetime
import DG

from config import FLAGS


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
        true_labels = tf.ones(shape=[true_examples.shape[0],1],dtype=tf.int32)
        false_labels = tf.zeros(shape=[true_examples.shape[0],1],dtype=tf.int32)
        # labels = tf.concat([false_labels,true_labels],0)
        return true_logits,true_labels,false_logits,false_labels
    def compute_G_loss(self,logits,labels):
        return -tf.reduce_mean(tf.log(tf.sigmoid(logits)))
        # return tf.reduce_mean(tf.losses.sigmoid_cross_entropy(1-labels,logits))
    def compute_D_loss(self,logits,labels):
        return -tf.reduce_mean(tf.log(tf.sigmoid(1-logits)))
        # return -tf.reduce_mean((1-tf.cast(labels,tf.float32))*tf.log(1-logits))
        # return tf.reduce_mean(tf.losses.sigmoid_cross_entropy(labels,logits))
    def train_op(self,g_loss,d_loss):

        learning_rate = FLAGS.learning_rate
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
            tf.summary.scalar(v.name+'_gradients',tf.reduce_sum(g))
        for g,v in d_grads_and_vars:
            tf.summary.histogram(v.name+'_grad',g)
            tf.summary.scalar(v.name+'_gradients',tf.reduce_sum(g))
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
    true_logits, true_labels, false_logits, false_labels = gan.build_graph(true_examples=true_examples,
                                                                  false_examples=false_examples)
    predict = tf.less_equal(tf.nn.sigmoid(false_logits),0.5)
    # p_label = tf.argmax(false_labels,1,output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(predict,tf.float32))
    tf.summary.scalar('acc',acc)
    true_d_loss = gan.compute_G_loss(logits=true_logits, labels=true_labels)
    false_d_loss = gan.compute_D_loss(logits=false_logits, labels=false_labels)
    d_loss = 0.5*true_d_loss+0.5*false_d_loss
    tf.summary.scalar('true_d_loss',true_d_loss)
    tf.summary.scalar('false_d_loss',false_d_loss)
    g_loss = gan.compute_G_loss(logits=false_logits, labels=false_labels)
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
            a,loss_d, loss_g, _, g_step, summary_str = \
                sess.run([acc, d_loss, g_loss, train_op[0], global_step, summary_op], feed_dict=feed_dict)
            for i in range(20):
                _=sess.run(g_op)

            if g_step%100 == 0:
                print(g_step,loss_d,loss_g,a)
                writer.add_summary(summary_str,global_step=g_step)
                saver.save(sess,FLAGS.ckpt,global_step=g_step)










def main():
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    discriminator = DG.Discriminator()
    generator = DG.Generator()
    sampler = DG.Sampler(seed=1234)
    gan = Model(generator=generator,discriminator=discriminator,sampler=sampler)
    train(gan,mnist.train)




if __name__ == '__main__':
    main()