#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model.py
@time: 2017/12/15 10:31
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utils
from gan_utils import BaseDiscriminator,BaseSampler,BaseGenerator,BaseGAN
import os
import sys
sys.path.append('./')
import matplotlib.pyplot as plt


from config import FLAGS







class ConvGenerator(BaseGenerator):

    def build_graph(self,tensor):
        with tf.variable_scope('conv1'):
            tensor = utils.conv_layer(tensor,filters=64,k_size=[5,5],strides=1)
        with tf.variable_scope('conv2'):
            tensor = utils.conv_layer(tensor,filters=32,k_size=[5,5],strides=1)
        with tf.variable_scope('pool1'):
            tensor = tf.nn.pool(tensor,window_shape=[5,5],strides=[4,4],padding="SAME",pooling_type='MAX')
        with tf.variable_scope('de_conv1'):
            tensor = utils.de_conv_layer(tensor,filters=64,kernel=[5,5],strides=4)
        with tf.variable_scope('de_conv2'):
            tensor = utils.de_conv_layer(tensor,filters=FLAGS.channel,kernel=[5,5],strides=1)
        return tensor

class ConvDiscriminator(BaseDiscriminator):
    def __init__(self,batch_size=FLAGS.batch_size):
        BaseDiscriminator.__init__(self,batch_size)
        self.image_w = FLAGS.image_w
        self.image_h = FLAGS.image_h
        self.channel = FLAGS.channel
    def inputs(self):
        return tf.placeholder(tf.float32,shape=[self.batch_size,self.image_w,self.image_h,self.channel])
    def build_graph(self,tensor):
        with tf.variable_scope('conv1'):
            tensor = utils.conv_layer(tensor,filters=64,k_size=[5,5],strides=1)
        with tf.variable_scope('conv2'):
            tensor = utils.conv_layer(tensor,filters=32,k_size=[5,5],strides=1)
        with tf.variable_scope('pool1'):
            tensor = tf.nn.pool(tensor,window_shape=[5,5],strides=[3,3],padding="SAME",pooling_type='MAX')
        with tf.variable_scope('conv3'):
            tensor = utils.conv_layer(tensor,filters=16,k_size=[5,5],strides=1)
        with tf.variable_scope('pool2'):
            tensor = tf.nn.pool(tensor,window_shape=[5,5],strides=[3,3],padding="SAME",pooling_type='MAX')
        tensor = tf.reshape(tensor,shape=[self.batch_size,-1])
        with tf.variable_scope('linear1'):
            tensor = utils.linear_layer(tensor,128)
        with tf.variable_scope('linear2'):
            tensor = utils.linear_layer(tensor,1,activate=tf.nn.sigmoid)
        return tensor

class PatchedConvDiscriminator(BaseDiscriminator):
    def __init__(self,batch_size=FLAGS.batch_size):
        BaseDiscriminator.__init__(self,batch_size)
        self.image_w = FLAGS.image_w
        self.image_h = FLAGS.image_h
        self.channel = FLAGS.channel
    def inputs(self):
        return tf.placeholder(tf.float32,shape=[self.batch_size,self.image_w,self.image_h,self.channel])
    def build_graph(self,tensor):
        with tf.variable_scope('patch_layer0'):
            patch_logits0 = tf.reduce_mean(utils.conv_layer(tensor,filters=1,k_size=[5,5],strides=5,
                                             padding='VALID',activate=tf.sigmoid))
        with tf.variable_scope('conv1'):
            tensor = utils.conv_layer(tensor,filters=64,k_size=[5,5],strides=1)
        with tf.variable_scope('patch_layer1'):
            patch_logits1 = tf.reduce_mean(utils.conv_layer(tensor,filters=1,k_size=[5,5],strides=5,
                                             padding='VALID',activate=tf.sigmoid))
        with tf.variable_scope('conv2'):
            tensor = utils.conv_layer(tensor,filters=32,k_size=[5,5],strides=1)
        with tf.variable_scope('patch_layer2'):
            patch_logits2 = tf.reduce_mean(utils.conv_layer(tensor,filters=1,k_size=[5,5],strides=5,
                                             padding='VALID',activate=tf.sigmoid))
        with tf.variable_scope('pool1'):
            tensor = tf.nn.pool(tensor,window_shape=[5,5],strides=[3,3],padding="SAME",pooling_type='MAX')
        with tf.variable_scope('conv3'):
            tensor = utils.conv_layer(tensor,filters=16,k_size=[5,5],strides=1)
        with tf.variable_scope('patch_layer3'):
            patch_logits3 = tf.reduce_mean(utils.conv_layer(tensor,filters=1,k_size=[5,5],strides=5,
                                             padding='VALID',activate=tf.sigmoid))
        with tf.variable_scope('pool2'):
            tensor = tf.nn.pool(tensor,window_shape=[5,5],strides=[3,3],padding="SAME",pooling_type='MAX')
        tensor = tf.reshape(tensor,shape=[self.batch_size,-1])
        with tf.variable_scope('linear1'):
            tensor = utils.linear_layer(tensor,128)
        with tf.variable_scope('linear2'):
            tensor = utils.linear_layer(tensor,1,activate=tf.nn.sigmoid)

        # mixed logits
        tensor = 0.5*tensor+0.125*patch_logits0+0.125*patch_logits1+0.125*patch_logits2+0.125*patch_logits3

        return tensor


class src_G(ConvGenerator):
    def __init__(self,batch_size=FLAGS.batch_size):
        ConvGenerator.__init__(self,batch_size)
    def __call__(self,tensor):
        with tf.variable_scope('src_G',reuse=tf.AUTO_REUSE):
            return self.build_graph(tensor)

class src_D(ConvDiscriminator):
    def __call__(self,tensor):
        with tf.variable_scope('src_D',reuse=tf.AUTO_REUSE):
            return self.build_graph(tensor)

class tgt_G(ConvGenerator):
    def __init__(self,batch_size=FLAGS.batch_size):
        ConvGenerator.__init__(self,batch_size)
        # self.tgt_shape=[128,128,4]
    def __call__(self,tensor):
        with tf.variable_scope('tgt_G',reuse=tf.AUTO_REUSE):
            return self.build_graph(tensor)

class tgt_D(ConvDiscriminator):
    def __call__(self,tensor):
        with tf.variable_scope('tgt_D',reuse=tf.AUTO_REUSE):
            return self.build_graph(tensor)



class CycleGAN(object):
    def __init__(self,src_generator,src_discriminator,tgt_generator,tgt_discriminator):
        self.src_g = src_generator
        self.src_d =src_discriminator
        self.tgt_g = tgt_generator
        self.tgt_d = tgt_discriminator
        self.batch_size = FLAGS.batch_size
        self.image_w = FLAGS.image_w
        self.image_h = FLAGS.image_h
        self.channel = FLAGS.channel
    def compute_G_loss(self,logits):
        return tf.reduce_mean(tf.log(1e-6+1-tf.sigmoid(logits)))
    def compute_real_D_loss(self,logits):
        return tf.reduce_mean(-tf.log(1e-6+tf.sigmoid(logits)))
    def compute_fake_D_loss(self,logits):
        return tf.reduce_mean(-tf.log(1e-6+1-tf.sigmoid(logits)))
    def recon_loss(self,image1,image2):
        return tf.reduce_mean(tf.abs(image1-image2))

    def train_op(self,d_loss,start):
        vars = []
        for var in tf.global_variables():
            if var.name.startswith(start):
                vars.append(var)
        opt = tf.train.AdamOptimizer()
        grads_and_vars = opt.compute_gradients(d_loss,var_list=vars)
        for g,v in grads_and_vars:
            tf.summary.histogram(v.name+'_grad',g)
            tf.summary.scalar(v.name+'_gradients',tf.reduce_sum(tf.abs(g)))
        op = opt.apply_gradients(grads_and_vars)
        return op

def train(gan,src,tgt):
    #source to target then reconstruct source
    real_src = tf.reshape(src.get_next(),[gan.batch_size,gan.image_w,gan.image_h,gan.channel])
    fake_tgt = gan.src_g(real_src)
    recon_src = gan.tgt_g(fake_tgt)

    # target to source then reconstruct target
    real_tgt = tf.reshape(tgt.get_next(),[gan.batch_size,gan.image_w,gan.image_h,gan.channel])
    fake_src = gan.tgt_g(real_tgt)
    recon_tgt = gan.src_g(fake_src)


    # logits from source and trans source
    real_src_logits = gan.src_d(real_src)
    fake_src_logits = gan.src_d(fake_src)

    # logits from target and trans target
    real_tgt_logits = gan.tgt_d(real_tgt)
    fake_tgt_logits = gan.tgt_d(fake_tgt)

    # reconstruct loss of target and source
    tgt_recon_loss = gan.recon_loss(real_tgt, recon_tgt)
    src_recon_loss = gan.recon_loss(real_src, recon_src)

    # generate loss of target and source
    src_G_loss = gan.compute_G_loss(fake_src_logits)+src_recon_loss+tgt_recon_loss
    tgt_G_loss = gan.compute_G_loss(fake_tgt_logits)+src_recon_loss+tgt_recon_loss

    # discriminator loss of source and target
    src_D_loss = gan.compute_real_D_loss(real_src_logits) + \
        gan.compute_fake_D_loss(fake_src_logits)
    tgt_D_loss = gan.compute_real_D_loss(real_tgt_logits) + \
        gan.compute_fake_D_loss(fake_tgt_logits)

    # train op
    tgt_g_op = gan.train_op(tgt_G_loss,'tgt_G')
    src_g_op = gan.train_op(src_G_loss,'src_G')

    tgt_d_op = gan.train_op(tgt_D_loss,'tgt_D')
    src_d_op = gan.train_op(src_D_loss,'src_D')

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.ckpt, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    global_step = tf.train.get_or_create_global_step()
    global_step = tf.assign_add(global_step, 1)
    init_op = tf.global_variables_initializer()

    train_op = [tgt_d_op, tgt_g_op, src_d_op, src_g_op]
    with tf.Session() as sess:
        utils.load_or_initial_model(sess, FLAGS.ckpt, saver, init_op)
        a = 0
        G_step = 0
        D_step = 0
        i = 0
        if not os.path.exists('out/'):
            os.makedirs('out/')
        for step in range(FLAGS.max_steps):
            # if step<1000:
            #     data = datasets.next_batch(gan.batch_size * 2)
            #     feed_dict = {real_example: data[0][:gan.batch_size], real_example2: data[0][gan.batch_size:]}
            #     a, loss_d, loss_g, _, g_step, summary_str = \
            #         sess.run([acc, d_loss, g_loss, train_op[0], global_step, summary_op], feed_dict=feed_dict)
            #     continue

            # feed_dict = {real_src:src_data,real_tgt:tgt_data}
            _, g_step, summary_str = \
                sess.run([train_op, global_step, summary_op])
            D_step += 1
            G_step += 1
            if g_step % 1 == 0:
                print(g_step, a, D_step, G_step)
                samples = sess.run(fake_src)
                samples = samples[:16]
                writer.add_summary(summary_str, global_step=g_step)
                # saver.save(sess, FLAGS.ckpt, global_step=g_step)
                fig = utils.plot(samples,shape=(256,256,3),cmap=None)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)

                samples = sess.run(fake_tgt)
                samples = samples[:16]
                fig = utils.plot(samples,shape=(256,256,3),cmap=None)
                plt.savefig('out/{}_1.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)
                i += 1


def decode_f(filename):
    image_raw = tf.read_file(filename,'rb')
    image = tf.image.decode_jpeg(image_raw,3)
    image = tf.reshape(tf.cast(image,tf.float32),[256,256,3])
    return image

def main():
    src_d = src_D()
    src_g = src_G()
    tgt_d = tgt_D()
    tgt_g = tgt_G()


    gan = CycleGAN(src_g,src_d,tgt_g,tgt_d)
    src_file = tf.gfile.Glob('E:/GAN/cycleGAN/monet2photo/trainA/*.jpg')
    src = tf.data.Dataset().list_files(src_file)
    src = src.map(decode_f).batch(gan.batch_size).repeat(5)
    src_iter = src.make_one_shot_iterator()

    tgt_file = tf.gfile.Glob('E:/GAN/cycleGAN/monet2photo/trainB/*.jpg')
    tgt = tf.data.Dataset().list_files(tgt_file)
    tgt = tgt.map(decode_f).batch(gan.batch_size).repeat(5)
    tgt_iter = tgt.make_one_shot_iterator()

    train(gan,src_iter,tgt_iter)






if __name__ == '__main__':
    main()