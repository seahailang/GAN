#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model.py
@time: 2017/12/14 14:30
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gan_utils
import utils
import mnist_utils
import os
import sys
sys.path.append('./')
import matplotlib.pyplot as plt

from config import FLAGS

class alpha_E(gan_utils.BaseGenerator,mnist_utils.MnistModel):
    def __init__(self,batch_size=FLAGS.batch_size):
        gan_utils.BaseGenerator.__init__(self,batch_size)
        mnist_utils.MnistModel.__init__(self)
    def build_graph(self,tensor):
        with tf.variable_scope('linear1'):
            tensor = utils.linear_layer(tensor,128)
        with tf.variable_scope('linear2'):
            tensor = utils.linear_layer(tensor,16,activate=tf.nn.sigmoid)
        return tensor
    def input(self):
        inputs=tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.example_shape])
        image = tf.reshape(inputs,shape=[self.batch_size,self.image_w,self.image_h,self.channel])
        tf.summary.image('in_image',image)
        return inputs
    def __call__(self,tensor):
        with tf.variable_scope('Encoder'):
            return self.build_graph(tensor)

class alpha_G(gan_utils.BaseGenerator,mnist_utils.MnistModel):
    def __init__(self,batch_size=FLAGS.batch_size):
        gan_utils.BaseGenerator.__init__(self,batch_size)
        mnist_utils.MnistModel.__init__(self)
    def build_graph(self,tensor):
        # with tf.variable_scope('linear1'):
        #     tensor = utils.linear_layer(tensor,128)
        with tf.variable_scope('linear2'):
            tensor = utils.linear_layer(tensor,self.example_shape,activate=tf.sigmoid)
        return tensor
    def __call__(self,tensor):
        with tf.variable_scope('Generator',reuse=tf.AUTO_REUSE):
            return self.build_graph(tensor)


class alpha_Dz(gan_utils.BaseDiscriminator):
    def __init__(self,batch_size=FLAGS.batch_size):
        gan_utils.BaseDiscriminator.__init__(self,batch_size)
        self.latent_dims = FLAGS.latent_dims
    def build_graph(self,tensor):
        with tf.variable_scope('linear1'):
            tensor = utils.linear_layer(tensor,128)
        with tf.variable_scope('linear2'):
            tensor = utils.linear_layer(tensor,1,activate=lambda x:x)
        return tensor
    def __call__(self,tensor):
        with tf.variable_scope('z_Discriminator',reuse=tf.AUTO_REUSE):
            return self.build_graph(tensor)

class Sampler(gan_utils.BaseSampler):
    def sample(self):
        z= tf.zeros(shape=[self.batch_size,self.latent_dims])
        return tf.random_normal(shape=[self.batch_size,self.latent_dims])


class alpha_D(gan_utils.BaseDiscriminator,mnist_utils.MnistModel):
    def __init__(self,batch_size=FLAGS.batch_size):
        gan_utils.BaseGenerator.__init__(self,batch_size)
        mnist_utils.MnistModel.__init__(self)
    def input(self):
        return tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.example_shape])
    def build_graph(self,tensor):
        with tf.variable_scope('linear1'):
            tensor = utils.linear_layer(tensor,128)
        with tf.variable_scope('linear2'):
            tensor = utils.linear_layer(tensor,1,activate=tf.nn.sigmoid)
        return tensor

class alpha_GAN(mnist_utils.MnistModel):
    def __init__(self,encoder,generator,discriminator_z,discriminator,sampler):
        mnist_utils.MnistModel.__init__(self)
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_z = discriminator_z
        self.batch_size=FLAGS.batch_size
        self.sampler = sampler


    def compute_G_loss(self,logits):
        return tf.reduce_mean(tf.log(1e-6+1-logits)-tf.log(1e-6+logits))
    def compute_real_D_loss(self,logits):
        return tf.reduce_mean(-tf.log(1e-6+logits))
    def compute_fake_D_loss(self,logits):
        return tf.reduce_mean(-tf.log(1e-6+1-logits))
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
    def g_train_op(self,g_loss):
        vars = []
        for var in tf.global_variables():
            if var.name.startswith('G') or var.name.startswith('E'):
                vars.append(var)
        opt = tf.train.AdamOptimizer()
        grads_and_vars = opt.compute_gradients(g_loss, var_list=vars)
        op = opt.apply_gradients(grads_and_vars)
        return op



def train(gan,datasets):
    real_example = gan.encoder.input()
    real_z = gan.sampler.sample()
    fake_z = gan.encoder(real_example)
    real_z_logits = gan.discriminator_z(real_z)
    fake_z_logits = gan.discriminator_z(fake_z)

    real_z_loss = gan.compute_real_D_loss(real_z_logits)
    fake_z_loss=  gan.compute_fake_D_loss(fake_z_logits)
    dz_loss = real_z_loss+fake_z_loss
    e_loss = gan.compute_G_loss(fake_z_logits)

    real_logits = gan.discriminator(real_example)
    fake_example = gan.generator(fake_z)
    fake_example_z = gan.generator(real_z)
    fake_logits = gan.discriminator(fake_example)
    fake_logits_z = gan.discriminator(fake_example_z)


    real_d_loss = gan.compute_real_D_loss(real_logits)
    fake_d_loss = gan.compute_fake_D_loss(fake_logits)
    fake_d_loss_z = gan.compute_fake_D_loss(fake_logits_z)
    d_loss = real_d_loss+fake_d_loss+fake_d_loss_z
    recon_loss = gan.recon_loss(real_example,fake_example)
    e_loss = e_loss+recon_loss
    g_loss =  gan.compute_G_loss(fake_logits)+recon_loss+gan.compute_G_loss(fake_logits_z)



    dz_op = gan.train_op(dz_loss, 'z_D')
    e_op = gan.train_op(e_loss, 'E')
    d_op = gan.train_op(d_loss,'D')
    g_op = gan.train_op(g_loss,'G')

    predict = tf.less_equal((fake_logits), 0.5)
    # p_label = tf.argmax(false_labels,1,output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(predict, tf.float32))
    tf.summary.scalar('acc', acc)

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.ckpt, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    global_step = tf.train.get_or_create_global_step()
    global_step = tf.assign_add(global_step, 1)
    init_op = tf.global_variables_initializer()
    train_op = [e_op,g_op,d_op,dz_op]
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
                data = datasets.next_batch(gan.batch_size)
                feed_dict = {real_example: data[0]}
                a, loss_d, loss_g, _, g_step, summary_str = \
                    sess.run([acc, d_loss, g_loss, train_op, global_step, summary_op], feed_dict=feed_dict)
                D_step += 1
                G_step += 1
                if g_step % 1000 == 0:
                    print(g_step, loss_d, loss_g, a, D_step, G_step)
                    samples = sess.run(fake_example,feed_dict=feed_dict)
                    samples = samples[:16]
                    writer.add_summary(summary_str, global_step=g_step)
                    # saver.save(sess, FLAGS.ckpt, global_step=g_step)
                    fig = utils.plot(samples)
                    plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)
                    fig = utils.plot(data[0][:16])
                    plt.savefig('out/{}_.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    plt.close(fig)


def main():
    encoder = alpha_E()
    generator = alpha_G()
    discriminatorz = alpha_Dz(FLAGS.batch_size)
    discriminator = alpha_D(FLAGS.batch_size)
    sampler = Sampler(seed=1234,batch_size=FLAGS.batch_size,latent_dims=FLAGS.latent_dims)
    gan = alpha_GAN(encoder,generator,discriminatorz,discriminator,sampler)
    mnist = mnist_utils.mnist.train
    train(gan,mnist)






if __name__ == '__main__':
    main()