#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: config.py
@time: 2017/12/4 19:11
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size',32,'')
tf.app.flags.DEFINE_integer('latent_dims',16,'')
tf.app.flags.DEFINE_float('learning_rate',0.00001,'')
tf.app.flags.DEFINE_integer('cat_num',10,'')
tf.app.flags.DEFINE_integer('channel',1,
                            '''image channel''')
tf.app.flags.DEFINE_integer('image_w',28,
                            '''image width''')
tf.app.flags.DEFINE_integer('image_h',28,
                            '''image height''')
tf.app.flags.DEFINE_integer('decay_steps',10000,'')
tf.app.flags.DEFINE_float('decay_rate',0.8,'')
tf.app.flags.DEFINE_string('ckpt','./ckpt0/','')
tf.app.flags.DEFINE_integer('max_steps','1000000','')

if __name__ == '__main__':
    pass