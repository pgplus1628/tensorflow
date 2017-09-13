#!/usr/bin/python

from __future__ import print_function
import tensorflow as tf
import numpy as np
import pprint as pp

print(tf.__version__)

with tf.device('/gpu:1') :
    a = tf.placeholder(tf.float32, [2,3])
    r = tf.argmax2d(a)

A = np.random.rand(2, 3).astype('float32')
pp.pprint(A)


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=False))

print(sess.run(r, {a:A}))
