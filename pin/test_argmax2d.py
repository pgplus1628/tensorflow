#!/usr/bin/python

from __future__ import print_function
import tensorflow as tf

print(tf.__version__)

data = tf.ones([2,3])

r = tf.argmax2d(data)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(r))
