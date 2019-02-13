import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

v1 = tf.get_variable('v1', shape=(), initializer=tf.zeros_initializer());
v2 = tf.get_variable('v2', shape=(), initializer=tf.zeros_initializer());
switch = tf.placeholder(tf.bool)

cond = tf.cond(switch,
    lambda: tf.assign(v1, 1.0), # true
    lambda: tf.assign(v2, 2.0)) # false

with tf.train.MonitoredSession() as sess:
  sess.run(cond, feed_dict={switch: False})
  print(sess.run([v1, v2])) # [0.0, 2.0]

v3 = tf.get_variable('v3', shape=(), initializer=tf.zeros_initializer());
v4 = tf.get_variable('v4', shape=(), initializer=tf.zeros_initializer());
switch = tf.placeholder(tf.bool)
assign_v3 = tf.assign(v3, 1.0)
assign_v4 = tf.assign(v4, 1.0)

# 컨디션을 만들기 위해  필요한 의존성은 먼저 실행된다.
# 즉 assign_v4 와 assign_v4 는 cond 를 만드는대 필요함으로 
# 해당 op 를 만드는 위의 두개의 할당문이 실행된다.
cond = tf.cond(switch,
    lambda: assign_v3,
    lambda: assign_v4)

with tf.train.MonitoredSession() as sess:
  sess.run(cond, feed_dict={switch: False})
  print(sess.run([v3, v4])) # [1.0, 1.0]


