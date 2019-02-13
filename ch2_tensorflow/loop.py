import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

k = tf.constant(2)
matrix = tf.ones([2, 2])
condition = lambda i, _: i < k
body = lambda i, m: (i+1, tf.matmul(m, matrix))
final_i, power = tf.while_loop(
    cond=condition,
    body=body,
    loop_vars=(0, tf.diag([1., 1.]))
    )
# loop 1
# i = 0 => i = 1
# m         *   matrix => m
# [[1, 0]      [[1, 1]    [[1, 1]
# [0, 1]]   *   [1, 1]] =>  [1, 1]]

# loop 2
# i = 1 => i = 2
# m         *   matrix => m
# [[1, 1]      [[1, 1]    [[2, 2]
# [1, 1]]   *   [1, 1]] =>  [2, 2]]

# i < 2 break;

with tf.train.MonitoredSession() as sess:
  print(sess.run([final_i, power])) # [2, [[2, 2], [2, 2]]]
