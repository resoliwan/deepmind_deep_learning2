import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_samples, w, b = 20, 0.5, 2

xs = np.asarray(range(num_samples))
ys = np.asarray([x * w + np.random.normal() for x in xs])

# plt.plot(xs, ys, 'bo')
# plt.show(False)

class Linear(object):
  def __init__(self):
    self.w = tf.get_variable('w', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())
    self.b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())

  def __call__(self, x):
    return self.w * x + self.b

# equation
xtf = tf.placeholder(tf.float32, shape=[num_samples], name='xs')
ytf = tf.placeholder(tf.float32, shape=[num_samples], name='ys')

model = Linear()
model_output = model(xtf)

cov = tf.reduce_sum((xtf - tf.reduce_mean(xtf)) * (ytf - tf.reduce_mean(ytf)))

var = tf.reduce_sum(tf.square(xtf - tf.reduce_mean(xtf)))

w_hat = cov / var
b_hat = tf.reduce_mean(ytf) - w_hat * tf.reduce_mean(xtf)

solve_w = model.w.assign(w_hat)
solve_b = model.b.assign(b_hat)

with tf.train.MonitoredSession() as sess:
  sess.run([solve_w, solve_b], feed_dict={xtf: xs, ytf: ys})
  preds = sess.run(model_output, feed_dict={xtf: xs})

plt.plot(xs, ys, 'bo')
plt.plot(xs, preds)
plt.show(False)
