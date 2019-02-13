import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_samples, w, b = 20, 0.5, 2

xs = np.asarray(range(num_samples))
ys = np.asarray([x * w + np.random.normal() for x in xs])

class Linear(object):
  def __init__(self):
    self.w = tf.get_variable('w', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())
    self.b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())

  def __call__(self, x):
    return self.w * x + self.b

xtf = tf.placeholder(tf.float32, shape=[num_samples], name='xs')
ytf = tf.placeholder(tf.float32, shape=[num_samples], name='ys')

model = Linear()
model_output = model(xtf)

loss = tf.losses.mean_squared_error(ytf, model_output)
update = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
feed_dict = {xtf: xs, ytf: ys}

with tf.train.MonitoredSession() as sess:
  for i in range(500):
    sess.run(update, feed_dict=feed_dict)
    if i in [1, 5, 25, 125, 499]:
      preds = sess.run(model_output, feed_dict=feed_dict)
      plt.plot(xs, preds, label=str(i))

plt.plot(xs, ys, 'bo')
plt.legend()
plt.show(False)

plt.close()

