import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 1000)

r0 = tf.nn.leaky_relu(
    x,
    alpha=0,
    name=None
    )
r1 = tf.nn.leaky_relu(
    x,
    alpha=0.1,
    name=None
    )
r2 = tf.nn.leaky_relu(
    x,
    alpha=0.5,
    name=None
    )
r3 = tf.nn.leaky_relu(
    x,
    alpha=1,
    name=None
    )

with tf.train.MonitoredSession() as sess:
    y0 = sess.run(r0)
    y1 = sess.run(r1)
    y2 = sess.run(r2)
    y3 = sess.run(r3)

plt.plot(x, y0, label='alpha=0')
plt.plot(x, y1, label='alpha=0.1')
plt.plot(x, y2, label='alpha=0.5')
plt.plot(x, y3, label='alpha=1')
plt.title('Leaky relu')
plt.xlabel('x')
plt.ylabel('leaky_relu(x, alpha)')
plt.grid()
plt.legend()
plt.margins(x=0)

plt.show(False)

plt.close()

# plt.axis('off')
# plt.gca().set_position([0, 0, 1, 1])

plt.savefig('../ch_3_nn/3_leaky_relu.svg')
