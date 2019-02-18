import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 1000)
r0 = tf.nn.relu(x)

with tf.train.MonitoredSession() as sess:
  y0 = sess.run(r0)

plt.plot(x, y0)
plt.title('Relu')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.margins(x=0)
plt.grid()
# plt.legend()

plt.show(False)

plt.close()

# plt.axis('off')
# plt.gca().set_position([0, 0, 1, 1])

plt.savefig('../ch_3_nn/3_relu.svg')
