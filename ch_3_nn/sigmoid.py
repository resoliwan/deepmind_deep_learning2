import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 1000)
r = tf.nn.sigmoid(x)

with tf.train.MonitoredSession() as sess:
  y = sess.run(r)

plt.plot(x, y)
plt.title('Sigmoid')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid()
# plt.legend()
plt.margins(x=0)
plt.show(False)

plt.close()

# plt.axis('off')
# plt.gca().set_position([0, 0, 1, 1])

plt.savefig('../ch_3_nn/3_sigmoid.svg')
