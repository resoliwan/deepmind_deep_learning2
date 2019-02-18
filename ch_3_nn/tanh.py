import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 1000)
y = np.tanh(x)

plt.plot(x, y, label='1')
plt.grid()
# plt.legend()
plt.title('Tanh')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.margins(x=0)

plt.show(False)

plt.close()

# plt.axis('off')
# plt.gca().set_position([0, 0, 1, 1])
plt.savefig('../ch_3_nn/3_tanh.svg')
