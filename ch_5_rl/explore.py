import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt


def to_normal(mu, sigma, x):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) );


x = np.linspace(-5, 5, 1000)
y = to_normal(1, 0.5, x)

y1 = to_normal(0, 0.9, x)
y2 = to_normal(-1, 1.7, x)


plt.plot(x, y, label="P[q(a_1)]")
plt.plot(x, y1, label="P[q(a_2)]")
plt.plot(x, y2, label="P[q(a_3)]")
plt.title('Explore')
plt.xlabel('expected value')
plt.ylabel('probability density')
plt.grid()
plt.legend()
plt.margins(x=0)
plt.show(False)

plt.close()

# plt.axis('off')
# plt.gca().set_position([0, 0, 1, 1])

plt.savefig('../ch_3_nn/5_explore.svg')
