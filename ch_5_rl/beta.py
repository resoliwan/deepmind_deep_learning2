import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats as ss


f, (ax1, ax2, ax3, ax4, ax5)= plt.subplots(1, 5, sharey='row', figsize=(13, 4))
f.suptitle("Beta distribution")
x1 = np.linspace(0, 1, 100)
y1 = ss.beta.pdf(x1, a=1, b=1)
ax1.set_title('1) a=1,b=1')
ax1.plot(x1, y1)
x2 = np.linspace(0, 1, 100)
y2 = ss.beta.pdf(x2, a=1, b=2)
ax2.set_title('2) a=1,b=2')
ax2.plot(x2, y2)
x3 = np.linspace(0, 1, 100)
y3 = ss.beta.pdf(x3, a=1, b=3)
ax3.set_title('3) a=1,b=3')
ax3.plot(x3, y3)
x4 = np.linspace(0, 1, 100)
y4 = ss.beta.pdf(x4, a=2, b=3)
ax4.set_title('4) a=2,b=3')
ax4.plot(x4, y4)
x5 = np.linspace(0, 1, 100)
y5 = ss.beta.pdf(x5, a=3, b=3)
ax5.set_title('5) a=3,b=3')
ax5.plot(x5, y5)

plt.show(False)

plt.close()

# plt.axis('off')
# plt.gca().set_position([0, 0, 1, 1])

plt.savefig('./ch_5_rl/5_beta.svg')
