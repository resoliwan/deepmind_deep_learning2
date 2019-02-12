import numpy as np
import matplotlib.pyplot as plt
num_samples, w, b = 20, 0.5, 2

xs = np.asarray(range(num_samples))
ys = np.asarray([x * w + np.random.normal() for x in xs])

plt.plot(xs, ys, 'bo')
plt.show(False)
