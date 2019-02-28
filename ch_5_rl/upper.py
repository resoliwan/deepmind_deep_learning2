import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt


def bonus(n, p):
    return np.sqrt( (-np.log(p) / (2*n)) )


def to_normal(mu, sigma, x):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) );


def hoefffding_inequality(mu, n, u):
    return np.exp(-2*n*(u**2))


hoefffding_inequality(0.5, 10, 1)

hoefffding_inequality(0.5, 10, 1E-9)

hoefffding_inequality(0.5, 10000000000000000, 1E-9)




# x = np.linspace(-1, 1, 1000)
# y = hoefffding_inequality(0.5, 10, x)
x = np.linspace(1, 1E-9, 1000)
y = bonus(100, x)

# y1 = to_normal(0, 0.9, x)
# y2 = to_normal(-1, 1.7, x)


plt.plot(x, y)
# plt.plot(x, y1, label="P[q(a_2)]")
# plt.plot(x, y2, label="P[q(a_3)]")
plt.title('Upper Confidence Bound')
plt.xlabel('p')
plt.ylabel('U_t(a)')
plt.grid()
plt.legend()
plt.margins(x=0)
plt.show(False)

plt.close()

plt.savefig('./ch_5_rl/5_upper_u.svg')

# x = np.linspace(0, 100, 100)
# y = np.exp(x)
x = np.linspace(0, 1E+20, 10)
y = hoefffding_inequality(0.5, x, 1E-10)
plt.plot(x, y)
# plt.plot(x, y1, label="P[q(a_2)]")
# plt.plot(x, y2, label="P[q(a_3)]")
plt.title('Upper Confidence Bound')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ticklabel_format(useOffset=False)
plt.xlabel('N_t(a)')
plt.ylabel('P(q(a) >= 0.5)')
plt.grid()
# plt.legend()
# plt.margins(x=0)

plt.show(False)

plt.close()

plt.savefig('./ch_5_rl/5_upper_n.svg')





bonus(1, 0.5)
bonus(100, 0.5)


x = np.arange(1, 1000, 1)
y = bonus(x, 0.5)

# y1 = to_normal(0, 0.9, x)
# y2 = to_normal(-1, 1.7, x)


plt.plot(x, y)
# plt.plot(x, y1, label="P[q(a_2)]")
# plt.plot(x, y2, label="P[q(a_3)]")
plt.title('Upper Confidence Bound')
plt.xlabel('N_t(a)')
plt.ylabel('U_t(a)')
plt.grid()
plt.legend()
plt.margins(x=0)

plt.show(False)

plt.close()

plt.savefig('./ch_5_rl/5_upper_n.svg')
