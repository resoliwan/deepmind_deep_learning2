import numpy as np
import tensorflow as tf

# fibonacci
def fibonacci(state):
  output = state[0] + state[1]
  return output, (state[1], output)

state = (0, 1) # (f_{0}, f_{1})
for item in range(10):
  output, state = fibonacci(state)
  #f_n, (f_{n-2}, f_{n-1})
  print(output)


# tensorflow implementation
class fibonacci_core(object):
  def __init__(self):
    self.output_size = 1
    self.state_size = tf.TensorShape([1,1])

  def __call__(self, input, state):
    output = state[0] + state[1]
    return output, (state[1], output)

  def zero_state(self, batch_size, dtype):
    return (tf.zeros((batch_size, 1), dtype=dtype),
        tf.ones((batch_size, 1), dtype=dtype))

  def initial_state(self, batch_size, dtype):
    return zero_state(self, batch_size, dtype)

inputs = tf.reshape(tf.range(10), [10, 1, 1])

fib_seq = tf.nn.dynamic_rnn(
    cell=fibonacci_core(),
    inputs=inputs,
    dtype=tf.float32,
    time_major=True)

with tf.train.MonitoredSession() as sess:
  print(sess.run(fib_seq)) # (1, 2, 3....)


