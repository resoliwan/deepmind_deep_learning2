import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.get_variable('x', shape=(), initializer=tf.zeros_initializer())
assign_x = tf.assign(x, 10.0)
# z = tf.assign(x, 10.0) + 1.0
z = x + 1.0

with tf.train.MonitoredSession() as sess:
  print(sess.run(z))
  # 결과 값은 1.0 and assing_x 는 z 에 의존성이 없음으로 실행 안됨 

with tf.train.MonitoredSession() as sess:
  print(sess.run([assign_x, z]))
  # 텐서 플로우는 fetchs 리스트에 인덱스 순서대로 실행하지 않음!
  # 결과 값은 (10.0, 1.0) 또는 (10.0, 11.0) 이됨 assign_x 와 z 가 레이스 컨디션이 됨.

with tf.control_dependencies([assign_x]):
  z = x + 1.0
with tf.train.MonitoredSession() as sess:
  print(sess.run(z))
  # z 를 assign_x 에 의존 하도록 설정했기 때문에 z 를 실행하면 assign_x 가 먼저 실행됨. 11.0
