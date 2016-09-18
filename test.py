import tensorflow as tf

import numpy as np

import random

with tf.Graph().as_default():
  with tf.variable_scope("something"):
    x = tf.get_variable("x", shape=[1, 2])
    y = tf.get_variable("y", shape=[1, 2])

  with tf.variable_scope("some"):
    x = tf.get_variable("x", shape=[1, 2])
    y = tf.get_variable("y", shape=[1, 2])

  assign_ops = []
  with tf.variable_scope("some", reuse=True):
    for target_v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="something"):
      name = target_v.name.split("/")[-1].split(":")[0]
      training_v = tf.get_variable(name)
      assign_ops.append(training_v.assign(target_v))


  sv = tf.train.Supervisor(logdir="/tmp/something")
  with sv.managed_session('', start_standard_services=False) as sess:
    for op in assign_ops:
      print "Ha ha " + str(sess.run(op))

"""x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  #print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
  """

