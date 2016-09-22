import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

import collections
import random

class Agent(object):
  def __init__(self, model=None):
    self.model = model
    self.config = self.model.config

    self.s = tf.placeholder(tf.float32, shape=[None,
                                               self.config.INPUT_DIM,
                                               self.config.HISTORY_LENGTH])
    self.a = tf.placeholder(tf.int32, shape=[None,])
    self.r = tf.placeholder(tf.float32, shape=[None,])
    self.is_terminal = tf.placeholder(tf.float32, shape=[None,])

    with tf.name_scope("compute_q_s_prime_a_prime"):
      self.compute_target_network_op = self.computeTargetNetwork()
    with tf.name_scope("q_learning"):
      self.loss_op, self.train_op = self.qLearning()
    with tf.name_scope("best_action"):
      self.best_action_op = self.bestAction()
    with tf.name_scope("update_target_network"):
      self.update_target_ops = self.model.update_target()

  def bestAction(self):
    with tf.variable_scope("training_network"):
      with slim.arg_scope(self.model.arg_scope(reuse=True)):
        q_s = self.model.compute(self.s)
        return tf.argmax(q_s, dimension=1)

  def qLearning(self):
    # Compute Q(s, a).
    with tf.name_scope("compute_q_s_a"):
      q_s_a = self.computeTrainingNetwork()

    # Placeholder for max_a' Q(s', a').
    self.q_s_prime_a_prime = tf.placeholder(tf.float32,
        shape=[self.config.BATCH_SIZE])

    # Loss
    delta = self.r + (1. - self.is_terminal) * self.config.GAMMA * self.q_s_prime_a_prime - q_s_a
    loss = tf.reduce_mean(tf.square(delta))

    # Optimizer
    global_step=slim.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer(self.config.ALPHA0).minimize(loss,
        global_step=global_step)

    return loss, train_op

  def computeTargetNetwork(self):
    with tf.variable_scope("target_network"):
      with slim.arg_scope(self.model.arg_scope(reuse=False)):
        q_s_prime = self.model.compute(self.s)
        a_prime = tf.argmax(q_s_prime, dimension=1)
        return self.computeQValue(q_s_prime, a_prime)


  def computeTrainingNetwork(self):
    with tf.variable_scope("training_network"):
      with slim.arg_scope(self.model.arg_scope(reuse=False)):
        q_s = self.model.compute(self.s)
        return self.computeQValue(q_s, self.a)

  def computeQValue(self, q_s, a):
    one_hot_a = tf.one_hot(a, self.config.NUM_ACTIONS, on_value=1.,
        off_value=0., axis=-1, dtype=tf.float32)
    q_s_a = tf.reduce_sum(tf.mul(q_s, one_hot_a), reduction_indices=1)
    return q_s_a


  def plan(self, sess, state, random_action, epsilon):
    state = state.reshape(1, self.config.INPUT_DIM, self.config.HISTORY_LENGTH)
    best_action = sess.run(self.best_action_op, feed_dict={self.s : state})[0]
    return random_action if random.random() < epsilon else best_action

  def observe(self, sess, states, actions, next_states, rewards, is_terminals):
    q_s_prime_a_prime = sess.run(self.compute_target_network_op,
        feed_dict = {self.s : next_states})

    loss, _ = sess.run([self.loss_op, self.train_op],
            feed_dict={ self.s : states,
                        self.a : actions,
                        self.q_s_prime_a_prime : q_s_prime_a_prime,
                        self.r : rewards,
                        self.is_terminal : is_terminals })
    return loss

  def update_target(self, sess):
    for op in self.update_target_ops:
      sess.run(op)
