import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

import collections
import random

class Agent(object):
  def __init__(self, model=None):
    self.model = model
    self.config = self.model.config
    self.compute_target_network_op = self.createComputeTargetNetWorkOp()
    self.loss, self.train_op = self.createQLearningOps()
    self.best_action_op = self.createBestActionOp()
    self.update_target_ops = self.model.update_target()

  def createBestActionOp(self):
    self.random_action = tf.placeholder(tf.int64)
    self.s_for_best_action = tf.placeholder(
        tf.float32, shape=[1, self.config.INPUT_DIM, self.config.HISTORY_LENGTH])
    with slim.arg_scope(self.model.arg_scope(reuse=True, is_training=False)):
      q_s_for_best_action = self.model.compute_training(self.s_for_best_action)
    self.q_s_for_best_action = tf.reshape(q_s_for_best_action, shape=[-1])
    best_action = tf.argmax(self.q_s_for_best_action, dimension=0)
    return tf.cond(tf.equal(self.random_action, tf.constant(-1, dtype=tf.int64)),
        lambda: best_action,
        lambda: self.random_action)

  def createComputeTargetNetWorkOp(self):
    # Compute max_a' Q(s', a') from the target network.
    self.s_prime = tf.placeholder(tf.float32, shape=[self.config.BATCH_SIZE,
                                                     self.config.INPUT_DIM,
                                                     self.config.HISTORY_LENGTH])
    with slim.arg_scope(self.model.arg_scope(reuse=False, is_training=False)):
      q_s_prime = self.model.compute_target(self.s_prime)
    a_prime = tf.argmax(q_s_prime, dimension=1)
    one_hot_a_prime = tf.one_hot(a_prime, self.config.NUM_ACTIONS, on_value=1.,
        off_value=0., axis=-1, dtype=tf.float32)
    q_s_prime_a_prime = tf.reduce_sum(tf.mul(q_s_prime, one_hot_a_prime),
        reduction_indices=1)
    return q_s_prime_a_prime


  def createQLearningOps(self):
    # Compute Q(s, a) from the training network.
    self.s = tf.placeholder(tf.float32, shape=[self.config.BATCH_SIZE,
                                               self.config.INPUT_DIM,
                                               self.config.HISTORY_LENGTH])
    with slim.arg_scope(self.model.arg_scope(reuse=False, is_training=True)):
      q_s = self.model.compute_training(self.s)
    self.a = tf.placeholder(tf.int32, shape=[self.config.BATCH_SIZE])
    one_hot_a = tf.one_hot(self.a, self.config.NUM_ACTIONS, on_value=1.,
        off_value=0., axis=-1, dtype=tf.float32)
    q_s_a = tf.reduce_sum(tf.mul(q_s, one_hot_a), reduction_indices=1)
    # Placeholder for max_a' Q(s', a').
    self.q_s_prime_a_prime = tf.placeholder(
        tf.float32, shape=[self.config.BATCH_SIZE])

    # Loss
    self.r = tf.placeholder(tf.float32, shape=[self.config.BATCH_SIZE])
    self.is_terminal = tf.placeholder(tf.float32, shape=[self.config.BATCH_SIZE])
    delta = self.r + (1. - self.is_terminal) * self.config.GAMMA * self.q_s_prime_a_prime - q_s_a
    loss = 0.5 * tf.reduce_mean(tf.square(delta))

    # Optimizer
    global_step = slim.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer(self.config.ALPHA0).minimize(loss,
        global_step=global_step)

    return loss, train_op


  def act(self, sess, state, random_action):
    state = state.reshape(1, self.config.INPUT_DIM, self.config.HISTORY_LENGTH)
    dist, best_action = sess.run([self.q_s_for_best_action, self.best_action_op],
        feed_dict={ self.s_for_best_action : state,
                    self.random_action : random_action})
    return best_action

  def observe(self, sess, state, action, next_state, reward, is_terminal):
    q_s_prime_a_prime = sess.run(self.compute_target_network_op,
            feed_dict = { self.s_prime : next_state })

    loss, _ = sess.run([self.loss, self.train_op],
            feed_dict={ self.s : state,
                        self.a : action,
                        self.q_s_prime_a_prime : q_s_prime_a_prime,
                        self.r : reward,
                        self.is_terminal : is_terminal })
    return loss

  def update_target(self, sess):
    for op in self.update_target_ops:
      sess.run(op)
