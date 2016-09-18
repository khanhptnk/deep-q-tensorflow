import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import random

import config

class Agent(object):
  def __init__(self, model=None, action_space=None, observation_space=None, alpha=0.1, gamma=0.8):
    self.model = model
    self.action_space = action_space
    self.num_actions = action_space.n
    self.observation_space = observation_space
    self.input_len = observation_space.shape[0] * config.NUM_FRAMES
    # TODO: make sure that environment parameters are unknown to the agent.
    self.gamma = gamma

    # Current state
    self.s = tf.placeholder(tf.float32, shape=[None, self.input_len])
    with slim.arg_scope(self.model.arg_scope()):
      self.q_s = self.model.compute(self.s)
    self.a = tf.placeholder(tf.int32, shape=[None, 1])
    one_hot_a = tf.one_hot(self.a, self.num_actions, on_value=1, off_value=0)
    one_hot_a = tf.reshape(tf.cast(one_hot_a, tf.float32), shape=[-1, self.num_actions])
    self.q_s_a = tf.reduce_sum(tf.mul(self.q_s, one_hot_a))
    self.best_a = tf.argmax(self.q_s, dimension=1)

    # Next state
    self.s_prime = tf.placeholder(tf.float32, shape=[None, self.input_len])
    with slim.arg_scope(self.model.arg_scope()):
      self.q_s_prime = self.model.compute(self.s_prime, reuse=True)
    self.a_prime = tf.argmax(self.q_s_prime, dimension=1)
    one_hot_a_prime = tf.one_hot(self.a_prime, self.num_actions, on_value=1, off_value=0)
    one_hot_a_prime = tf.reshape(tf.cast(one_hot_a_prime, tf.float32), shape=[-1, self.num_actions])
    self.q_s_prime_a_prime = tf.reduce_sum(tf.mul(self.q_s_prime, one_hot_a_prime))

    # Loss
    self.r = tf.placeholder(tf.float32, shape=[None, 1])
    self.isTerminal = tf.placeholder(tf.float32, shape=[None, 1])
    delta = self.r + (1 - self.isTerminal) * self.gamma * self.q_s_prime_a_prime - self.q_s_a
    self.loss = 0.5 * tf.reduce_mean(tf.square(delta))

    # Optimizer
    self.global_step = slim.get_or_create_global_step()
    self.train_op = tf.train.GradientDescentOptimizer(alpha).minimize(self.loss, global_step=self.global_step)

  def act(self, sess, state, epsilon):
    state = state.reshape(1, self.input_len)
    rand_num = random.random()
    if rand_num < epsilon:
      return self.action_space.sample()
    dist, best_action = sess.run([self.q_s, self.best_a], feed_dict={self.s : state})
    if sum(state.ravel().tolist()) == self.input_len:
      print dist
    return best_action[0]

  def observe(self, sess, state, action, next_state, reward, isTerminal):
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.s : state,
                                                 self.a : action,
                                                 self.s_prime : next_state,
                                                 self.r : reward,
                                                 self.isTerminal : isTerminal})
    return loss
