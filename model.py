import tensorflow as tf
import tensorflow.contrib.slim as slim

import collections

class Model(object):
  def __init__(self):
    pass

  def arg_scope(self):
    raise NotImplementedError

  def compute(self, inputs):
    raise NotImplementedError

  def compute_training(self, inputs):
    raise NotImplementedError

  def update_target(self, inputs):
    raise NotImplementedError


class MLPModel(Model):
  def __init__(self, config):
    self.config = config
    super(MLPModel, self).__init__()

  def arg_scope(self, reuse=None):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(1e-5),
                        activation_fn=tf.nn.relu,
                        reuse=reuse) as sc:
        return sc

  def compute(self, inputs):
    inputs = tf.reshape(inputs,
        shape=[-1, self.config.INPUT_DIM * self.config.HISTORY_LENGTH])
    outputs = slim.fully_connected(inputs, 100, scope="fc_0")
    outputs = slim.fully_connected(
        outputs, self.config.NUM_ACTIONS, activation_fn=None, scope="softmax")
    return outputs

  def update_target(self):
    update_ops = []
    training_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="training_network")
    target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")
    for training_v, target_v in zip(training_variables, target_variables):
      update_ops.append(target_v.assign(training_v))
    return update_ops


