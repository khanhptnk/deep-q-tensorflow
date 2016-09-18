import tensorflow as tf
import tensorflow.contrib.slim as slim

import collections

class Model(object):
  def __init__(self):
    pass

  def arg_scope(self):
    raise NotImplementedError

  def compute_target(self, inputs):
    raise NotImplementedError

  def compute_training(self, inputs):
    raise NotImplementedError

  def update_target(self, inputs):
    raise NotImplementedError


class LinearModel(Model):
  def __init__(self, config):
    self.config = config
    self.scopes = ["training_fc_0",
                   "training_fc_1",
                   "training_fc_2",
                   "training_fc_3",
                   "training_fc_4"]
    super(LinearModel, self).__init__()

  def arg_scope(self, reuse=None, is_training=None):
    batch_norm_params = {
      "is_training" : is_training,
      "decay" : 0.9997,
      "epsilon" : 0.001,
      "variables_collections" : {
        "beta" : None,
        "gamma" : None,
        "moving_mean" : ["moving_vars"],
        "moving_variance" : ["moving_vars"]
      }
    }
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(1e-5),
                        activation_fn=tf.nn.relu,
                        #normalizer_fn=slim.batch_norm,
                        #normalizer_params=batch_norm_params,
                        reuse=reuse) as sc:
        return sc

  def compute_target(self, inputs):
    inputs = tf.reshape(inputs,
        shape=[-1, self.config.INPUT_DIM * self.config.HISTORY_LENGTH])
    outputs = slim.fully_connected(inputs, 100, scope="target_fc_0")
    outputs = slim.fully_connected(outputs, 200, scope="target_fc_1")
    outputs = slim.fully_connected(outputs, 200, scope="target_fc_2")
    outputs = slim.fully_connected(outputs, 100, scope="target_fc_3")
    outputs = slim.fully_connected(
        outputs, self.config.NUM_ACTIONS, activation_fn=None, scope="target_fc_4")
    return outputs

  def compute_training(self, inputs):
    inputs = tf.reshape(inputs,
        shape=[-1, self.config.INPUT_DIM * self.config.HISTORY_LENGTH])
    outputs = slim.fully_connected(inputs, 100, scope=self.scopes[0])
    outputs = slim.fully_connected(outputs, 200, scope=self.scopes[1])
    outputs = slim.fully_connected(outputs, 200, scope=self.scopes[2])
    outputs = slim.fully_connected(outputs, 100, scope=self.scopes[3])
    outputs = slim.fully_connected(
        outputs, self.config.NUM_ACTIONS, activation_fn=None, scope=self.scopes[4])
    return outputs

  def update_target(self):
    assign_ops = []
    for training_scope in self.scopes:
      with tf.variable_scope(training_scope, reuse=True):
        target_scope = training_scope.replace("training", "target")
        for target_v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope):
          name = "/".join(target_v.name.split("/")[1:]).split(":")[0]
          training_v = tf.get_variable(name)
          assign_ops.append(target_v.assign(training_v))
    return assign_ops

