import tensorflow as tf

import config
import cPickle
import gym
import os

from agent import Agent
from game import Game
from model import MLPModel

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean("load", True, "Whether to load a previous session.")

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  graph = tf.Graph()
  with graph.as_default():
    env = gym.make("CartPole-v0")
    model = MLPModel(config=config.CartPoleConfig())
    agent = Agent(model=model)

    logdir = "/tmp/cart_pole"
    if not os.path.isdir(logdir):
      os.makedirs(logdir)

    game = Game(env=env, agent=agent, logdir=logdir, should_render=True,
        should_load=FLAGS.load)
    game.train(1000)
    #game.play(50)


if __name__ == "__main__":
  tf.app.run()
