import tensorflow as tf

import config
import gym
import os

from agent import Agent
from game import Game
from model import LinearModel

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  graph = tf.Graph()
  with graph.as_default():
    env = gym.make("CartPole-v0")
    model = LinearModel(config=config)
    agent = Agent(model=model, config=config)
    game = Game(env=env, agent=agent, config=config, should_render=True)

    logdir = "/tmp/cart_pole"
    if not os.path.isdir(logdir):
      os.makedirs(logdir)

    game.train(10000, logdir)
    game.play(50, logdir)


if __name__ == "__main__":
  tf.app.run()
