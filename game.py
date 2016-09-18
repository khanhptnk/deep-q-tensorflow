import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np
import random

from experience import Experience

class Game(object):
  def __init__(self, env=None, agent=None, config=None, should_render=None):
    self.env = env
    self.agent = agent
    self.config = config
    self.should_render = should_render
    self.experience = Experience(config)
    self.step = 0
    self.epsilon = 0.5

  def runEpisode(self, sess, index, is_training=True):
    # Reset environment.
    state = self.env.reset()

    # Initialize stats.
    num_steps = 0
    total_reward = 0.
    total_loss = 0.

    old_step = self.step
    self.epsilon = self.epsilon * 0.99 if is_training else 0.

    for t in xrange(self.config.NUM_STEPS):
      # Render environment.
      if self.should_render:
        self.env.render()
      # Agent acts according to the current action values.
      queried_state = self.experience.getLastestState()
      random_action = -1
      if is_training:
        if random.random() < self.epsilon:
          random_action = self.env.action_space.sample()
      action = self.agent.act(sess, queried_state, random_action=random_action)
      # Environment executes the action.
      next_state, reward, done, info = self.env.step(action)
      total_reward += reward

      # Add observation to experience.
      self.experience.add(state, action, next_state, reward, 1 if done else 0)

      # If training, agent updates parameters according to the observation.
      if is_training and self.step > self.config.START_LEARN:
        # Update parameters of the current training network.
        if self.step % self.config.TRAIN_FREQUENCY == 0:
          (batch_states, batch_actions, batch_next_states, batch_rewards,
              batch_is_terminal) = self.experience.sample()
          total_loss += self.agent.observe(
              sess, batch_states, batch_actions, batch_next_states,
              batch_rewards, batch_is_terminal)
        # Update parameters of the target network
        if self.step % self.config.UPDATE_TARGET_FREQUENCY == 0:
          self.agent.update_target(sess)
      # End of episode.
      if done:
        break
      # Update state
      state = next_state
      # Increment time.
      self.step += 1

    # Message at the end of the episode.
    num_steps = self.step - old_step + 1
    print "Episode", index, "has finshed in", num_steps, "steps"
    avg_loss = total_loss / num_steps
    print  "    Reward: {:10}     Loss: {:.6f}".format(total_reward, avg_loss)

    return total_reward

  def train(self, num_episodes, logdir):
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=50)
    with sv.managed_session('') as sess:
      total_reward = 0.
      for i in xrange(num_episodes):
        total_reward += self.runEpisode(sess, i)
        print "Running average reward: " + str(total_reward / (i + 1))
      #for i in xrange(num_episodes):
      #  total_reward += self.runEpisode(sess, i, is_training=False)
      #  print "Running average reward: " + str(total_reward / (i + 1))
      #sv.saver.save(sess, logdir, global_step=sv.global_step)

  def play(self, num_episodes, logdir):
    sv = tf.train.Supervisor(logdir=logdir)
    with sv.managed_session('') as sess:
      total_reward = 0.
      for i in xrange(num_episodes):
        total_reward += self.runEpisode(sess, i, is_training=False)
        print "Running average reward: " + str(total_reward / (i + 1))
