import tensorflow as tf
import tensorflow.contrib.slim as slim

import cPickle
import numpy as np
import os
import random

from experience import Experience

class Game(object):
  def __init__(self, env=None, agent=None, logdir=None, should_render=None, should_load=None):
    self.env = env
    self.agent = agent
    self.config = self.agent.config
    self.logdir = logdir
    self.should_render = should_render
    self.experience = Experience(self.config)

    if should_load:
      self.load()
    else:
      self.step = 0
      self.epsilon = 0.5
      self.total_train_reward = 0.
      self.current_episode = 0

  def save(self):
    with open("save/variables.save", "wb") as w:
      save_list = [self.step, self.epsilon, self.total_train_reward, self.current_episode]
      cPickle.dump(save_list, w, protocol=cPickle.HIGHEST_PROTOCOL)
    self.experience.save()

  def load(self):
    with open("save/variables.save", "rb") as f:
      self.step, self.epsilon, self.total_train_reward, self.current_episode = cPickle.load(f)
    self.experience.load()

  def runEpisode(self, sess, is_training=True):
    # Reset environment.
    state = self.env.reset()

    # Initialize stats.
    num_steps = 0
    total_reward = 0.
    total_loss = 0.

    old_step = self.step
    self.epsilon = max(self.epsilon * 0.99, 0.1) if is_training else 0.1

    for t in xrange(self.config.NUM_STEPS):
      # Render environment.
      if self.should_render:
        self.env.render()
      # Agent acts according to the current action values.
      queried_state = self.experience.getLastestState()
      random_action = -1
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
    print "Episode", self.current_episode, "has finshed in", num_steps, "steps"
    avg_loss = total_loss / num_steps
    print  "    Reward: {:10}     Loss: {:.6f}".format(total_reward, avg_loss)

    return total_reward

  def train(self, num_episodes):
    sv = tf.train.Supervisor(logdir=self.logdir, save_model_secs=50)
    with sv.managed_session('') as sess:
      for i in xrange(num_episodes):
        self.total_train_reward += self.runEpisode(sess)
        if self.current_episode % 20 == 0:
          self.save()
        self.current_episode += 1
        print "Running average reward:", self.total_train_reward / self.current_episode
      sv.saver.save(sess, self.logdir, global_step=sv.global_step)
      self.save()

  def play(self, num_episodes):
    sv = tf.train.Supervisor(logdir=self.logdir)
    with sv.managed_session('') as sess:
      total_reward = 0.
      for i in xrange(num_episodes):
        total_reward += self.runEpisode(sess, is_training=False)
        print "Running average reward: " + str(total_reward / (i + 1))
