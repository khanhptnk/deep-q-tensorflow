import cPickle
import numpy as np
import os
import random

class Experience(object):
  def __init__(self, config):
    self.config = config
    self.state_mem = np.zeros((config.MEMORY_SIZE, config.INPUT_DIM))
    self.action_mem = np.zeros(config.MEMORY_SIZE)
    self.reward_mem = np.zeros(config.MEMORY_SIZE)
    self.is_terminal_mem = np.zeros(config.MEMORY_SIZE)
    self.mem_index = 0
    self.count = 0

    self.batch_states = np.zeros((self.config.BATCH_SIZE, self.config.INPUT_DIM,
        self.config.HISTORY_LENGTH))
    self.batch_next_states = np.zeros((self.config.BATCH_SIZE, self.config.INPUT_DIM,
        self.config.HISTORY_LENGTH))

  def add(self, state, action, next_state, reward, is_terminal):
    self.state_mem[self.mem_index, ...] = state
    self.action_mem[self.mem_index] = action
    self.reward_mem[self.mem_index] = reward
    self.is_terminal_mem[self.mem_index] = is_terminal
    self.count = max(self.count, self.mem_index + 1)
    self.mem_index = (self.mem_index + 1) % self.config.MEMORY_SIZE

  def getState(self, index):
    index %= self.config.MEMORY_SIZE
    low = index - self.config.HISTORY_LENGTH + 1
    if low >= 0:
      return self.state_mem[low : (index + 1), ...].transpose()
    indexes = [(index - i) % self.count for i in reversed(xrange(self.config.HISTORY_LENGTH))]
    return self.state_mem[indexes, ...].transpose()

  def getLastestState(self):
    return self.getState(self.mem_index - 1)

  def sample(self):
    assert self.count > self.config.HISTORY_LENGTH
    indexes = []

    while len(indexes) < self.config.BATCH_SIZE:
      while True:
        index = random.randint(self.config.HISTORY_LENGTH, self.count - 1)
        # If the history wraps around the current memory index, pick another one.
        if index >= self.mem_index and index - self.config.HISTORY_LENGTH < self.mem_index:
          continue
        # If the history contains two episodes, pick another one.
        if self.is_terminal_mem[(index - self.config.HISTORY_LENGTH) : index].any():
          continue
        break
      self.batch_states[len(indexes), ...] = self.getState(index - 1)
      self.batch_next_states[len(indexes), ...] = self.getState(index)
      indexes.append(index)
    batch_actions = self.action_mem[indexes]
    batch_rewards = self.reward_mem[indexes]
    batch_is_terminal = self.is_terminal_mem[indexes]

    return (self.batch_states, batch_actions, self.batch_next_states,
        batch_rewards, batch_is_terminal)

  def save(self):
    for _, (name, array) in enumerate(
        zip(["state_mem", "action_mem", "reward_mem", "is_terminal_mem"],
            [self.state_mem, self.action_mem, self.reward_mem, self.is_terminal_mem])):
      np.save(name, array)
    with open("experience_variables.save", "wb") as w:
      cPickle.dump([self.count, self.mem_index], w, protocol=cPickle.HIGHEST_PROTOCOL)

  def load(self):
    for _, (name, array) in enumerate(
        zip(["state_mem", "action_mem", "reward_mem", "is_terminal_mem"],
            [self.state_mem, self.action_mem, self.reward_mem, self.is_terminal_mem])):
      array = np.load(name + ".npy")
    with open("experience_variables.save", "rb") as f:
      self.count, self.mem_index = cPickle.load(f)


