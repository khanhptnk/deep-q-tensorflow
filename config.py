class Config(object):
  def __init__(self):
    self.NUM_STEPS = 10000
    self.HISTORY_LENGTH = 1
    self.BATCH_SIZE = 10
    self.MEMORY_SIZE = 10000
    self.START_LEARN = 100
    self.TRAIN_FREQUENCY = 4
    self.UPDATE_TARGET_FREQUENCY = 100


class CartPoleConfig(Config):
  def __init__(self):
    self.INPUT_DIM = 4
    self.NUM_ACTIONS = 2
    self.GAMMA = 0.97
    self.ALPHA0 = 0.01
    self.TEST_EPSILON = 0.01
    super(CartPoleConfig, self).__init__()

class MountainCarConfig(Config):
  def __init__(self):
    self.INPUT_DIM = 2
    self.NUM_ACTIONS = 3
    self.GAMMA = 0.97
    self.ALPHA0 = 0.001
    self.TEST_EPSILON = 0.1
    self.BATCH_SIZE = 32
    super(MountainCarConfig, self).__init__()

