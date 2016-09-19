class CartPoleConfig(object):
  def __init__(self):
    self.INPUT_DIM = 4

    self.NUM_ACTIONS = 2

    self.NUM_STEPS = 10000

    self.HISTORY_LENGTH = 5

    self.BATCH_SIZE = 10

    self.MEMORY_SIZE = 10000

    self.START_LEARN = 100

    self.TRAIN_FREQUENCY = 4

    self.UPDATE_TARGET_FREQUENCY = 10

    self.GAMMA = 0.97

    self.ALPHA0 = 0.01



