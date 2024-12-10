from collections import deque, namedtuple
import torchvision.transforms as T
import random
from PIL import Image

Transition = namedtuple('Transition',
                        ('state', 'action', 'action_mask', 'done', 'next_state', 'next_action_mask', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
