from collections import deque, namedtuple
import torchvision.transforms as T
import random
from PIL import Image

Transition = namedtuple('Transition',
                        ('state', 'state_additional', 'action', 'time', 'done', 'next_state',
                         'next_state_additional', 'reward'))


class resizer:
    def __init__(self, width_and_height_size):
        self.resize = T.Compose([T.ToPILImage(),
                                 T.Resize(width_and_height_size, interpolation=Image.CUBIC),
                                 T.ToTensor()])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
