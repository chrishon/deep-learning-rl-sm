import torch
import torch.nn as nn


class DQN_net(nn.Module):

    def __init__(self, output_size=7, h=6, w=7):
        super(DQN_net, self).__init__()
        #  nn.Conv2d(1, 4, (3, 3), padding=(0,0)) ,nn.Tanh(), nn.Flatten(),
        self.net = nn.Sequential(nn.Conv2d(1, 4, (3, 3), padding=(0, 0)), nn.Tanh(), nn.Flatten(-3, -1),
                                 nn.Linear(80, 64),
                                 nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, output_size))

    def forward(self, state, mask=None):
        action_vec = self.net(state)
        return torch.softmax(action_vec, dim=-1) if mask is None else torch.softmax(action_vec + mask, dim=-1)
