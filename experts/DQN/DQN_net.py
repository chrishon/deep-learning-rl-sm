import torch
import torch.nn as nn


class DQN_net(nn.Module):

    def __init__(self, output_size=7, h=6, w=7):
        super(DQN_net, self).__init__()
        self.flatten = nn.Flatten()
        in_size = h * w
        self.net = nn.Sequential(nn.Linear(in_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, output_size))

    def forward(self, state, mask=None):
        action_vec = self.net(state)
        return torch.softmax(action_vec, dim=-1) if mask is None else torch.softmax(action_vec + mask, dim=-1)
