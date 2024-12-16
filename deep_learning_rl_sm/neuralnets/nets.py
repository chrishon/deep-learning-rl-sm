import torch
import torch.nn as nn
import numpy as np
from torch import distributions as torch_dist


class Actor(nn.Module):
    def __init__(self, num_actions, h_dim, device, discrete=False):
        super().__init__()
        self.mu = nn.Linear(h_dim, num_actions, device=device)
        # (FROM PAPER) self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.log_std = nn.Linear(h_dim, num_actions, device=device)
        self.log_std_min = -20
        self.log_std_max = 2
        self.categorical_dist = discrete

    def forward(self, x):
        action_mean = torch.softmax(self.mu(x), dim=-1) if self.categorical_dist else torch.tanh(self.mu(x))
        action_std = None if self.categorical_dist else torch.exp(
            self.log_std(x).clamp(self.log_std_min, self.log_std_max))
        return torch_dist.Categorical(action_mean) if self.categorical_dist else torch_dist.Normal(action_mean,
                                                                                                   action_std)
        # return torch_dist.Normal(action_mean, action_std)
