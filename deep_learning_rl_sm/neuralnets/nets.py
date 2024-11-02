import torch
import torch.nn as nn
import numpy as np
from torch import distributions as pyd


class Actor(nn.Module):
    def __init__(self, a_dim, h_dim):
        super().__init__()
        self.mu = nn.Linear(h_dim, a_dim)
        # (FROM PAPER) self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.log_std = nn.Linear(h_dim, a_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        action_mean = torch.tanh(self.mu(x))
        action_std = torch.exp(self.log_std(x).clamp(self.log_std_min, self.log_std_max))
        return pyd.Normal(action_mean, action_std)



