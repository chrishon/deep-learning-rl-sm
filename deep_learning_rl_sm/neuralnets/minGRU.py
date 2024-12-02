import torch
import torch.nn.functional as F
from torch.nn import Linear, Module


def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


def parallel_scan_log(log_coefficients, log_values):
    # log_coefficients: (batch_size, seq_len, input_size)
    # log_values: (batch_size, seq_len + 1, input_size)
    a_star = F.pad(torch.cumsum(log_coefficients, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]


class minGRU(Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim=dim
        self.f_hidden = Linear(dim, dim, bias=False)
        self.f_z = Linear(dim, dim, bias=False)
        # output of f_z can be viewed as the proportion of the info from the current timestep that is incorporated into
        # the next hidden state (for more info see paper "Were RNNs All We Needed?")

        # This code is also available in the paper "Were RNNs All We Needed?"
        # We could still change the code to our specifications for this project
        # however original code is already written extremely cleanly and i see no reason to
        # change names, rearrange code etc. for the sake of it

        """
        Note: This version is not in log-space
        Sequential Algorithm for minGRU:
                z_t ← σ(f_z(x_t))
                h˜_t ← g(f_h(x_t))
                h_t ← (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h˜_t
        """

        # We use the log-space version of the algorithm for additional numerical stability
        # (i.e. long sequences more likely to result in numerical underflow)
        # by e.g. converting to log-values, summing and then exponentiating we achieve the same result as
        # multiplying the original values but with better numerical stability

    def forward(self, x, h_prev=None):
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        if h_prev is None:
            h_prev = torch.zeros(x.shape[0], 1, self.dim)
        k = self.f_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_prev = log_g(h_prev)
        log_tilde_h = log_g(self.f_hidden(x))
        h = parallel_scan_log(log_coeffs, torch.cat([log_h_prev, log_z + log_tilde_h], dim=1))
        return h


# Test
# TODO how do we only predict next action & reward like REINFORMER instead of action reward & state (full transition)
