#Source: https://github.com/YecanLee/min-LSTM-torch/blob/main/minLSTMcell.py

import torch
import torch.nn.functional as F
import torch.nn as nn

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

class minLSTM(nn.Module):
    def __init__(self, dim):
        super(minLSTM, self).__init__()
        self.dim = dim
        #self.input_shape = input_shape
        
        # Initialize the linear layers for the forget gate, input gate, and hidden state transformation
        self.linear_f = nn.Linear(self.dim, self.dim)
        self.linear_i = nn.Linear(self.dim, self.dim)
        self.linear_h = nn.Linear(self.dim, self.dim)
    

    def forward(self, x_t, pre_h=None):
        """
        pre_h: (batch_size, units) - previous hidden state (h_prev)
        x_t: (batch_size, input_size) - input at time step t
        """

        if pre_h is None:
            pre_h = torch.zeros(x_t.shape[0], 1, self.dim, device=x_t.device)

        # Forget gate: log_f_t = log(sigmoid(W_f * x_t))
        k_f = self.linear_f(x_t)
        log_f = -F.softplus(-k_f) # (batch_size, units)

        k_i = self.linear_i(x_t)
        log_i = -F.softplus(-k_i)


        # Hidden state: log_tilde_h_t = log(W_h * x_t)
        log_tilde_h = log_g(self.linear_h(x_t))  # (batch_size, units)

        

        # Compute the new hidden state using parallel_scan_log
        log_pre_h = log_g(pre_h)  # Convert previous hidden state to log space
        
        # Use parallel_scan_log to compute the hidden state
        h_t = parallel_scan_log(log_f, torch.cat([log_pre_h, log_i + log_tilde_h], dim=1))

        return h_t  # Return the hidden state