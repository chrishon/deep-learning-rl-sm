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

    def forward(self, x_t,pre_h=None):
        """
        pre_h: (batch_size, units) - previous hidden state (h_prev)
        x_t: (batch_size, input_size) - input at time step t
        """

        if pre_h is None:
            pre_h = torch.zeros(x_t.shape[0], 1, self.dim)

        # Forget gate: f_t = sigmoid(W_f * x_t)
        f_t = torch.sigmoid(self.linear_f(x_t))  # (batch_size, units)

        # Input gate: i_t = sigmoid(W_i * x_t)
        i_t = torch.sigmoid(self.linear_i(x_t))  # (batch_size, units)

        # Hidden state: tilde_h_t = W_h * x_t
        tilde_h_t = self.linear_h(x_t)  # (batch_size, units)

        # Normalize the gates
        sum_f_i = f_t + i_t
        f_prime_t = f_t / sum_f_i  # (batch_size, units)
        i_prime_t = i_t / sum_f_i  # (batch_size, units)

        # New hidden state: h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t
        h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t  # (batch_size, units)

        return h_t  # (batch_size, units)
    
    #TODO: implement this correctly
    def forward_log(self, x_t, pre_h=None):
        """
        pre_h: (batch_size, units) - previous hidden state (h_prev)
        x_t: (batch_size, input_size) - input at time step t
        """

        if pre_h is None:
            pre_h = torch.zeros(x_t.shape[0], 1, self.dim, device=x_t.device)

        # Forget gate: log_f_t = log(sigmoid(W_f * x_t))
        log_f_t = log_g(self.linear_f(x_t))  # (batch_size, units)

        # Input gate: log_i_t = log(sigmoid(W_i * x_t))
        log_i_t = log_g(self.linear_i(x_t))  # (batch_size, units)

        # Hidden state: log_tilde_h_t = log(W_h * x_t)
        log_tilde_h_t = log_g(self.linear_h(x_t))  # (batch_size, units)

        # Log normalization of the gates
        log_sum_f_i = torch.logsumexp(torch.stack([log_f_t, log_i_t], dim=-1), dim=-1)  # (batch_size, units)
        log_f_prime_t = log_f_t - log_sum_f_i  # log(f_t / (f_t + i_t))
        log_i_prime_t = log_i_t - log_sum_f_i  # log(i_t / (f_t + i_t))

        # Compute the new hidden state using parallel_scan_log
        log_pre_h = log_g(pre_h)  # Convert previous hidden state to log space
        log_values = torch.cat([log_pre_h, log_i_prime_t + log_tilde_h_t], dim=1)  # Combine contributions
        log_coefficients = torch.cat([log_f_prime_t, log_i_prime_t], dim=1)  # Log coefficients for scan

        # Use parallel_scan_log to compute the hidden state
        h_t = parallel_scan_log(log_coefficients, log_values)

        return h_t  # Return the hidden state