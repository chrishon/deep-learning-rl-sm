import torch
import torch.nn as nn
import numpy as np
from deep_learning_rl_sm.neuralnets.minGRU import minGRU
from deep_learning_rl_sm.neuralnets.nets import Actor


class minGRU_Reinformer(nn.Module):
    def __init__(
            self,
            state_dim,
            act_dim,
            n_blocks,
            h_dim,
            context_len,
            n_heads,
            drop_p,
            init_tmp,
            target_entropy,
            max_timestep=4096):
        super().__init__()

        self.a_dim = act_dim
        self.s_dim = state_dim
        self.h_dim = h_dim

        # minGRU blocks
        self.num_inputs = 3
        seq_len_in = self.num_inputs * context_len
        min_gru_blocks = [
            minGRU(self.h_dim)
            for _ in range(n_blocks)
        ]
        self.min_gru_stacked = nn.Sequential(*min_gru_blocks)

        # projection heads (project to embedding) /same as paper
        self.embed_ln = nn.LayerNorm(self.h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, self.h_dim)
        self.embed_state = nn.Linear(np.prod(self.s_dim), self.h_dim)
        self.embed_rtg = nn.Linear(1, self.h_dim)
        self.embed_action = nn.Linear(self.a_dim, self.h_dim)

        # prediction heads /same as paper
        self.predict_rtg = nn.Linear(self.h_dim, 1)
        # stochastic action (output is distribution)
        self.predict_action = Actor( self.a_dim, self.h_dim)
        self.predict_state = nn.Linear(self.h_dim, np.prod(self.s_dim))

        # For entropy /same as paper
        self.log_tmp = torch.tensor(np.log(init_tmp))
        self.log_tmp.requires_grad = True
        self.target_entropy = target_entropy

    def forward(
            self,
            timesteps,
            states,
            actions,
            returns_to_go,
    ):
        B, T, _ = states.shape

        embd_t = self.embed_timestep(timesteps)
        print("embed_t dim: ", embd_t.shape)
        # time embeddings ≈ pos embeddings
        # add time embedding to each embedding below for temporal context
        embd_s = self.embed_state(states) + embd_t
        embd_a = self.embed_action(actions) + embd_t
        print(self.embed_rtg(returns_to_go).shape)
        embd_rtg = self.embed_rtg(returns_to_go) + embd_t

        # stack states, RTGs, and actions and reshape sequence as
        # (s_0, R_0, a_0, s_1, R_1, a_1, s_2, R_2, a_2 ...)
        h = (
            torch.stack(
                (
                    embd_s,
                    embd_rtg,
                    embd_a,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, self.num_inputs * T, self.h_dim)
        )

        h = self.embed_ln(h)
        print("h shape: ", h.shape)
        # transformer and prediction
        h = self.min_gru_stacked(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence s_0, R_0, a_0 ... s_t
        # h[:, 1, t] is conditioned on the input sequence s_0, R_0, a_0 ... s_t, R_t
        # h[:, 2, t] is conditioned on the input sequence s_0, R_0, a_0 ... s_t, R_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus
        # the 3 input variables at that timestep (s_t, R_t, a_t) in sequence.
        h = h.reshape(B, T, self.num_inputs, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        rtg_preds = self.predict_rtg(h[:, 0])  # predict rtg given s
        action_dist_preds = self.predict_action(h[:, 1])  # predict action given s, R
        state_preds = self.predict_state(h[:, 2])  # predict next state given s, R, a

        return rtg_preds, action_dist_preds, state_preds

    def temp(self):
        return torch.exp(self.log_tmp)
