import math

from DQN_net import DQN_net
import torch
from torch.optim import Adam
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(object):
    def __init__(self, batchsize, gamma, eps_start, eps_end, eps_decay, n_actions=7):
        self.BATCH_SIZE = batchsize
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay

        self.n_actions = n_actions

        self.policy_net = DQN_net().to(device)
        self.target_net = DQN_net().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # hard update
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters())

    def set_eval(self):  # sets networks to evaluation mode (faster)
        # Sets the model in evaluation mode
        self.policy_net.eval()
        self.target_net.eval()

    def set_train(self):  # sets networks to training mode (needed for learning but slower)
        # Sets the model in training mode
        self.policy_net.train()
        self.target_net.train()

    def update(self, transition_batch):
        # TODO save action masks to replay buffer
        self.set_train()
        state_batch = torch.cat(transition_batch.state).to(device)
        action_batch = torch.stack(transition_batch.action).to(device)
        reward_batch = torch.cat(transition_batch.reward_planner).to(device)
        done_batch = torch.cat(transition_batch.done).to(device)
        next_state_batch = torch.cat(transition_batch.next_state).to(device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # TODO use argmax
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (1 - done_batch) * next_state_values * self.GAMMA + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1)  # change shape for loss

        # Compute MSE loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_action(self, state, steps_done, action_mask):
        # action_mask contains 1s for all valid actions
        sample = random.random()
        # linear decay
        eps_threshold = self.EPS_START - (self.EPS_START - self.EPS_END) * min(steps_done / self.EPS_DECAY, 1.0)
        if sample > eps_threshold:
            mask = torch.tensor([-torch.inf if entry == 0 else 0 for entry in action_mask])
            with torch.no_grad():
                action_vector = self.policy_net(state, mask)
                return torch.argmax(action_vector, dim=-1).unsqueeze(-1)
        else:
            # valid action indices
            indices = [index for index, val in enumerate(action_mask) if val == 1]
            num_actions = self.n_actions - (7 - sum(action_mask))
            actionNo = random.randrange(num_actions)
            actionNo = torch.tensor([[indices[actionNo]]], device=device)
            return actionNo

    @staticmethod
    def hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
