import gymnasium as gym
import copy
from abc import ABC, abstractmethod
import numpy as np
import torch


def calculate_returns(seq):
    end_idx = len(seq) - 1
    ret_2_go = 0
    for i in range(len(seq)):
        if isinstance(seq[end_idx - i], np.ndarray):
            # reached first element of sequence which is just the initial state
            continue
        # return to go is the 5th entry and reward is 4th entry!
        seq[end_idx - i][4] = ret_2_go
        ret_2_go += seq[end_idx - i][3]
    return seq


def convert_state(s):
    s[s == 2] = -1
    s = torch.tensor(s, dtype=torch.float32)
    # print(type(s))
    return s


class OurEnv(gym.Env):
    def __init__(self, action_mask: gym.spaces.Space):
        super(OurEnv, self).__init__()
        self.action_mask = action_mask
        self.state_dim = ()
        self.action_dim = ()

    def generate_seq(self, no_sequences: int, max_seq_len: int = 10000, actor=None, adv_actor=None):
        sequences = []
        act_space = self.action_space
        for _ in range(no_sequences):
            first_state, _ = self.reset()
            seq = [copy.deepcopy(first_state)]
            # print(copy.deepcopy(first_state))
            s = first_state
            for t in range(max_seq_len):
                if actor is not None:
                    action = actor.get_action_from_net(state=convert_state(np.copy(s)), action_mask=self.action_mask)
                    next_state, reward, done, _, _ = self.step_2P(action, 1)
                    if not done:
                        action_adv = adv_actor.get_action_from_net(state=convert_state(np.copy(next_state)),
                                                                   action_mask=self.action_mask)
                        next_state, _, done, _, _ = self.step_2P(action_adv, 2)
                else:
                    action = act_space.sample(self.action_mask)
                    next_state, reward, done, _, _ = self.step(action)
                # We calculate return to go in backward pass!
                # (t, mask, act, rew, ret, nxt_state)
                lst = [np.array(t), np.array(self.action_mask), np.array(action), np.array(reward), np.array(0),
                       np.array(copy.deepcopy(next_state))]
                seq.append(lst)
                s = next_state
                if done:
                    break
            seq = calculate_returns(seq)
            sequences.append(seq)
        # sequences has the form: [state, [t, mask, act, rew, ret, state],[t, mask, act, rew, ret, state],...]
        return sequences

    @abstractmethod
    def step_2P(self, action, player):
        pass
