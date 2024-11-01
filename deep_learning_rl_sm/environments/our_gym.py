import gymnasium as gym
import copy

class OurEnv(gym.Env):
    def __init__(self, action_mask: gym.spaces.Space):
        super(OurEnv, self).__init__()
        self.action_mask = action_mask

    def generate_seq(self, no_sequences: int, max_seq_len: int = 1000, actor=None, adv_actor=None):
        sequences = []
        act_space = self.action_space
        for _ in range(no_sequences):
            first_state, _ = self.reset()
            seq = [copy.deepcopy(first_state)]
            for _ in range(max_seq_len):
                action = act_space.sample(self.action_mask)
                next_state, reward, done, _, _ = self.step(action)
                seq.append(action)
                seq.append(reward)
                seq.append(copy.deepcopy(next_state))
                if done:
                    break
            sequences.append(tuple(seq))
            print(sequences)
        return sequences