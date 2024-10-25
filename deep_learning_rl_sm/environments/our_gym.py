import gymnasium as gym

class OurEnv(gym.Env):
    def __init__(self, action_mask: gym.spaces.Space):
        super(OurEnv, self).__init__()
        self.action_mask = action_mask