import numpy as np
import copy
from deep_learning_rl_sm.environments.our_gym import OurEnv
from environments import connect_four


# use this to test the seq gen in the individual envs
c4 = connect_four.ConnectFour()
seqs = c4.generate_seq(2)
print(seqs[0])
print()
print(seqs[1])
