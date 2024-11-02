from deep_learning_rl_sm.trainer.trainer import Trainer
from deep_learning_rl_sm.neuralnets import minGRU_Reinformer
from deep_learning_rl_sm.environments.connect_four import ConnectFour

env = ConnectFour()
sequences = env.generate_seq()

# Contine Training here
