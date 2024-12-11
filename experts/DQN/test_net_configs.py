from deep_learning_rl_sm.environments.connect_four import ConnectFour
from experts.DQN.DQN import DQN
import torch


def run_game_vs_rand(environment, agent):
    w_ratio = 0
    d_ratio = 0
    l_ratio = 0
    for _ in range(1000):
        done = False
        last_reward = None
        s, _ = environment.reset()
        while not done:
            action_agent = agent.get_action_from_net(state=torch.flatten(torch.tensor(s, dtype=torch.float32)),
                                                     action_mask=environment.action_mask)
            s, r, done, _, _ = environment.step(action=action_agent)
            # print("reward:"+str(r))
            last_reward = r
        if last_reward == 1:
            w_ratio += 1
        elif last_reward == -1:
            l_ratio += 1
        elif last_reward == 0:
            d_ratio += 1

    w_ratio /= 1000
    d_ratio /= 1000
    l_ratio /= 1000
    return w_ratio, d_ratio, l_ratio


BATCH_SIZE = 64
GAMMA = 0.99
eps_start = 1.0
eps_end = 0.1
eps_decay = 5000
agent_dqn = DQN(BATCH_SIZE, GAMMA, eps_start, eps_end, eps_decay)
adversary_dqn = DQN(BATCH_SIZE, GAMMA, eps_start, eps_end, eps_decay)
agent_dqn.policy_net.load_state_dict(torch.load("net_configs/agent_dqn.pth", weights_only=True))
agent_dqn.policy_net.eval()
adversary_dqn.policy_net.load_state_dict(torch.load("net_configs/adversary_dqn.pth", weights_only=True))
adversary_dqn.policy_net.eval()

# evaluation vs random
win_ratio, draw_ratio, loss_ratio = run_game_vs_rand(ConnectFour(), agent_dqn)
print(win_ratio)
print(draw_ratio)
print(loss_ratio)
