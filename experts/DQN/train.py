import torch
import numpy as np
import argparse
from utils import ReplayMemory, Transition
from experts.DQN.DQN import DQN
from deep_learning_rl_sm.environments.connect_four import ConnectFour


# IMPORTANT REMEMBER CONVERT STATE!!! for dataset creation with nets
# Hopefully improves training stability/efficiency (better representation???)
def convert_state(s):
    s[s == 2] = -1
    return s


def run_game(environment, agent, adversary):
    done = False
    s, _ = environment.reset()
    s = convert_state(np.copy(s))
    while not done:
        action_agent = agent.get_action_from_net(state=torch.tensor(s, dtype=torch.float32),
                                                 action_mask=environment.action_mask)
        s, _, done, _, _ = environment.step_2P(action=action_agent, player=1)
        s = convert_state(np.copy(s))
        environment.display_board()
        if done:
            break
        action_adversary = adversary.get_action_from_net(state=torch.tensor(s, dtype=torch.float32),
                                                         action_mask=environment.action_mask)
        s, _, done, _, _ = environment.step_2P(action=action_adversary, player=2)
        s = convert_state(np.copy(s))
        environment.display_board()


def run_game_vs_rand(environment, agent):
    w_ratio = 0
    d_ratio = 0
    l_ratio = 0
    for _ in range(1000):
        done = False
        last_reward = None
        s, _ = environment.reset()
        s = convert_state(np.copy(s))
        while not done:
            action_agent = agent.get_action_from_net(state=torch.tensor(s, dtype=torch.float32),
                                                     action_mask=environment.action_mask)
            s, r, done, _, _ = environment.step(action=action_agent)
            s = convert_state(np.copy(s))
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


def agent_loop(agent, current_state, Replay_memory, environment, adv=False):
    # Select and perform an action
    current_state = torch.tensor(current_state, dtype=torch.float32)
    current_state = current_state.unsqueeze(0)
    action = agent.select_action(current_state, num_passes, environment.action_mask)
    action_mask = torch.tensor(environment.action_mask).unsqueeze(0)
    action = action.squeeze()
    player = 1 if adv is False else 2
    next_state, reward, done, time_restriction, _ = environment.step_2P(action, player=player)
    next_state = convert_state(np.copy(next_state))
    next_action_mask = torch.tensor(environment.action_mask).unsqueeze(0)

    # convert to tensors
    done_mask = torch.Tensor([done])
    reward = torch.tensor(reward, dtype=torch.float32)

    # Store the transition in memory
    Replay_memory.push(current_state.unsqueeze(0), action.unsqueeze(0), action_mask, done_mask.unsqueeze(0),
                       torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0), next_action_mask,
                       reward.unsqueeze(0))

    # Perform one step of the optimization (on the policy network/s)
    if len(Replay_memory) >= args.BATCH_SIZE:
        transitions = Replay_memory.sample(args.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        agent.update(batch)

        # soft update
        agent.soft_update()

    return next_state, done


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--BATCH_SIZE", default=64, type=int,
                        help="Size of the batch used in training to update the networks(default: 32)")
    parser.add_argument("--num_games", default=1000000, type=int,
                        help="Num. of total timesteps of training (default: 3000)")
    parser.add_argument("--gamma", default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--tau", default=0.001,
                        help="Update factor for the soft update of the target networks (default: 0.001)")
    parser.add_argument("--EVALUATE", default=500, type=int,
                        help="Number of episodes between testing cycles(default: 50)")
    parser.add_argument("--mem_size", default=50000, type=int,
                        help="Size of the Replay Buffer")
    parser.add_argument("--noise_SD", default=1.0, type=float,
                        help="noise Standard deviation")
    parser.add_argument("--eps_start", default=1.0,
                        help="used for random actions in DQN")
    parser.add_argument("--eps_end", default=0.1,
                        help="used for random actions in DQN")
    parser.add_argument("--eps_decay", default=5000,
                        help="used for random actions in DQN")
    parser.add_argument("--target_update", default=20, type=int,
                        help="number of iterations before dqn_target receives hard update")
    args = parser.parse_args()

    # TODO (with enough time)
    #  . implement Prioritized experience replay where priority given by TD-error

    memory_agent = ReplayMemory(args.mem_size)
    memory_adv = ReplayMemory(args.mem_size)

    env = ConnectFour()
    agent_dqn = DQN(args.BATCH_SIZE, args.gamma, args.eps_start, args.eps_end, args.eps_decay)
    adversary_dqn = DQN(args.BATCH_SIZE, args.gamma, args.eps_start, args.eps_end, args.eps_decay)

    episodeList = []
    averageRewardList = []
    i_episode = 0
    num_passes = 0
    max_average_reward = float("-inf")
    best_w_ratio_vs_random = 0.0

    # TRAINING
    while i_episode < args.num_games:
        print("episode " + str(i_episode))
        state, _ = env.reset()
        state = convert_state(np.copy(state))
        final_state = False
        while not final_state:
            num_passes += 1
            if num_passes % 2 == 1:
                state, final_state = agent_loop(agent_dqn, state, memory_agent, env, adv=False)
            else:
                state, final_state = agent_loop(adversary_dqn, state, memory_adv, env, adv=True)

        if i_episode % args.EVALUATE == 0:
            if len(memory_agent) >= args.BATCH_SIZE:
                print("testing network...")
                print()
                if i_episode % 4 * args.EVALUATE == 0:
                    print("agent vs agent:")
                    run_game(ConnectFour(), agent_dqn, adversary_dqn)
                    print()
                    print()
                print("agent vs rand:")
                win_ratio, draw_ratio, loss_ratio = run_game_vs_rand(ConnectFour(), agent_dqn)
                print(win_ratio)
                print(draw_ratio)
                print(loss_ratio)
                """if win_ratio > best_w_ratio_vs_random:
                    best_w_ratio_vs_random = win_ratio
                    print("saving network parameters...")
                    torch.save(agent_dqn.policy_net.state_dict(), "net_configs/agent_dqn.pth")
                    torch.save(adversary_dqn.policy_net.state_dict(), "net_configs/adversary_dqn.pth")"""
                print()

        i_episode += 1

    print('Completed training...')
