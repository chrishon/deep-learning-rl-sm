import torch
import numpy as np
import pandas as p
import argparse
import time
from utils import ReplayMemory, Transition, resizer
from experts.DQN.DQN import DQN
from deep_learning_rl_sm.environments.connect_four import ConnectFour

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--BATCH_SIZE", default=64, type=int,
                        help="Size of the batch used in training to update the networks(default: 32)")
    parser.add_argument("--num_games", default=1000, type=int,
                        help="Num. of total timesteps of training (default: 3000)")
    parser.add_argument("--gamma", default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--tau", default=0.001,
                        help="Update factor for the soft update of the target networks (default: 0.001)")
    parser.add_argument("--EVALUATE", default=50, type=int,
                        help="Number of episodes between testing cycles(default: 50)")
    parser.add_argument("--mem_size", default=10000, type=int,
                        help="Size of the Replay Buffer(default: 10000)")
    parser.add_argument("--noise_SD", default=1.0, type=float,
                        help="noise Standard deviation(default: 0.7)")
    parser.add_argument("--eps_start", default=1.0,
                        help="used for random actions in DQN")
    parser.add_argument("--eps_end", default=0.1,
                        help="used for random actions in DQN")
    parser.add_argument("--eps_decay", default=700,
                        help="used for random actions in DQN")
    parser.add_argument("--target_update", default=20, type=int,
                        help="number of iterations before dqn_target is receives hard update")
    args = parser.parse_args()
    # TODO if enough time replace hard update with soft update

    memory = ReplayMemory(args.mem_size)
    env = ConnectFour()
    # TODO everything from here on needs to be changed potentially
    agent_dqn = DQN(args.BATCH_SIZE, args.width, args.height, args.gamma, args.eps_start, args.eps_end, args.eps_decay,
                    args.num_planners)

    resizer = resizer(args.width)  # args.width should equal args.height -> i.e square

    episodeList = []
    averageRewardList = []
    i_episode = 0
    num_passes = 0
    max_average_reward = float("-inf")

    # TRAINING
    while time.time() < args.timeout:
        print("episode " + str(i_episode))
        # obs is a dict
        obs, _ = env.reset()
        task_img = torch.from_numpy(obs.get('task_img'))
        img = resizer.resize(task_img)
        state = img.unsqueeze(0)
        state_additional = torch.cat((torch.tensor(env.max_time_executed, dtype=torch.float32),
                                      torch.tensor(env.consecutive_time_running, dtype=torch.float32),
                                      torch.tensor([env.time_left], dtype=torch.float32)))
        final_state = False
        time_restriction = False
        while not final_state and not time_restriction:
            num_passes += 1
            # Select and perform an action
            action = agent_dqn.select_action(state, torch.unsqueeze(state_additional, dim=0), num_passes)
            actionTime = torch.tensor([100])
            action = action.squeeze(0)
            env_action = np.concatenate(
                ((np.array(action.detach())), np.array(actionTime.detach())))
            obs, reward, final_state, time_restriction, _ = env.step(env_action)
            print("action number: ", action)
            print(reward)
            print(final_state)
            print(time_restriction)
            next_state = state
            next_state_additional = torch.tensor(obs.get('task_additional'), dtype=torch.float32)
            mask = torch.Tensor([final_state])
            reward = torch.tensor(reward, dtype=torch.float32)

            # Store the transition in memory
            memory.push(state, state_additional, env.task_idx, action, actionTime, mask, next_state,
                        next_state_additional, reward)

            # Move to the next state
            state = next_state
            state_additional = next_state_additional
            # Perform one step of the optimization (on the policy network/s)
            if len(memory) >= args.BATCH_SIZE:
                transitions = memory.sample(args.BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                agent_dqn.update(batch)

            if num_passes % args.target_update == 0:
                # hard update
                agent_dqn.target_net.load_state_dict(agent_dqn.policy_net.state_dict())

        if i_episode % args.EVALUATE == 0:
            if len(memory) >= args.BATCH_SIZE:
                print("testing network...")
                # TODO plot a random game here to approx. view progress
                if max_average_reward < averageRewardList[-1]:
                    max_average_reward = averageRewardList[-1]
                    torch.save(agent_dqn.policy_net.state_dict(), "net_configs/dqn.pth")

        i_episode += 1

    print('Completed training...')
