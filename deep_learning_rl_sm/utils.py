import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")


def warn(*args, **kwargs):
    pass


from deep_learning_rl_sm.environments import connect_four
from deep_learning_rl_sm.benchmarks.get_benchmark import *
import torch
import os
import numpy as np
import h5py

from experts.DQN.DQN import DQN


def download_all_benchmarks():
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    for env_bm in ENV_BENCHMARKS:
        download_dataset_from_url(env_bm)


def benchmark_data(filepath):
    """Get the data in numpy format from a benchmark filepath"""
    file = h5py.File(filepath, "r+")
    actions = file['actions'][()]
    next_observations = file['next_observations'][()]
    observations = file['observations'][()]
    rewards = file['rewards'][()]
    terminals = file['terminals'][()]
    timeouts = file['timeouts'][()]
    dones = terminals | timeouts
    trajs = [-1] + [i for i, x in enumerate(dones) if x]  # gets trajectory indices
    obs, act, r, rtg, d, l, ret_sum = [], [], [], [], [], [], []
    for i in range(len(trajs) - 1):
        ind_st = trajs[i] + 1
        ind_end = trajs[i + 1] + 1
        l.append(ind_end - ind_st)
        traj_rews = rewards[ind_st:ind_end]
        final_rew = sum(traj_rews)
        cumsum_rewards = np.cumsum(traj_rews)
        rew_to_go = final_rew - cumsum_rewards
        ob = observations[ind_st:ind_end]
        ac = actions[ind_st:ind_end]
        done = dones[ind_st:ind_end]
        r.append(traj_rews)
        obs.append(ob)
        act.append(ac)
        rtg.append(rew_to_go)
        d.append(done)
        ret_sum.append(final_rew)
    obs_concat = np.concatenate(obs, axis=0)
    state_mean, state_std = np.mean(obs_concat, axis=0), np.std(obs_concat, axis=0) + 1e-6
    obs = [(ob - state_mean) / state_std for ob in obs]  # Normalize
    del ret_sum, r, d
    return obs, act, rtg, state_mean, state_std


def extract_dataset(data):
    # below the assumed format for the elements/trajectories in the dataset
    # [first_state, [t1, mask1, act1, rew1, ret1, nxt_state1], [t2, mask2, act2, rew2, ret2, nxt_state2],...]
    time_steps = []
    action_masks = []
    states = []
    actions = []
    returns_to_go = []
    rewards = []
    dones = []
    for data_traj in data:
        time_steps.append(torch.tensor([t[0].item() for idx, t in enumerate(data_traj) if idx != 0]))
        states.append(torch.tensor([t[5] if type(t) is list else t for t in data_traj]))
        action_masks.append(torch.tensor([t[1] for idx, t in enumerate(data_traj) if idx != 0]))
        actions.append(torch.tensor([t[2].item() for idx, t in enumerate(data_traj) if idx != 0]))
        returns_to_go.append(torch.tensor([t[4] for idx, t in enumerate(data_traj) if idx != 0]))
        rewards.append(torch.tensor([t[3].item() for idx, t in enumerate(data_traj) if idx != 0]))
        dones.append(torch.tensor([0 for idx, _ in enumerate(data_traj) if idx != 0]))
        dones[-1][-1] = 1  # sequence only stops when game completed (in our cases so far)
    return time_steps, action_masks, actions, returns_to_go, rewards, states, dones


def generate_data(batch_size=1000, agent=None, adv=None):
    c4 = connect_four.ConnectFour()
    data = c4.generate_seq(batch_size, actor=agent, adv_actor=adv)
    time_steps, action_masks, actions, returns_to_go, rewards, states, traj_masks = extract_dataset(data)

    # [1,2,3,4,...] so 0 can be used as a padding index
    time_steps = [time_step_tensor + 1 for time_step_tensor in time_steps]

    # convert actions to a one hot representation
    for idx, action in enumerate(actions):
        actions[idx] = F.one_hot(action, num_classes=7)

    # TODO is this padding scheme viable?
    # PADDING PARAMETERS
    padded_state = torch.full((22, 42), -100)  # Padding value for states
    padded_mask = torch.full((21, 7), -100)  # Padding for mask
    padded_action = torch.full((21, 7), 0)
    padded_else = torch.full((21, 1), -100)  # Padding value for other variables
    padded_timestep = torch.full((21, 1), 0)

    # PADDING STATES
    for idx, state in enumerate(states):
        pad_state = padded_state.clone()
        pad_state[:state.shape[0], :] = state.flatten(start_dim=-2)
        states[idx] = pad_state
    states = torch.stack(states)

    # PADDING OTHER VARIABLES (actions, rewards, dones, time_steps, action_masks, returns_to_go)
    for idx in range(len(actions)):
        pad_act = padded_action.clone()
        pad_rew = padded_else.clone()
        pad_dones = padded_else.clone()
        pad_time = padded_timestep.clone()
        pad_mask = padded_mask.clone()
        pad_rtg = padded_else.clone()

        pad_act[:actions[idx].shape[0]] = actions[idx]
        pad_rew[:rewards[idx].shape[0]] = rewards[idx].unsqueeze(-1)
        pad_dones[:traj_masks[idx].shape[0]] = traj_masks[idx].unsqueeze(-1)
        pad_time[:time_steps[idx].shape[0]] = time_steps[idx].unsqueeze(-1)
        pad_rtg[:returns_to_go[idx].shape[0]] = returns_to_go[idx].unsqueeze(-1)
        pad_mask[:action_masks[idx].shape[0]] = action_masks[idx]

        actions[idx] = pad_act
        rewards[idx] = pad_rew
        traj_masks[idx] = pad_dones
        time_steps[idx] = pad_time
        action_masks[idx] = pad_mask
        returns_to_go[idx] = pad_rtg

    # STACK PADDED TENSORS
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    dones = torch.stack(traj_masks)
    time_steps = torch.stack(time_steps)
    action_masks = torch.stack(action_masks)
    returns_to_go = torch.stack(returns_to_go)

    # Save the data
    if not os.path.exists("./data/"):
        os.makedirs("./data/")
    data_path = 'data/offline_data.pt'
    torch.save({
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'time_steps': time_steps,
        'action_masks': action_masks,
        'returns_to_go': returns_to_go
    }, data_path)
    return data_path


def main():
    # get saved agent and adversary (trained 2 Double-DQNs for connect-4)
    BATCH_SIZE = 64
    GAMMA = 0.99
    eps_start = 1.0
    eps_end = 0.1
    eps_decay = 5000
    agent_dqn = DQN(BATCH_SIZE, GAMMA, eps_start, eps_end, eps_decay)
    adversary_dqn = DQN(BATCH_SIZE, GAMMA, eps_start, eps_end, eps_decay)
    # TODO path might need to be changed for you (i needed an absolute path : Tim)
    agent_dqn.policy_net.load_state_dict(
        torch.load("C:/Users/Tim/Documents/ETHz/DL/deep-learning-rl-sm/experts/DQN/net_configs/agent_dqn.pth",
                   weights_only=True))
    agent_dqn.policy_net.eval()
    # "C:/Users/Tim/Documents/ETHz/DL/deep-learning-rl-sm/experts/DQN/net_configs/adversary_dqn.pth"
    adversary_dqn.policy_net.load_state_dict(
        torch.load("C:/Users/Tim/Documents/ETHz/DL/deep-learning-rl-sm/experts/DQN/net_configs/adversary_dqn.pth",
                   weights_only=True))
    adversary_dqn.policy_net.eval()

    # have dataset with size that is a multiple of batch size (128) used
    dp = generate_data(batch_size=1280, agent=agent_dqn, adv=adversary_dqn)

    loaded_data = torch.load(dp)

    # Access individual items from the loaded dictionary
    s = loaded_data['states']
    a = loaded_data['actions']
    r = loaded_data['rewards']
    d = loaded_data['dones']
    t_s = loaded_data['time_steps']
    a_m = loaded_data['action_masks']
    r_t_g = loaded_data['returns_to_go']
    # print(s[0].reshape((22, 6, 7)))
    print(a[0][0])
    print(s.shape)  # state trajectories have one more timestep than the rest
    print(a.shape)
    print(r.shape)
    print(d.shape)
    print(t_s.shape)
    print(a_m.shape)
    print(r_t_g.shape)
    print()
    print("saved data...")


if __name__ == "__main__":
    main()
