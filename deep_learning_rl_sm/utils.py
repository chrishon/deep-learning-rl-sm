from deep_learning_rl_sm.environments import connect_four
import torch


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
        time_steps.append(torch.tensor([t[0] if type(t) is list else [] for t in data_traj]))
        states.append(torch.tensor([t[5] if type(t) is list else t for t in data_traj]))
        action_masks.append(torch.tensor([t[1] if type(t) is list else [] for t in data_traj]))
        actions.append(torch.tensor([t[2] if type(t) is list else [] for t in data_traj]))
        returns_to_go.append(torch.tensor([t[4] if type(t) is list else [] for t in data_traj]))
        rewards.append(torch.tensor([t[3] if type(t) is list else [] for t in data_traj]))
        dones.append(torch.tensor([0 for _ in data_traj]))
        dones[-1][-1] = 1  # sequence only stops when game completed (in our cases so far)
    return time_steps, action_masks, actions, returns_to_go, rewards, states, dones


"""
what we need:
timesteps,
states,
actions,
returns_to_go,
rewards,
traj_mask == dones
"""


def generate_data():
    # Convert to tensors
    c4 = connect_four.ConnectFour()
    data = c4.generate_seq(1000)
    # (t, mask, act, rew, ret, nxt_state)
    time_steps, action_masks, actions, returns_to_go, rewards, states, traj_masks = extract_dataset(data)
    # TODO padding
    states = torch.stack(states)  # Shape: (num_samples, state_dim)
    actions = torch.tensor(actions, dtype=torch.long)  # Shape: (num_samples,)
    rewards = torch.tensor(rewards, dtype=torch.float32)  # Shape: (num_samples,)
    dones = torch.tensor(traj_masks, dtype=torch.float32)  # Shape: (num_samples,)

    # Save the data
    data_path = 'deep_learning_rl_sm/data/offline_data.pt'
    torch.save({
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'dones': dones
    }, data_path)


generate_data()
