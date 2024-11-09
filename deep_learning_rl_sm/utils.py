from deep_learning_rl_sm.environments import connect_four
import torch
import os
import urllib

env_dict = {"c4": connect_four.ConnectFour}

act_dims = {"halfcheetah" : 6, "hopper" : 3, "walker2d": 6}
state_dims = {"halfcheetah" : 17, "hopper" : 11, "walker2d": 17}

BENCHMARK_LIST = ['halfcheetah-medium-v0','halfcheetah-medium-replay-v0','halfcheetah-medium-expert-v0',
                  'walker2d-medium-v0','walker2d-medium-replay-v0','walker2d-medium-expert-v0',
                  'hopper-medium-v0','hopper-medium-replay-v0','hopper-medium-expert-v0']

DATASET_URLS = {
    'maze2d-open-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-sparse.hdf5',
    'maze2d-umaze-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5',
    'maze2d-medium-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5',
    'maze2d-large-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5',
    'maze2d-eval-umaze-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-umaze-sparse-v1.hdf5',
    'maze2d-eval-medium-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-medium-sparse-v1.hdf5',
    'maze2d-eval-large-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-large-sparse-v1.hdf5',
    'maze2d-open-dense-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-dense.hdf5',
    'maze2d-umaze-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-dense-v1.hdf5',
    'maze2d-medium-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-dense-v1.hdf5',
    'maze2d-large-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-dense-v1.hdf5',
    'maze2d-eval-umaze-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-umaze-dense-v1.hdf5',
    'maze2d-eval-medium-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-medium-dense-v1.hdf5',
    'maze2d-eval-large-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-large-dense-v1.hdf5',
    'minigrid-fourrooms-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/minigrid/minigrid4rooms.hdf5',
    'minigrid-fourrooms-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/minigrid/minigrid4rooms_random.hdf5',
    'pen-human-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_demos_clipped.hdf5',
    'pen-cloned-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-demos-v0-bc-combined.hdf5',
    'pen-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_expert_clipped.hdf5',
    'hammer-human-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_demos_clipped.hdf5',
    'hammer-cloned-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-demos-v0-bc-combined.hdf5',
    'hammer-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_expert_clipped.hdf5',
    'relocate-human-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_demos_clipped.hdf5',
    'relocate-cloned-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-demos-v0-bc-combined.hdf5',
    'relocate-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_expert_clipped.hdf5',
    'door-human-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_demos_clipped.hdf5',
    'door-cloned-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-demos-v0-bc-combined.hdf5',
    'door-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_expert_clipped.hdf5',
    'halfcheetah-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_random.hdf5',
    'halfcheetah-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5',
    'halfcheetah-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_expert.hdf5',
    'halfcheetah-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_mixed.hdf5',
    'halfcheetah-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium_expert.hdf5',
    'walker2d-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_random.hdf5',
    'walker2d-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium.hdf5',
    'walker2d-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_expert.hdf5',
    'walker2d-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker_mixed.hdf5',
    'walker2d-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium_expert.hdf5',
    'hopper-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_random.hdf5',
    'hopper-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5',
    'hopper-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5',
    'hopper-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_mixed.hdf5',
    'hopper-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium_expert.hdf5',
    'ant-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_random.hdf5',
    'ant-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_medium.hdf5',
    'ant-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_expert.hdf5',
    'ant-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_mixed.hdf5',
    'ant-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_medium_expert.hdf5',
    'ant-random-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_random_expert.hdf5',
    'antmaze-umaze-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
    'antmaze-umaze-diverse-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
    'antmaze-medium-play-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
    'antmaze-medium-diverse-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
    'antmaze-large-play-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
    'antmaze-large-diverse-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
    'antmaze-umaze-v2' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5',
    'antmaze-umaze-diverse-v2' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
    'antmaze-medium-play-v2' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5',
    'antmaze-medium-diverse-v2' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
    'antmaze-large-play-v2' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5',
    'antmaze-large-diverse-v2' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
    'flow-ring-random-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-ring-v0-random.hdf5',
    'flow-ring-controller-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-ring-v0-idm.hdf5',
    'flow-merge-random-v0':'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-merge-v0-random.hdf5',
    'flow-merge-controller-v0':'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-merge-v0-idm.hdf5',
    'kitchen-complete-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
    'kitchen-partial-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_light_slider-v0.hdf5',
    'kitchen-mixed-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    'carla-lane-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_lane_follow_flat-v0.hdf5',
    'carla-town-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town_subsamp_flat-v0.hdf5',
    'carla-town-full-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town_flat-v0.hdf5',
    'bullet-halfcheetah-random-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_random.hdf5',
    'bullet-halfcheetah-medium-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_medium.hdf5',
    'bullet-halfcheetah-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_expert.hdf5',
    'bullet-halfcheetah-medium-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_medium_expert.hdf5',
    'bullet-halfcheetah-medium-replay-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_medium_replay.hdf5',
    'bullet-hopper-random-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_random.hdf5',
    'bullet-hopper-medium-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_medium.hdf5',
    'bullet-hopper-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_expert.hdf5',
    'bullet-hopper-medium-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_medium_expert.hdf5',
    'bullet-hopper-medium-replay-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_medium_replay.hdf5',
    'bullet-ant-random-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_random.hdf5',
    'bullet-ant-medium-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_medium.hdf5',
    'bullet-ant-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_expert.hdf5',
    'bullet-ant-medium-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_medium_expert.hdf5',
    'bullet-ant-medium-replay-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_medium_replay.hdf5',
    'bullet-walker2d-random-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_random.hdf5',
    'bullet-walker2d-medium-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_medium.hdf5',
    'bullet-walker2d-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_expert.hdf5',
    'bullet-walker2d-medium-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_medium_expert.hdf5',
    'bullet-walker2d-medium-replay-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_medium_replay.hdf5',
    'bullet-maze2d-open-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-open-sparse.hdf5',
    'bullet-maze2d-umaze-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-umaze-sparse.hdf5',
    'bullet-maze2d-medium-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-medium-sparse.hdf5',
    'bullet-maze2d-large-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-large-sparse.hdf5',
}
def get_env_dims(env: str = "c4"):
    envir = env_dict[env]()
    return envir.state_dim, envir.action_dim

def download_dataset_from_url(dataset_url) -> str:
    dataset_filepath = "./"+dataset_url.split("/")[-1]
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath

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

def generate_random_data(batch_size=1000, env : str = "c4"):
    data_path = f'deep_learning_rl_sm/data/offline_{env}_data.pt'
    if os.path.exists(data_path):
        return data_path
    c4 = env_dict[env]()
    data = c4.generate_seq(batch_size)
    time_steps, action_masks, actions, returns_to_go, rewards, states, traj_masks = extract_dataset(data)

    # PADDING PARAMETERS
    padded_state = torch.full((22, 42), -100)  # Padding value for states
    padded_mask = torch.full((21, 7), -100)  # Padding for mask
    padded_else = torch.full((21, 1), -100)  # Padding value for other variables

    # PADDING STATES
    for idx, state in enumerate(states):
        pad_state = padded_state.clone()
        pad_state[:state.shape[0], :] = state.flatten(start_dim=-2)
        states[idx] = pad_state
    states = torch.stack(states)

    # PADDING OTHER VARIABLES (actions, rewards, dones, time_steps, action_masks, returns_to_go)
    for idx in range(len(actions)):
        pad_act = padded_else.clone()
        pad_rew = padded_else.clone()
        pad_dones = padded_else.clone()
        pad_time = padded_else.clone()
        pad_mask = padded_mask.clone()
        pad_rtg = padded_else.clone()

        pad_act[:actions[idx].shape[0]] = actions[idx].unsqueeze(-1)
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

def test_c4_generate():
    dp = generate_random_data()

    loaded_data = torch.load(dp)

    # Access individual items from the loaded dictionary
    s = loaded_data['states']
    a = loaded_data['actions']
    r = loaded_data['rewards']
    d = loaded_data['dones']
    t_s = loaded_data['time_steps']
    a_m = loaded_data['action_masks']
    r_t_g = loaded_data['returns_to_go']
    print(s.shape)  # state trajectories have one more timestep than the rest
    print(a.shape)
    print(r.shape)
    print(d.shape)
    print(t_s.shape)
    print(a_m.shape)
    print(r_t_g.shape)
