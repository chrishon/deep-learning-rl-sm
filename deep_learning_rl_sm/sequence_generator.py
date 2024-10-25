import copy
from deep_learning_rl_sm.environments.our_gym import Our_Env
from environments import connect_four


def generate_seq(environment: Our_Env, no_sequences: int, max_seq_len=1000, actor=None, adv_actor= None):
    sequences = []
    act_space = environment.action_space
    for _ in range(no_sequences):
        first_state, _ = environment.reset()
        seq = [copy.deepcopy(first_state)]
        for _ in range(max_seq_len):
            action = act_space.sample(environment.action_mask)
            next_state, reward, done, _, _ = environment.step(action)
            seq.append(action)
            seq.append(reward)
            seq.append(copy.deepcopy(next_state))
            if done:
                break
        sequences.append(tuple(seq))
        print(sequences)
    return sequences


c4 = connect_four.connect_four_env()
generate_seq(c4, 1)
