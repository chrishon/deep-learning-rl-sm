import argparse
import gymnasium as gym
import numpy as np
import torch
import wandb
from eval import Reinformer_eval
from deep_learning_rl_sm.utils import *
from deep_learning_rl_sm.trainer.trainer import Trainer
from deep_learning_rl_sm.neuralnets.minGRU_Reinformer import minGRU_Reinformer
from deep_learning_rl_sm.neuralnets.lamb import Lamb
from deep_learning_rl_sm.environments import connect_four
from torch.utils.data import Dataset
import random

def cumsum(x):
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        cumsum[t] = x[t] + cumsum[t+1]
    return cumsum

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["reinformer"], default="reinformer")
    parser.add_argument("--env", default="halfcheetah")
    parser.add_argument("--env_discrete", type=bool, default=False)
    parser.add_argument("--dataset",choices=["medium","medium_expert","medium_replay"], type=str, default="medium")
    parser.add_argument("--num_eval_ep", type=int, default=10)
    parser.add_argument("--max_eval_ep_len", type=int, default=1000)
    parser.add_argument("--dataset_dir", type=str, default="deep_learning_rl_sm/benchmarks/data/halfcheetah_medium_expert-v2.hdf5")
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--grad_norm", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--num_steps_per_iter", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--use_wandb", action='store_true', default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if args.use_wandb:
        wandb.login()
        wandb.init(
            name=args.env + "-" + args.dataset,
            project="Reinformer",
            force=True,
            config=vars(args)
        )
    max_ep_len = 1000 #Same for all 3 envs (Hopper, Walker, HalfCheetah)
    scale = 1000 # Normalization for rewards/returns
    observations, acts,rewards, rew_to_gos, dones, traj_lengths,reward_sum = benchmark_data(args.dataset_dir)
    env = args.env+"_"+args.dataset+"-v2"
    def get_normalized_score(score, env = env):
        return (score - REF_MIN_SCORE[env]) / (REF_MAX_SCORE[env] - REF_MIN_SCORE)
    def evaluator(model):
            return_mean, _, _, _ = Reinformer_eval(
                model=model,
                device=device,
                context_len=args["context_len"],
                env = env,
                state_mean=state_mean,
                state_std=state_std,
                num_eval_ep=args["num_eval_ep"],
                max_test_ep_len=args["max_eval_ep_len"]
            )
            return get_normalized_score(
                return_mean
            ) * 100
    obs_concat = np.concatenate(observations, axis=0)
    state_mean, state_std = np.mean(obs_concat, axis=0), np.std(obs_concat, axis=0) + 1e-6
    state_dim, act_dim = observations[0].shape[1], acts[0].shape[1]
    # entropy to encourage exploration in RL typically -action_dim for continuous actions and -log(action_dim) when discrete
    args = vars(args)
    target_entropy = -np.log(np.prod(act_dim)) if args["env_discrete"] else -np.prod(act_dim)
    model = minGRU_Reinformer(state_dim=state_dim, act_dim=act_dim, n_blocks=args["n_blocks"],
                            h_dim=args["embed_dim"], context_len=args["context_len"], n_layers=args["n_layers"],
                            drop_p=args["dropout_p"], init_tmp=args["init_temperature"],
                            target_entropy=target_entropy, discrete=args["env_discrete"])
    model=model.to(device)
    optimizer = Lamb(
        model.parameters(),
        lr=args["lr"],
        weight_decay=args["wd"],
        eps=args["eps"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args["warmup_steps"], 1)
    )
    K = args["K"]
    num_trajectories = len(observations)
    sorted_inds = np.argsort(reward_sum) #Sort by highest total returns
    #p_sample = traj_lengths / sum(traj_lengths) #Sample trajectories based off their length
    
    def get_batch(batch_size=256, max_len=K):
        batch_inds = rng.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=False,
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            ob = observations[int(sorted_inds[batch_inds[i]])]
            ac = acts[int(sorted_inds[batch_inds[i]])]
            rew = rewards[int(sorted_inds[batch_inds[i]])]
            rew_to_go = rew_to_gos[int(sorted_inds[batch_inds[i]])]
            done = dones[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, rew.shape[0] - max_len) #Reinformer style

            # get sequences from dataset
            s.append(np.array(ob[si:si + max_len]).reshape(1, -1, state_dim))
            a.append(np.array(ac[si:si + max_len]).reshape(1, -1, act_dim))
            r.append(np.array(rew[si:si + max_len]).reshape(1, -1, 1))
            rtg.append(np.array(rew_to_go[si:si + max_len]).reshape(1, -1, 1))
            d.append(np.array(done[si:si + max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            #This is maybe bad impl. ?!
            """rtg.append(cumsum(rew[si:])[:s[-1].shape[1] + 1].reshape(1, -1, 1)) #1 element larger?!
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)"""

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.zeros((1, max_len - tlen)), d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask
    
    trainer = Trainer(model=model, get_batch = get_batch, optimizer=optimizer, scheduler=scheduler, parsed_args=args, batch_size=args["batch_size"], device=device)
    
    for it in range(args["max_iters"]):
        outputs = trainer.train_iteration_benchmark(num_steps=args['num_steps_per_iter'], iter_num=it+1, print_logs=True)
