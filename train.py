import argparse
import h5py
import numpy as np
import torch
import wandb
from deep_learning_rl_sm.trainer.trainer import Trainer
from deep_learning_rl_sm.neuralnets.minGRU_Reinformer import minGRU_Reinformer
from deep_learning_rl_sm.neuralnets.lamb import Lamb
from deep_learning_rl_sm.environments import connect_four
from deep_learning_rl_sm.utils import download_dataset_from_url,BENCHMARK_LIST,generate_random_data,DATASET_URLS,get_env_dims,state_dims,act_dims

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["reinformer"], default="reinformer")
    parser.add_argument("--dataset", type=str,choices=BENCHMARK_LIST + ["c4"], default="c4")
    parser.add_argument("--num_eval_ep", type=int, default=10)
    parser.add_argument("--max_eval_ep_len", type=int, default=1000)
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--grad_norm", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_train_iters", type=int, default=10)
    parser.add_argument("--num_updates_per_iter", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)
    # use_wandb = False
    parser.add_argument("--use_wandb", action='store_true', default=False)
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    dataset = args.dataset
    if BENCHMARK_LIST.__contains__(dataset):
        #Benchmark Envs
        discrete = False
        dataset_filepath = download_dataset_from_url(DATASET_URLS[dataset])
        f = h5py.File(dataset_filepath,"r")
        data = f[list(f.keys())[0]][()]
        d_env = dataset.split("-")[0]
        state_dim,action_dim = state_dims[d_env], act_dims[d_env]
    else:
        #Our Envs
        discrete = True
        dataset_filepath = generate_random_data(args.batch_size, args.dataset)
        data = torch.load(dataset_filepath)
        state_dim, action_dim = get_env_dims(args.dataset)
        
    if args.use_wandb:
        wandb.init(
            name= args.dataset,
            project="Reinformer",
            config=vars(args)
        )

scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / args["warmup_steps"], 1)
        )

import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        # Storing each data item separately for easy access
        self.states = data['states']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        self.time_steps = data['time_steps']
        self.action_masks = data['action_masks']
        self.returns_to_go = data['returns_to_go']
        
    def __len__(self):
        # Return the number of samples (assuming all lists have the same length)
        return len(self.states)

    def __getitem__(self, idx):
        # Return a tuple of each item type for a given index
        return (self.states[idx,:,:], 
                self.actions[idx,:,:], 
                self.rewards[idx,:,:], 
                self.dones[idx,:,:], 
                self.time_steps[idx,:,:], 
                self.action_masks[idx,:,:], 
                self.returns_to_go[idx,:,:])
dataset = CustomDataset(data)
# expect the next line to cause problems (need padding to make this work)
trainer = Trainer(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler, parsed_args=args)
trainer.train(args)

