import argparse

import numpy as np
import torch
import wandb
from deep_learning_rl_sm.trainer.trainer import Trainer
from deep_learning_rl_sm.neuralnets.minGRU_Reinformer import minGRU_Reinformer
from deep_learning_rl_sm.neuralnets.lamb import Lamb
from deep_learning_rl_sm.environments import connect_four

# TODO generate our datasets separately so we only have to load them here! input format!
env = connect_four.ConnectFour()
# maybe push generate sequences into the trainer class somewhere
data = torch.load("deep_learning_rl_sm/data/offline_data.pt")

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", choices=["reinformer"], default="reinformer")
parser.add_argument("--env", type=str, default=env)
parser.add_argument("--dataset", type=str, default="medium")
parser.add_argument("--num_eval_ep", type=int, default=10)
parser.add_argument("--max_eval_ep_len", type=int, default=1000)
parser.add_argument("--dataset_dir", type=str, default="data/d4rl_dataset/")
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
args = parser.parse_args()

if args.use_wandb:
    wandb.init(
        name=args.env + "-" + args.dataset,
        project="Reinformer",
        config=vars(args)
    )
# TODO have discrete set based on environment choice instead of manual
discrete = True
# TODO explore different target entropies
# entropy to encourage exploration in RL typically -action_dim for continuous actions and -log(action_dim) when discrete
target_entropy = -np.log(np.prod(env.action_dim)) if discrete else -np.prod(env.action_dim)
args = vars(args)
model = minGRU_Reinformer(state_dim=env.state_dim, act_dim=env.action_dim, n_blocks=args["n_blocks"],
                          h_dim=args["embed_dim"], context_len=args["context_len"], n_heads=args["n_heads"],
                          drop_p=args["dropout_p"], init_tmp=args["init_temperature"],
                          target_entropy=target_entropy)
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
