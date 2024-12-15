import argparse

import numpy as np
import torch
import wandb
from deep_learning_rl_sm.trainer.trainer import Trainer
from deep_learning_rl_sm.neuralnets.minimal_reinformer import MinimalReinformer
from deep_learning_rl_sm.neuralnets.minGRU_Reinformer import minGRU_Reinformer
from deep_learning_rl_sm.neuralnets.lamb import Lamb
from deep_learning_rl_sm.environments import connect_four
from data.custom_dataset import CustomDataset

# TODO generate our datasets separately so we only have to load them here! input format!
env = connect_four.ConnectFour()
# maybe push generate sequences into the trainer class somewhere


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", choices=["reinformer"], default="reinformer")
parser.add_argument("--env", type=str, default=env)
parser.add_argument("--env_discrete", type=bool, default=True)
parser.add_argument("--dataset", type=str, default="medium")
parser.add_argument("--num_eval_ep", type=int, default=10)
parser.add_argument("--max_eval_ep_len", type=int, default=1000)
parser.add_argument("--dataset_dir", type=str, default="data/d4rl_dataset/")
parser.add_argument("--context_len", type=int, default=5)
parser.add_argument("--min_rnn", type=str, default="minGRU")
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
parser.add_argument("--use_wandb", action='store_true', default=False)
parser.add_argument("--block_type", choices=["MinGruBlockV1", "MinGruBlockV2"], default="MinGruBlockV1")
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

# old model
"""model = MinimalReinformer(state_dim=env.state_dim, act_dim=env.action_dim, n_blocks=args["n_blocks"],min_rnn=args["min_rnn"],
                          h_dim=args["embed_dim"], context_len=args["context_len"], n_heads=args["n_heads"],
                          drop_p=args["dropout_p"], init_tmp=args["init_temperature"],discrete=args["env_discrete"],
                          target_entropy=target_entropy)"""

# new model
model = minGRU_Reinformer(state_dim=env.state_dim, act_dim=env.action_dim, n_blocks=args["n_blocks"],
                          h_dim=args["embed_dim"], context_len=args["context_len"], n_heads=args["n_heads"],
                          drop_p=args["dropout_p"], init_tmp=args["init_temperature"],discrete=args["env_discrete"],
                          target_entropy=target_entropy, block_type=args["block_type"])
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

datapath = "data/offline_data.pt"
dataset = CustomDataset(datapath)
# expect the next line to cause problems (need padding to make this work)
trainer = Trainer(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler, parsed_args=args)
trainer.train(args)
