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
        
    # TODO explore different target entropies
    # entropy to encourage exploration in RL typically -action_dim for continuous actions and -log(action_dim) when discrete
    target_entropy = -np.log(np.prod(action_dim)) if discrete else -np.prod(action_dim)
    args = vars(args)
    model = minGRU_Reinformer(state_dim=state_dim, act_dim=action_dim, n_blocks=args["n_blocks"],
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
    # expect the next line to cause problems (need padding to make this work)
    trainer = Trainer(model=model, dataset=data, optimizer=optimizer, scheduler=scheduler, parsed_args=args)
    trainer.train(args)

if __name__ == "__main__":
    main()