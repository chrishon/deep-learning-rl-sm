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
from torch.utils.data import Dataset, DataLoader
import random

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
    # use_wandb = False
    parser.add_argument("--use_wandb", action='store_true', default=False)
    return parser.parse_args()

class D4RLDataset(Dataset):
    def __init__(self,s,a,rtg,seq_len):
        self.s = s
        self.s_shape = list(s[0].shape)
        self.a = a
        self.a_shape = list(a[0].shape)
        self.rtg = rtg
        self.rtg_shape = list(rtg[0].shape)
        self.seq_len = seq_len
        self.rng = random.randint
                
    def __len__(self):
        return len(self.s)    
        
    def __getitem__(self, idx):
        si = self.rng(0, self.s_shape[0] - self.seq_len)
        s,a,rtg = self.s[idx][si:si+self.seq_len],self.a[idx][si:si+self.seq_len],self.rtg[idx][si:si+self.seq_len]
        pad_len = self.seq_len - s.shape[0]
        s = torch.cat([torch.from_numpy(s),torch.zeros([pad_len]+self.s_shape[1:])], dim=0)
        a = torch.cat([torch.from_numpy(a),torch.zeros([pad_len]+self.a_shape[1:])], dim=0)
        rtg = torch.cat([torch.from_numpy(rtg),torch.zeros([pad_len]+self.rtg_shape[1:])], dim=0)
        mask = torch.cat([torch.ones(s.shape[0]),torch.zeros(pad_len)],dim=0).type(torch.int8)
        t = torch.arange(start=0,end=self.seq_len,step=1)
        return (t,s,a,rtg,mask)
         

if __name__ == "__main__":
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    device = args.device
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if args.use_wandb:
        wandb.init(
            name=args.env + "-" + args.dataset,
            project="Reinformer",
            config=vars(args)
        )
    env = args.env+"_"+args.dataset+"-v2"
    fp = download_dataset_from_url(env)
    max_ep_len = 1000 #Same for all 3 envs (Hopper, Walker, HalfCheetah)
    scale = 1000 # Normalization for rewards/returns
    if args.env in ["walker2d"]:
            env_name = "Walker2D"
    if args.env in ["hopper"]:
            env_name = "Hopper"
    if args.env in ["halfcheetah"]:
            scale = 5000
            env_name = "HalfCheetah"
    observations, acts, rew_to_gos,state_mean,state_std = benchmark_data(fp)
    environment = gym.make(env_name + "-v2")
    def get_normalized_score(score, env = env):
        return (score - REF_MIN_SCORE[env]) / (REF_MAX_SCORE[env] - REF_MIN_SCORE[env])
    def evaluator(model):
            return_mean, _, _, _ = Reinformer_eval(
                model=model,
                device=device,
                context_len=max_ep_len,
                env = environment,
                state_mean=state_mean,
                state_std=state_std,
                num_eval_ep=args["num_eval_ep"],
                max_test_ep_len=args["max_eval_ep_len"]
            )
            return get_normalized_score(
                return_mean
            ) * 100
    state_dim, act_dim = observations[0].shape[1], acts[0].shape[1]
    # entropy to encourage exploration in RL typically -action_dim for continuous actions and -log(action_dim) when discrete
    args = vars(args)
    target_entropy = -np.log(np.prod(act_dim)) if args["env_discrete"] else -np.prod(act_dim)
    model = minGRU_Reinformer(state_dim=state_dim, act_dim=act_dim,
                            h_dim=args["embed_dim"], n_layers=args["n_layers"],
                            drop_p=args["dropout_p"], init_tmp=args["init_temperature"],
                            target_entropy=target_entropy, discrete=args["env_discrete"], batch_size = args["batch_size"], device=device, max_timestep=max_ep_len)
    model=model.to(device)
    #torch.compile(model = model, mode="max-autotune")
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
    #perhaps dynamically incease K
    dataset = D4RLDataset(observations,acts,rew_to_gos,args["K"])
    del observations,acts,rew_to_gos
    traj_data_loader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    trainer = Trainer(model=model, data_loader = traj_data_loader, optimizer=optimizer, scheduler=scheduler, parsed_args=args, batch_size=args["batch_size"], device=device)
    del model, traj_data_loader
    torch.backends.cudnn.benchmark = True
    d4rl_norm_scores = []
    for it in range(args["max_iters"]):
        outputs = trainer.train_iteration_benchmark(num_steps=args['num_steps_per_iter'], iter_num=it+1, print_logs=True)
        #Eval
        with torch.no_grad():
            d4rl_norm_scores.append(evaluator(trainer.model))
        if args["use_wandb"]:
            wandb.log(outputs)
