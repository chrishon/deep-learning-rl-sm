import argparse
import wandb
from deep_learning_rl_sm.trainer.trainer import Trainer
from deep_learning_rl_sm.neuralnets.minGRU_Reinformer import minGRU_Reinformer
from deep_learning_rl_sm.environments.connect_four import ConnectFour

env = ConnectFour()
# maybe push generate sequences into the trainer class somewhere
sequences = env.generate_seq(no_sequences=100, max_seq_len=1000)

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
# use_wandb = False
parser.add_argument("--use_wandb", action='store_true', default=False)
args = parser.parse_args()

if args.use_wandb:
    wandb.init(
        name=args.env + "-" + args.dataset,
        project="Reinformer",
        config=vars(args)
    )
# TODO fill in args for model and trainer!
model = minGRU_Reinformer()
trainer = Trainer(model=model,)
trainer.train(vars(args))
