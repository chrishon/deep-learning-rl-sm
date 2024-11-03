import random
import gymnasium as gym
import numpy as np
import torch
import wandb
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, dataset, criterion, optimizer: Optimizer, parsed_args, batch_size: int = 32,
                 learning_rate: float = 1e-3, num_epochs: int = 10, device=None):
        self.tau = parsed_args["tau"]
        self.grad_norm = parsed_args["grad_norm"]
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define DataLoader
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Define optimizer and loss function
        self.optimizer = optimizer
        self.criterion = criterion

    def train_step(
            self,
            timesteps,
            states,
            actions,
            returns_to_go,
            rewards,
            traj_mask
    ):
        # TODO add necessary args to class during init e.g. tau
        self.model.train()
        # data to gpu ------------------------------------------------
        timesteps = timesteps.to(self.device)  # B x T
        states = states.to(self.device)  # B x T x state_dim
        actions = actions.to(self.device)  # B x T x act_dim
        returns_to_go = returns_to_go.to(self.device).unsqueeze(
            dim=-1
        )  # B x T x 1

        rewards = rewards.to(self.device).unsqueeze(
            dim=-1
        )  # B x T x 1
        traj_mask = traj_mask.to(self.device)  # B x T

        # model forward ----------------------------------------------
        (
            returns_to_go_preds,
            actions_dist_preds,
            _,
        ) = self.model.forward(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
        )

        returns_to_go_target = torch.clone(returns_to_go).view(
            -1, 1
        )[
            traj_mask.view(-1, ) > 0
            ]
        returns_to_go_preds = returns_to_go_preds.view(-1, 1)[
            traj_mask.view(-1, ) > 0
            ]

        # returns_to_go_loss -----------------------------------------
        norm = returns_to_go_target.abs().mean()
        u = (returns_to_go_target - returns_to_go_preds) / norm
        returns_to_go_loss = torch.mean(
            torch.abs(
                self.tau - (u < 0).float()
            ) * u ** 2
        )
        # action_loss ------------------------------------------------
        actions_target = torch.clone(actions)
        log_likelihood = actions_dist_preds.log_prob(
            actions_target
        ).sum(axis=2)[
            traj_mask > 0
            ].mean()
        entropy = actions_dist_preds.entropy().sum(axis=2).mean()
        action_loss = -(log_likelihood + self.model.temperature().detach() * entropy)

        loss = returns_to_go_loss + action_loss

        # optimization -----------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_norm
        )
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
                self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        self.scheduler.step()

        return loss.detach().cpu().item()

    def train(self, parsed_args):
        self.model.train()
        seed = parsed_args["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = parsed_args["env"]
        dataset = parsed_args["dataset"]
        parsed_args["batch_size"] = 16 if dataset == "complete" else parsed_args["batch_size"] = 256
        if env in ["kitchen", "maze2d", "antmaze"]:
            parsed_args["num_eval_ep"] = 100
        # TODO set data path
        # dataset_path = os.path.join(variant["dataset_dir"], f"{d4rl_env}.pkl")
        device = torch.device(parsed_args["device"])

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=parsed_args["batch_size"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        iterate_data = iter(data_loader)
        # TODO implement get_state_stats for our envs???
        state_mean, state_std = dataset.get_state_stats()

        # TODO initialize env random seed

        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        model_type = parsed_args["model_type"]

        max_train_iters = parsed_args["max_train_iters"]
        num_updates_per_iter = parsed_args["num_updates_per_iter"]
        score_list_normalized = []
        for _ in range(1, max_train_iters + 1):
            for epoch in range(num_updates_per_iter):
                try:
                    (
                        timesteps,
                        states,
                        actions,
                        returns_to_go,
                        rewards,
                        traj_mask,
                    ) = next(iterate_data)
                except StopIteration:
                    iterate_data = iter(data_loader)  # start again with original load
                    (
                        timesteps,
                        states,
                        actions,
                        returns_to_go,
                        rewards,
                        traj_mask,
                    ) = next(iterate_data)

                loss = self.train_step(
                    timesteps=timesteps,
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go,
                    rewards=rewards,
                    traj_mask=traj_mask
                )
                if parsed_args["use_wandb"]:
                    wandb.log(
                        data={
                            "training/loss": loss,
                        }
                    )
            normalized_score = evaluator(
                model=Trainer.model
            )
            score_list_normalized.append(normalized_score)
            if parsed_args["use_wandb"]:
                wandb.log(
                    data={
                        "evaluation/score": normalized_score
                    }
                )

        if parsed_args["use_wandb"]:
            wandb.log(
                data={
                    "evaluation/max_score": max(score_list_normalized),
                    "evaluation/last_score": score_list_normalized[-1]
                }
            )
        print(score_list_normalized)
        print("finished training!")

    def evaluate(self, dataset):
        """Evaluate the model on a given dataset."""
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_states, batch_actions, batch_rewards in data_loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)

                action_logits = self.model(batch_states)

                # Flatten tensors for computing loss
                action_logits = action_logits.view(-1, action_logits.size(-1))
                batch_actions = batch_actions.view(-1)

                # Compute loss
                loss = self.criterion(action_logits, batch_actions)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

    def predict(self, states):
        self.model.eval()
        with torch.no_grad():
            states = states.to(self.device)
            action_logits = self.model(states)
            predicted_actions = torch.argmax(action_logits, dim=-1)
        return predicted_actions.cpu().numpy()
