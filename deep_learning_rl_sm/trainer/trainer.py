import random
import numpy as np
import torch
import wandb
from torch.optim import Optimizer
import torch.nn as nn
import time
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, optimizer: Optimizer, scheduler, batch_size: int = 32,
                 learning_rate: float = 1e-3, num_epochs: int = 10, device=None, dataset=None, parsed_args=None):
        self.tau = parsed_args["tau"]
        self.grad_norm = parsed_args["grad_norm"]
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define DataLoader
        if dataset is not None:
            self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                          drop_last=True)
            self.data_iter = iter(self.data_loader)

        # Define optimizer and loss function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.env = parsed_args['env']

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_tmp],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

    def train_step(
            self,
            timesteps,
            states,
            actions,
            returns_to_go,
            rewards,
            traj_mask
    ):
        self.model.train()
        # data to gpu ------------------------------------------------
        timesteps = timesteps.to(self.device)  # B x T
        states = states[:, 1:, :].float()
        states = states.to(self.device)  # B x T x state_dim
        actions = actions.float()
        actions = actions.to(self.device)  # B x T x act_dim
        returns_to_go = returns_to_go.to(self.device).unsqueeze(
            dim=-1
        )  # B x T x 1
        returns_to_go = returns_to_go.float()
        min_timestep = timesteps.min()
        max_timestep = timesteps.max()
        rewards = rewards.to(self.device).unsqueeze(
            dim=-1
        )  # B x T x 1
        traj_mask = traj_mask.to(self.device)  # B x T

        # model forward ----------------------------------------------
        (
            returns_to_go_preds,
            actions_dist_preds,
            _,
        ) = self.model(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
        )

        returns_to_go_target = torch.clone(returns_to_go).view(
            -1, 1
        )[
            (traj_mask.view(-1, ) == 0) | (traj_mask.view(-1, ) == 1)
            ]
        returns_to_go_preds = returns_to_go_preds.view(-1, 1)[
            (traj_mask.view(-1, ) == 0) | (traj_mask.view(-1, ) == 1)
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
        # cannot calculate logprobs of out of dist actions need to fill with viable value for padding and then mask to 0
        actions_target = torch.clone(actions)
        mask = actions_target != -100  # padding for actions is -100 (better padding scheme probably necessary)
        actions_target_4logprob = actions_target.clone()
        actions_target_4logprob[~mask] = 0
        actions_target_4logprob = actions_target_4logprob.squeeze()

        actions_target_4logprob = torch.argmax(actions_target_4logprob, dim=-1)

        log_likelihood = (actions_dist_preds.log_prob(
            actions_target_4logprob
        )).unsqueeze(-1) * mask
        # print(log_likelihood.shape)
        log_likelihood = log_likelihood.sum(axis=2).view(-1)[
            (traj_mask.view(-1) == 0) | (traj_mask.view(-1) == 1)
            ].mean()

        entropy = actions_dist_preds.entropy().unsqueeze(-1).sum(axis=2).mean()
        action_loss = -(log_likelihood + self.model.temp().detach() * entropy)

        loss = returns_to_go_loss + action_loss
        print("Loss: " + str(loss))
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
                self.model.temp() * (entropy - self.model.target_entropy).detach()
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
        dataset = self.dataset  # parsed_args["dataset"]
        # parsed_args["batch_size"] = 16 if dataset == "complete" else 256
        if env in ["kitchen", "maze2d", "antmaze"]:
            parsed_args["num_eval_ep"] = 100

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=parsed_args["batch_size"],
            shuffle=True,
            drop_last=True
        )
        iterate_data = iter(data_loader)
        # model_type = parsed_args["model_type"]
        max_train_iters = parsed_args["max_train_iters"]
        num_updates_per_iter = parsed_args["num_updates_per_iter"]
        score_list_normalized = []

        num_updates = 0
        for _ in range(1, max_train_iters + 1):
            for epoch in range(num_updates_per_iter):
                print()
                print("iteration: " + str(epoch))
                if epoch % 10 == 0 and epoch != 0:
                    self.evaluate_online()
                try:
                    print("Trying to get batch")
                    (
                        states,
                        actions,
                        rewards,
                        traj_mask,
                        timesteps,
                        action_masks,
                        returns_to_go

                    ) = next(iterate_data)
                except StopIteration:
                    iterate_data = iter(data_loader)  # start again with original load
                    (
                        states,
                        actions,
                        rewards,
                        traj_mask,
                        timesteps,
                        action_masks,
                        returns_to_go

                    ) = next(iterate_data)

                loss = self.train_step(
                    timesteps=timesteps.squeeze(2),
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go.squeeze(2),
                    rewards=rewards,
                    traj_mask=traj_mask
                )
                if parsed_args["use_wandb"]:
                    wandb.log(
                        data={
                            "training/loss": loss,
                        }
                    )

                if num_updates % 50 == 0 and num_updates != 0:
                    win_loss_ratio_test = self.evaluate_online()
                    print("win loss ratio: " + str(win_loss_ratio_test))
                num_updates += 1

            normalized_score = self.evaluate(
                dataset=None
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

    def evaluate_online(self):
        ratio = 0
        for trial in range(50):
            # states vector is 1 timestep larger because of init state
            # flatten board state
            s0 = torch.flatten(torch.tensor(self.env.reset()[0]))
            # print("s0 shape: " + str(s0.shape[0]))
            states = torch.zeros((1, 21, s0.shape[0]))
            states[0, 0] = s0  # (B=1, T=1, State)
            timesteps = torch.tensor([i for i in range(21)]) + 1  # remember 0 is used for padding
            timesteps = timesteps.unsqueeze(0)
            actions = torch.full((21, 7), 0)
            returns_to_go = torch.full((21, 1), -100)

            states = states.float()
            timesteps = timesteps
            actions = actions.float()
            returns_to_go = returns_to_go.float()
            """print(f"Timestep online Shape: {timesteps.shape}")
            print(f"states online Shape: {states.shape}")
            print(f"actions online Shape: {actions.shape}")
            print(f"returns_to_go online Shape: {returns_to_go.shape}")"""
            while True:

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
                for timestep in timesteps[0]:
                    # print(f"Timestep: {timestep}")

                    action_mask = self.env.action_mask  # Should return a tensor of 1s/0s
                    action_mask = torch.tensor(action_mask, dtype=torch.float32).to(self.device)

                    # Apply mask to the probabilities
                    """print("action_mask", action_mask)
                    print("actions_dist_preds.probs[0]", actions_dist_preds.probs[0])"""
                    original_probs = actions_dist_preds.probs[0][timestep - 1]
                    masked_probs = original_probs * action_mask  # Mask out invalid actions
                    # print("masked_probs ", masked_probs)
                    masked_probs = masked_probs / masked_probs.sum()  # Renormalize
                    # print("masked_probs ", masked_probs)

                    # Create a new Categorical distribution with masked probabilities
                    masked_action_dist = torch.distributions.Categorical(probs=masked_probs)
                    act = masked_action_dist.sample().item()
                    # print(f"states: {states.shape}")

                    state, reward, done, _, _ = self.env.step(act)  # TODO
                    # print(f"state: {state.shape}")
                    actions[timestep - 1, :] = act
                    states[0, timestep - 1, :] = torch.flatten(torch.tensor(state))
                    """print("rtg shaoe1:, ", returns_to_go.shape)
                    print("rtg shaoe:, ", returns_to_go_preds.squeeze(0).shape)"""
                    returns_to_go = returns_to_go_preds.squeeze(0)
                    ratio += reward if reward == 1 else 0
                    if done:
                        break
                break
        ratio /= 50

        return ratio

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

    def train_iteration_benchmark(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        # training -----------------------------------------------
        self.model.train()
        for i in range(num_steps):
            if i % 50 == 0:
                print(f"Iteration {i}, time: {time.time() - train_start}")
            train_loss = self.train_step_benchmark()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        eval_start = time.time()

        # evaluation -----------------------------------------------
        """self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start"""

        # timing -----------------------------------------------
        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.time_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        # diagnostics -----------------------------------------------
        """for k in self.diagnostics:
            logs[k] = self.diagnostics[k]"""

        # prints ---------------------------------------------------
        if print_logs:
            print('=' * 80)
            print(f'Epoch {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def get_next(self):
        try:
            (
                timesteps,
                states,
                actions,
                returns_to_go,
                traj_mask,
            ) = next(self.data_iter)
        except StopIteration:
            del self.data_iter
            self.data_iter = iter(self.data_loader)
            (
                timesteps,
                states,
                actions,
                returns_to_go,
                traj_mask,
            ) = next(self.data_iter)
        return (
            timesteps,
            states,
            actions,
            returns_to_go,
            traj_mask,
        )

    def train_step_benchmark(self):
        timesteps, states, actions, rtg, traj_mask = self.get_next()
        timesteps = timesteps.to(self.device)  # B x T
        states = states.to(self.device)  # B x T x state_dim
        actions = actions.to(self.device)  # B x T x act_dim
        rtg = rtg.to(self.device).unsqueeze(
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
            returns_to_go=rtg,
        )
        returns_to_go_target = torch.clone(rtg).view(
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
        action_loss = -(log_likelihood + self.model.temp().detach() * entropy)

        loss = returns_to_go_loss + action_loss

        # optimizer -----------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_norm
        )
        self.optimizer.step()

        # scheduler -----------------------------------------------
        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
                self.model.temp() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        self.scheduler.step()

        # diagnostics -----------------------------------------------
        """with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()"""

        return loss.detach().cpu().item()
