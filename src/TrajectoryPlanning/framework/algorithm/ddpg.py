import config
import math
import models
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import utils.fileio
import utils.math
import utils.print
import utils.string_utils
from envs.game_state import GameStateBase
from framework.agent import AgentBase

critic_checkpoint_file = 'critic'
actor_checkpoint_file = 'actor'
critic_targ_checkpoint_file = 'critic_targ'
actor_targ_checkpoint_file = 'actor_targ'

class DDPG (AgentBase):
    def __init__(self, dim_state: int, dim_action: int, model_group: str, name: str = None) -> None:
        super().__init__(dim_state, dim_action, model_group, name)

        # Load configs.
        self.batchsize = self.configs.get(config.Training.Agent.BatchSize_)
        self.gamma = self.configs.get(config.Training.Agent.Gamma_)
        self.lr_actor = self.configs.get(config.Training.Agent.LRActor_)
        self.lr_critic = self.configs.get(config.Training.Agent.LRCritic_)
        self.tau = self.configs.get(config.Training.Agent.Tau_)

        # Initialize models.
        self.critic = models.create(self.model_group + '/critic', dim_state, dim_action)
        self.actor = models.create(self.model_group + '/actor', dim_state, dim_action)
        self.critic_targ = models.create(self.model_group + '/critic', dim_state, dim_action)
        self.actor_targ = models.create(self.model_group + '/actor', dim_state, dim_action)

        utils.math.hard_update(self.critic_targ, self.critic) # make sure target is with the same weight.
        utils.math.hard_update(self.actor_targ, self.actor)
        self.__init_optim()

        # Initialize losses.
        self.critic_loss = nn.MSELoss()

        # Plots.
        self.plot_critic_loss = 'critic_loss'
        self.plot_actor_loss = 'actor_loss'
        self.plot_manager.create_plot(self.plot_critic_loss, 'Critic Loss', 'Loss')
        self.plot_manager.create_plot(self.plot_actor_loss, 'Actor Loss', 'Loss')

    def __init_optim(self) -> None:
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

    def sample_action(self, state: GameStateBase, deterministic: bool) -> np.ndarray:
        state = torch.tensor(state.as_input(), dtype=config.Common.DataType.Torch)
        return self.actor(state).detach().numpy()

    def learn(self):
        # Sample BatchSize transitions from replay buffer for optimization.
        sampled_trans = self.replay_buffer.sample(self.batchsize)

        # Convert to ndarray.
        states = np.array([t.state.as_input() for t in sampled_trans], dtype=config.Common.DataType.Numpy)
        actions = np.array([t.action for t in sampled_trans], dtype=config.Common.DataType.Numpy)
        rewards = np.array([t.reward for t in sampled_trans], dtype=config.Common.DataType.Numpy)[:, np.newaxis]
        next_states = np.array([t.next_state.as_input() for t in sampled_trans], dtype=config.Common.DataType.Numpy)

        # Convert to torch tensor.
        states = torch.tensor(states, dtype=config.Common.DataType.Torch)
        actions = torch.tensor(actions, dtype=config.Common.DataType.Torch)
        rewards = torch.tensor(rewards, dtype=config.Common.DataType.Torch)
        next_states = torch.tensor(next_states, dtype=config.Common.DataType.Torch)

        # Optimize critic.
        next_actions_targ = self.actor_targ(next_states)
        next_q_targ = self.critic_targ(next_states, next_actions_targ)
        q_targ = rewards + self.gamma * next_q_targ
        q_pred = self.critic(states, actions)
        critic_loss = self.critic_loss(q_pred, q_targ)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        critic_loss_val = critic_loss.item()
        assert not math.isnan(critic_loss_val)
        self.plot_manager.push(self.plot_critic_loss, critic_loss_val)

        # Optimize actor.
        actions_pred = self.actor(states)
        actor_q_pred = self.critic(states, actions_pred)
        actor_loss = -torch.mean(actor_q_pred)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        actor_loss_val = actor_loss.item()
        assert not math.isnan(actor_loss_val)
        self.plot_manager.push(self.plot_actor_loss, actor_loss_val)

        # Update targets.
        utils.math.soft_update(self.actor_targ, self.actor, self.tau)
        utils.math.soft_update(self.critic_targ, self.critic, self.tau)

        # [optional] Update transition priority.
        if self.configs.get(config.Training.Agent.PER.Enabled_):
            priorities = critic_loss_val
            priorities **= self.configs.get(config.Training.Agent.PER.Alpha_)
            priorities *= self.configs.get(config.Training.Agent.PER.K_)
            for i in range(len(sampled_trans)):
                sampled_trans[i].p = float(priorities[i][0])

    def _save(self, path: str) -> None:
        torch.save(self.critic.state_dict(), path + critic_checkpoint_file)
        torch.save(self.actor.state_dict(), path + actor_checkpoint_file)
        torch.save(self.critic_targ.state_dict(), path + critic_targ_checkpoint_file)
        torch.save(self.actor_targ.state_dict(), path + actor_targ_checkpoint_file)

    def _load(self, path: str) -> None:
        # Check files.
        path = utils.string_utils.to_folder_path(path)
        paths = {
            'critic': path + critic_checkpoint_file,
            'actor': path + actor_checkpoint_file,
            'critic_targ': path + critic_targ_checkpoint_file,
            'actor_targ': path + actor_targ_checkpoint_file,
        }
        for p in paths.values():
            if not os.path.exists(p):
                raise FileNotFoundError()

        # Load models.
        self.critic.load_state_dict(torch.load(paths['critic']))
        self.actor.load_state_dict(torch.load(paths['actor']))
        self.critic_targ.load_state_dict(torch.load(paths['critic_targ']))
        self.actor_targ.load_state_dict(torch.load(paths['actor_targ']))
        self.__init_optim()