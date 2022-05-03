import config
import math
import models
import numpy as np
import torch
import utils.fileio
import utils.math
import utils.print
import utils.string_utils
from envs.state import State
from framework.agent import Agent
from torch import nn
from torch import optim

checkpoint_file = 'checkpoint.pt'

class DDPG(Agent):
    def __init__(self, *arg, **kwarg) -> None:
        super().__init__(*arg, **kwarg)

        # Load configs.
        self.batchsize = self.configs.get(config.Agent.BatchSize_)
        self.gamma = self.configs.get(config.Agent.Gamma_)
        self.lr_actor = self.configs.get(config.Agent.LRActor_)
        self.lr_critic = self.configs.get(config.Agent.LRCritic_)
        self.tau = self.configs.get(config.Agent.Tau_)

        # Initialize models.
        self.critic = models.create(self.model_group + '/critic', self.dim_state, self.dim_action)
        self.actor = models.create(self.model_group + '/actor', self.dim_state, self.dim_action)
        self.critic_targ = models.create(self.model_group + '/critic', self.dim_state, self.dim_action)
        self.actor_targ = models.create(self.model_group + '/actor', self.dim_state, self.dim_action)

        utils.math.hard_update(self.critic_targ, self.critic) # make sure target is with the same weight.
        utils.math.hard_update(self.actor_targ, self.actor)

        # Initialize optimizers.
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Initialize losses.
        self.critic_loss = nn.MSELoss()

        # Plots.
        self.plot_critic_loss = 'critic_loss'
        self.plot_actor_loss = 'actor_loss'
        self.plot_manager.create_plot(self.plot_critic_loss, 'Critic Loss', 'Loss')
        self.plot_manager.create_plot(self.plot_actor_loss, 'Actor Loss', 'Loss')

    def sample_action(self, state: State, deterministic: bool) -> np.ndarray:
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
        if self.configs.get(config.Agent.PER.Enabled_):
            priorities = critic_loss_val
            priorities **= self.configs.get(config.Agent.PER.Alpha_)
            priorities *= self.configs.get(config.Agent.PER.K_)
            for i in range(len(sampled_trans)):
                sampled_trans[i].p = float(priorities[i][0])

    def _save(self) -> None:
        torch.save({
            'critic': self.critic.state_dict(),
            'actor': self.actor.state_dict(),
            'critic_targ': self.critic_targ.state_dict(),
            'actor_targ': self.actor_targ.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_loss': self.critic_loss,
        }, self.save_dir + checkpoint_file)

    def _load(self) -> None:
        checkpoint = torch.load(self.save_dir + checkpoint_file)
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_targ.load_state_dict(checkpoint['critic_targ'])
        self.actor_targ.load_state_dict(checkpoint['actor_targ'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic_loss = checkpoint['critic_loss']