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
from torch.distributions import Normal
from torch.optim import Adam

checkpoint_file = 'checkpoint.pt'

class SAC(Agent):
    def __init__(self, *arg, **kwarg) -> None:
        super().__init__(*arg, **kwarg)

        # Load configs.
        self.auto_entropy_tuning = self.configs.get(config.Agent.SAC.AutoEntropyTuning_)
        self.batchsize = self.configs.get(config.Agent.BatchSize_)
        self.gamma = self.configs.get(config.Agent.Gamma_)
        self.lr_alpha = self.configs.get(config.Agent.SAC.LRAlpha_)
        self.lr_actor = self.configs.get(config.Agent.LRActor_)
        self.lr_critic = self.configs.get(config.Agent.LRCritic_)
        self.tau = self.configs.get(config.Agent.Tau_)

        # Initialize models.
        self.q1 = models.create(self.model_group + '/critic', self.dim_state, self.dim_action)
        self.q2 = models.create(self.model_group + '/critic', self.dim_state, self.dim_action)
        self.q1_targ = models.create(self.model_group + '/critic', self.dim_state, self.dim_action)
        self.q2_targ = models.create(self.model_group + '/critic', self.dim_state, self.dim_action)
        self.actor = models.create(self.model_group + '/actor', self.dim_state, self.dim_action)
        self.log_alpha = torch.zeros(1, requires_grad=True)

        utils.math.hard_update(self.q1_targ, self.q1) # make sure target is with the same weight.
        utils.math.hard_update(self.q2_targ, self.q2)
        
        # Initialize optimizers.
        self.critic_optim = Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.lr_critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr_alpha)

        # Auto entropy tuning.
        self.target_entropy = -float(self.dim_action)
        self.alpha = self.log_alpha.exp()

        # Initialize losses.
        self.critic_loss = nn.MSELoss()

        # Plots.
        self.plot_critic_loss = 'critic_loss'
        self.plot_actor_loss = 'actor_loss'
        self.plot_manager.create_plot(self.plot_critic_loss, 'Critic Loss', 'Loss', window=20)
        self.plot_manager.create_plot(self.plot_actor_loss, 'Actor Loss', 'Loss', window=20)

        if self.auto_entropy_tuning:
            self.plot_log_alpha = 'log_alpha'
            self.plot_manager.create_plot(self.plot_log_alpha, 'Temperature', 'Ln alpha')

    def __sample_action(self, states: np.ndarray) -> np.ndarray:
        mean, log_std = self.actor(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_unbound = normal.rsample() # reparameterization trick
        actions = torch.tanh(action_unbound)
        log_prob = normal.log_prob(action_unbound) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return actions, log_prob, torch.tanh(mean)

    def sample_action(self, state: State, deterministic: bool) -> np.ndarray:
        state = torch.tensor(state.as_input()[np.newaxis, :], dtype=config.Common.DataType.Torch)
        action, _, mean = self.__sample_action(state)
        return (mean if deterministic else action).detach().numpy()[0]

    def learn(self):
        # Sample a minibatch from replay buffer for optimization.
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

        # Optimize critics.
        with torch.no_grad():
            next_actions, next_log_ps, _ = self.__sample_action(next_states)
            next_q1_targ = self.q1_targ(next_states, next_actions)
            next_q2_targ = self.q2_targ(next_states, next_actions)
            min_next_q_targ = torch.min(next_q1_targ, next_q2_targ) - self.alpha * next_log_ps
            q = rewards + self.gamma * min_next_q_targ

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = self.critic_loss(q1, q)
        q2_loss = self.critic_loss(q2, q)
        critic_loss = q1_loss + q2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        critic_loss_val = critic_loss.item()
        assert not math.isnan(critic_loss_val)
        self.plot_manager.push(self.plot_critic_loss, critic_loss_val)

        # Optimize actor.
        actions_pred, log_ps_pred, _ = self.__sample_action(states)
        q1_pred = self.q1(states, actions_pred)
        q2_pred = self.q2(states, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)
        actor_loss = (self.alpha * log_ps_pred - q_pred).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        actor_loss_val = actor_loss.item()
        assert not math.isnan(actor_loss_val)
        self.plot_manager.push(self.plot_actor_loss, actor_loss_val)

        # Optimize alpha.
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_ps_pred + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            alpha_loss_val = alpha_loss.item()
            assert not math.isnan(alpha_loss_val)
            self.plot_manager.push(self.plot_log_alpha, self.log_alpha.item())

            self.alpha = self.log_alpha.exp()

        # Update targets.
        utils.math.soft_update(self.q1_targ, self.q1, self.tau)
        utils.math.soft_update(self.q2_targ, self.q2, self.tau)

        # [optional] Update transition priority.
        if self.configs.get(config.Agent.PER.Enabled_):
            priorities = torch.abs(q1 - q) + torch.abs(q2 - q)
            priorities **= self.configs.get(config.Agent.PER.Alpha_)
            priorities *= self.configs.get(config.Agent.PER.K_)
            for i in range(len(sampled_trans)):
                sampled_trans[i].p = float(priorities[i][0])

    def _save(self) -> None:
        torch.save({
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_targ': self.q1_targ.state_dict(),
            'q2_targ': self.q2_targ.state_dict(),
            'actor': self.actor.state_dict(),
            'log_alpha': self.log_alpha,
            'critic_optim': self.critic_optim.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'critic_loss': self.critic_loss,
        }, self.save_dir + checkpoint_file)

    def _load(self) -> None:
        checkpoint = torch.load(self.save_dir + checkpoint_file)
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_targ.load_state_dict(checkpoint['q1_targ'])
        self.q2_targ.load_state_dict(checkpoint['q2_targ'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.log_alpha: torch.Tensor = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp()
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
        self.critic_loss = checkpoint['critic_loss']