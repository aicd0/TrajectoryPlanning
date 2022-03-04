import config
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import utils.fileio
import utils.math
import utils.string_utils
from framework.model import Actor, Critic
from framework.random_process import OrnsteinUhlenbeckProcess
from framework.replay_buffer import ReplayBuffer
from simulator.targets import GameState

checkpoint_critic = 'critic'
checkpoint_actor = 'actor'
checkpoint_critic_targ = 'critic_targ'
checkpoint_actor_targ = 'actor_targ'

class Agent:
    def __init__(self, dim_state: int, dim_action: int) -> None:
        self.dim_state = dim_state
        self.dim_action = dim_action

        # Initialize critic network and actor network.
        self.critic = Critic(dim_state, dim_action)
        self.actor = Actor(dim_state, dim_action)
        self.__init_optimizer()

        # Initialize target critic network and target actor network.
        self.critic_targ = Critic(dim_state, dim_action)
        self.actor_targ = Actor(dim_state, dim_action)

        utils.math.hard_update(self.critic_targ, self.critic)
        utils.math.hard_update(self.actor_targ, self.actor) # make sure target is with the same weight.

        # Initialize losses.
        self.critic_loss = nn.MSELoss()

        # Initialize the replay buffer.
        self.replay_buffer = ReplayBuffer(config.DDPG.ReplayBuffer)

        # Initialize OU noise.
        self.random_process = OrnsteinUhlenbeckProcess(size=dim_action,
            theta=config.DDPG.OUNoise.Theta, mu=config.DDPG.OUNoise.Mu,
            sigma=config.DDPG.OUNoise.Sigma)
        self.epsilon = 1.0

    def __init_optimizer(self) -> None:
        # Initialize optimizers.
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config.DDPG.LRCritic)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.DDPG.LRActor)

    def save(self, path: str) -> None:
        path = utils.string_utils.to_folder_path(path)
        utils.fileio.mktree(path)
        torch.save(self.critic.state_dict(), path + checkpoint_critic)
        torch.save(self.actor.state_dict(), path + checkpoint_actor)
        torch.save(self.critic_targ.state_dict(), path + checkpoint_critic_targ)
        torch.save(self.actor_targ.state_dict(), path + checkpoint_actor_targ)

    def try_load(self, path: str) -> bool:
        path = utils.string_utils.to_folder_path(path)
        checkpoint_critic_path = path + checkpoint_critic
        checkpoint_actor_path = path + checkpoint_actor
        checkpoint_critic_targ_path = path + checkpoint_critic_targ
        checkpoint_actor_targ_path = path + checkpoint_actor_targ
        if not os.path.exists(checkpoint_critic_path):
            return False
        if not os.path.exists(checkpoint_actor_path):
            return False
        if not os.path.exists(checkpoint_critic_targ_path):
            return False
        if not os.path.exists(checkpoint_actor_targ_path):
            return False
        try:
            self.critic.load_state_dict(torch.load(checkpoint_critic_path))
            self.actor.load_state_dict(torch.load(checkpoint_actor_path))
            self.critic_targ.load_state_dict(torch.load(checkpoint_critic_targ_path))
            self.actor_targ.load_state_dict(torch.load(checkpoint_actor_targ_path))
        except RuntimeError:
            return False
        self.__init_optimizer()
        print('Agent loaded.')
        return True

    def sample_random_action(self) -> np.ndarray:
        return np.random.uniform(config.DDPG.UniformNoise.Min, config.DDPG.UniformNoise.Max, self.dim_action)

    def sample_action(self, state: GameState, noise: bool, detach: bool = False) -> np.ndarray:
        # Action prediction from actor.
        state = torch.tensor(state.as_input(), dtype=config.DataType.Torch)
        if detach:
            action = np.zeros((self.dim_action), config.DataType.Numpy)
        else:
            action = self.actor(state).detach().numpy()

        if noise:
            action_noise = self.random_process.sample()
            action_noise *= max(self.epsilon, 0)
            action = np.add(action, action_noise, dtype=config.DataType.Numpy)
            self.epsilon -= 1 / config.DDPG.Epsilon
        
        action = np.clip(action, -1., 1.)
        return action

    def learn(self):
        assert len(self.replay_buffer) >= config.DDPG.BatchSize

        # Sample BatchSize transitions from replay buffer for optimization.
        sampled_trans = self.replay_buffer.sample(config.DDPG.BatchSize)

        # Convert to ndarray.
        states = np.array([t.state.as_input() for t in sampled_trans], dtype=config.DataType.Numpy)
        actions = np.array([t.action for t in sampled_trans], dtype=config.DataType.Numpy)
        rewards = np.array([t.reward for t in sampled_trans], dtype=config.DataType.Numpy)[:, np.newaxis]
        next_states = np.array([t.next_state.as_input() for t in sampled_trans], dtype=config.DataType.Numpy)

        # Convert to torch tensor.
        states = torch.tensor(states, dtype=config.DataType.Torch)
        actions = torch.tensor(actions, dtype=config.DataType.Torch)
        rewards = torch.tensor(rewards, dtype=config.DataType.Torch)
        next_states = torch.tensor(next_states, dtype=config.DataType.Torch)

        # Optimize critic network.
        next_actions_targ = self.actor_targ(next_states)
        next_q_targ = self.critic_targ(next_states, next_actions_targ)
        q_targ = rewards + config.DDPG.Gamma * next_q_targ
        q_pred = self.critic(states, actions)
        critic_loss = self.critic_loss(q_pred, q_targ)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Optimize actor network.
        actions_pred = self.actor(states)
        actor_q_pred = self.critic(states, actions_pred)
        actor_loss = -torch.mean(actor_q_pred)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update target networks.
        Tau = config.DDPG.Tau
        utils.math.soft_update(self.actor_targ, self.actor, Tau)
        utils.math.soft_update(self.critic_targ, self.critic, Tau)

        # Statistics.
        critic_loss_val = critic_loss.item()
        actor_loss_val = actor_loss.item()

        if math.isnan(critic_loss_val) or math.isnan(actor_loss_val):
            raise RuntimeError()