import config
import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import utils.fileio
import utils.string_utils
from framework.model import Actor, Critic
from framework.state import State

checkpoint_critic = 'critic'
checkpoint_actor = 'actor'
checkpoint_critic_targ = 'critic_targ'
checkpoint_actor_targ = 'actor_targ'

class Transition:
    def __init__(self, state: State, action: np.ndarray, reward: float, next_state: State) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class ReplayBuffer:
    def __init__(self, size: int) -> None:
        self.__capacity = size
        self.__size = 0
        self.__buffer: list[Transition] = []
        self.__begin = 0

    def __len__(self) -> int:
        return self.__size

    def append(self, transition: Transition) -> None:
        if self.__size < self.__capacity:
            self.__buffer.append(transition)
            self.__size += 1
        else:
            self.__buffer[self.__begin] = transition
            self.__begin += 1
            if self.__begin >= self.__size:
                self.__begin = 0

    def sample(self, count: int) -> list[Transition]:
        assert 0 <= count <= self.__size
        return random.sample(self.__buffer, count)

class Agent:
    def __init__(self, dim_state: int, dim_action: int) -> None:
        # Initialize critic network and actor network.
        self.critic = Critic(dim_state, dim_action)
        self.actor = Actor(dim_state, dim_action)

        # Initialize target critic network and target actor network.
        self.critic_targ = Critic(dim_state, dim_action)
        self.actor_targ = Actor(dim_state, dim_action)

        # Initialize the replay buffer.
        self.replay_buffer = ReplayBuffer(config.DDPG.ReplayBuffer)

        self.__init_model()

    def __init_model(self) -> None:
        # Initialize optimizers.
        self.critic_optim = optim.SGD(self.critic.parameters(), lr=config.DDPG.LRCritic)
        self.actor_optim = optim.SGD(self.actor.parameters(), lr=config.DDPG.LRActor)

        # Initialize losses.
        self.critic_loss = nn.MSELoss()

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
        self.critic.load_state_dict(torch.load(checkpoint_critic_path))
        self.actor.load_state_dict(torch.load(checkpoint_actor_path))
        self.critic_targ.load_state_dict(torch.load(checkpoint_critic_targ_path))
        self.actor_targ.load_state_dict(torch.load(checkpoint_actor_targ_path))
        self.__init_model()
        print('Agent loaded.')
        return True

    def sample_action(self, state: State, noise: bool, detach: bool = False):
        # Action prediction from actor.
        state = torch.tensor(state.as_input, dtype=config.DataType.Torch)
        if detach:
            action = np.zeros((1), config.DataType.Numpy)
        else:
            action = self.actor(state).detach().numpy()

        if noise:
            # Add noise.
            dim_action = len(action)
            action_noise = config.DDPG.NoiseAmount
            action_noise = np.random.uniform(-action_noise, action_noise, dim_action)
            action = np.add(action, action_noise, dtype=config.DataType.Numpy)
        
        # Saturation.
        for i in range(len(action)):
            action[i] = max(-1, min(1, action[i]))

        return action

    def learn(self):
        assert len(self.replay_buffer) >= config.DDPG.BatchSize

        critic_loss_sum = 0.
        actor_loss_sum = 0.

        test_period = 200
        critic_loss_period = 0.
        actor_loss_period = 0.

        for i in range(1, config.DDPG.MaxIterations + 1):
            # Sample BatchSize transitions from replay buffer for optimization.
            sampled_trans = self.replay_buffer.sample(config.DDPG.BatchSize)

            # Convert to ndarray.
            states = np.array([t.state.as_input for t in sampled_trans], dtype=config.DataType.Numpy)
            actions = np.array([t.action for t in sampled_trans], dtype=config.DataType.Numpy)
            rewards = np.array([t.reward for t in sampled_trans], dtype=config.DataType.Numpy)[:, np.newaxis]
            next_states = np.array([t.next_state.as_input for t in sampled_trans], dtype=config.DataType.Numpy)

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

            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.data.copy_(p.data * Tau + pt.data * (1 - Tau))
                
            for p, pt in zip(self.actor.parameters(), self.actor_targ.parameters()):
                pt.data.copy_(p.data * Tau + pt.data * (1 - Tau))

            # Statistics.
            critic_loss_val = critic_loss.item()
            actor_loss_val = actor_loss.item()
            critic_loss_sum += critic_loss_val
            actor_loss_sum += actor_loss_val
            critic_loss_period += critic_loss_val
            actor_loss_period += actor_loss_val

            if math.isnan(critic_loss_val) or math.isnan(actor_loss_val):
                raise RuntimeError()

            if i % test_period == 0:
                c_loss = critic_loss_period / test_period
                a_loss = actor_loss_period / test_period
                print('C|A: %.9g %.9g\r' % (c_loss, a_loss), end='')
                critic_loss_period = 0.
                actor_loss_period = 0.

        print('Critic loss: %f' % (critic_loss_sum / i))
        print('Actor loss: %f' % (actor_loss_sum / i))
        