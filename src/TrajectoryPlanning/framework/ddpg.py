import config
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from framework.model import Actor, Critic
from framework.state import State

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
        self.__buffer = []
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

        # Initialize optimizers.
        self.critic_optim = optim.SGD(self.critic.parameters(), lr=config.DDPG.LRCritic)
        self.actor_optim = optim.SGD(self.actor.parameters(), lr=config.DDPG.LRActor)

        # Initialize losses.
        self.critic_loss = nn.MSELoss()

    def sample_action(self, state: State):
        # Action prediction from actor.
        state = torch.tensor(state.as_input, dtype=config.TorchDType)
        action_pred = self.actor(state).detach().numpy()

        # Add noise.
        dim_action = len(action_pred)
        action_noise = config.DDPG.ActionNoise
        action_noise = np.random.uniform(-action_noise, action_noise, dim_action)
        action = np.add(action_pred, action_noise, dtype=config.NumpyDType)
        
        # Saturation.
        for i in range(len(action)):
            action[i] = max(-1, min(1, action[i]))

        return action

    def learn(self, iters):
        assert len(self.replay_buffer) >= config.DDPG.BatchSize

        critic_loss_sum = 0.
        actor_loss_sum = 0.

        test_period = 200
        critic_loss_period = 0.
        actor_loss_period = 0.

        for i in range(iters):
            # Sample BatchSize transitions from replay buffer for optimization.
            sampled_trans = self.replay_buffer.sample(config.DDPG.BatchSize)

            # Convert to ndarray.
            states = np.array([t.state.as_input for t in sampled_trans], dtype=config.NumpyDType)
            actions = np.array([t.action for t in sampled_trans], dtype=config.NumpyDType)
            rewards = np.array([t.reward for t in sampled_trans], dtype=config.NumpyDType)[:, np.newaxis]
            next_states = np.array([t.next_state.as_input for t in sampled_trans], dtype=config.NumpyDType)

            # Convert to torch tensor.
            states = torch.tensor(states, dtype=config.TorchDType)
            actions = torch.tensor(actions, dtype=config.TorchDType)
            rewards = torch.tensor(rewards, dtype=config.TorchDType)
            next_states = torch.tensor(next_states, dtype=config.TorchDType)

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
            Tau_ = 1 - Tau

            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.data = p.data * Tau + pt.data * Tau_
                
            for p, pt in zip(self.actor.parameters(), self.actor_targ.parameters()):
                pt.data = p.data * Tau + pt.data * Tau_

            # Statistics.
            critic_loss_val = critic_loss.item()
            actor_loss_val = actor_loss.item()
            critic_loss_sum += critic_loss_val
            actor_loss_sum += actor_loss_val
            critic_loss_period += critic_loss_val
            actor_loss_period += actor_loss_val

            if math.isnan(critic_loss_period) or math.isnan(actor_loss_period):
                raise RuntimeError()

            if (i + 1) % test_period == 0:
                c_loss = critic_loss_period / test_period
                a_loss = actor_loss_period / test_period
                print('C|A: %f\t%f\r' % (c_loss, a_loss), end='')
                critic_loss_period = 0.
                actor_loss_period = 0.

        print('Critic loss: %f' % (critic_loss_sum / iters))
        print('Actor loss: %f' % (actor_loss_sum / iters))