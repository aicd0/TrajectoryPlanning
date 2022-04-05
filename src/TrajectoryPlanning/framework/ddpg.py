import config
import framework.models
import json
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import utils.fileio
import utils.math
import utils.print
import utils.string_utils
from framework.configuration import Configuration
from framework.random_process import OrnsteinUhlenbeckProcess
from framework.replay_buffer import ReplayBuffer
from simulator import GameState

config_file = 'config.txt'
critic_checkpoint_file = 'critic'
actor_checkpoint_file = 'actor'
critic_targ_checkpoint_file = 'critic_targ'
actor_targ_checkpoint_file = 'actor_targ'
replay_buffer_checkpoint_file = 'replay_buffer.txt'

class Agent:
    __agent_number = 1

    def __init__(self, dim_state: int, dim_action: int, model: str, name: str = None) -> None:
        if name is None:
            self.name = 'Agent_' + str(Agent.__agent_number)
            Agent.__agent_number += 1
        else:
            self.name = name

        self.configs = Configuration('agent_' + self.name)
        self.dim_state = dim_state
        self.dim_action = dim_action

        # Initialize networks.
        self.critic = framework.models.create(model + '/critic', dim_state, dim_action)
        self.actor = framework.models.create(model + '/actor', dim_state, dim_action)
        self.critic_targ = framework.models.create(model + '/critic', dim_state, dim_action)
        self.actor_targ = framework.models.create(model + '/actor', dim_state, dim_action)

        utils.math.hard_update(self.critic_targ, self.critic) # make sure target is with the same weight.
        utils.math.hard_update(self.actor_targ, self.actor)

        # Initialize optimizers.
        self.__init_optimizer()

        # Initialize losses.
        self.critic_loss = nn.MSELoss()

        # Initialize the replay buffer.
        self.replay_buffer = ReplayBuffer(self.configs)

        # Initialize OU noise.
        self.random_process = OrnsteinUhlenbeckProcess(size=self.dim_action,
            theta=self.configs.get(config.Train.DDPG.OUNoise.FieldTheta),
            mu=self.configs.get(config.Train.DDPG.OUNoise.FieldMu),
            sigma=self.configs.get(config.Train.DDPG.OUNoise.FieldSigma))

    def __init_optimizer(self) -> None:
        self.critic_optim = optim.Adam(self.critic.parameters(),
            lr=self.configs.get(config.Train.DDPG.FieldLRCritic))
        self.actor_optim = optim.Adam(self.actor.parameters(),
            lr=self.configs.get(config.Train.DDPG.FieldLRActor))

    def save(self, path: str) -> None:
        path = utils.string_utils.to_folder_path(path)
        utils.fileio.mktree(path)
        
        # Save models.
        torch.save(self.critic.state_dict(), path + critic_checkpoint_file)
        torch.save(self.actor.state_dict(), path + actor_checkpoint_file)
        torch.save(self.critic_targ.state_dict(), path + critic_targ_checkpoint_file)
        torch.save(self.actor_targ.state_dict(), path + actor_targ_checkpoint_file)

        # Save replay buffer.
        replay_buffer_checkpoint_file_path = path + replay_buffer_checkpoint_file
        replay_buffer_checkpoint_temp_file_path = replay_buffer_checkpoint_file_path + '.tmp'
        with open(replay_buffer_checkpoint_temp_file_path, 'w') as f:
            tmp = self.replay_buffer.to_serializable()
            json.dump(tmp, f)
        if os.path.exists(replay_buffer_checkpoint_file_path):
            os.remove(replay_buffer_checkpoint_file_path)
        os.rename(replay_buffer_checkpoint_temp_file_path, replay_buffer_checkpoint_file_path)

    def load(self, path: str, learning_enabled: bool = True) -> bool:
        # Check files.
        path = utils.string_utils.to_folder_path(path)
        paths = {
            'critic': path + critic_checkpoint_file,
            'actor': path + actor_checkpoint_file,
            'critic_targ': path + critic_targ_checkpoint_file,
            'actor_targ': path + actor_targ_checkpoint_file,
            'replay_buffer': path + replay_buffer_checkpoint_file,
        }
        for p in paths.values():
            if not os.path.exists(p):
                return False

        # Load models.
        self.critic.load_state_dict(torch.load(paths['critic']))
        self.actor.load_state_dict(torch.load(paths['actor']))
        self.critic_targ.load_state_dict(torch.load(paths['critic_targ']))
        self.actor_targ.load_state_dict(torch.load(paths['actor_targ']))

        # [optional] Load replay buffer.
        if learning_enabled:
            with open(paths['replay_buffer'], 'r') as f:
                tmp = json.load(f)
                self.replay_buffer = ReplayBuffer.from_serializable(tmp, config.Train.DDPG.ReplayBuffer)

        self.__init_optimizer()
        utils.print.put('Agent loaded')
        return True

    def sample_random_action(self) -> np.ndarray:
        noise_min = self.configs.get(config.Train.DDPG.UniformNoise.FieldMin)
        noise_max = self.configs.get(config.Train.DDPG.UniformNoise.FieldMax)
        return np.random.uniform(noise_min, noise_max, self.dim_action)

    def sample_action(self, state: GameState, noise_amount: float = -1, detach: bool = False) -> np.ndarray:
        state = torch.tensor(state.as_input(), dtype=config.DataType.Torch)

        if detach:
            action = np.zeros((self.dim_action), config.DataType.Numpy)
        else:
            action = self.actor(state).detach().numpy()

        if noise_amount > 0:
            action_noise = self.random_process.sample() * noise_amount
            action = np.add(action, action_noise, dtype=config.DataType.Numpy)
        
        action = np.clip(action, -1, 1)
        return action

    def learn(self):
        batchsize = self.configs.get(config.Train.DDPG.FieldBatchSize)
        assert len(self.replay_buffer) >= batchsize

        # Sample BatchSize transitions from replay buffer for optimization.
        sampled_trans = self.replay_buffer.sample(batchsize)

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
        gamma = self.configs.get(config.Train.DDPG.FieldGamma)
        q_targ = rewards + gamma * next_q_targ
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

        # Value check.
        critic_loss_val = critic_loss.item()
        actor_loss_val = actor_loss.item()
        assert not math.isnan(critic_loss_val)
        assert not math.isnan(actor_loss_val)

        # Update target networks.
        Tau = self.configs.get(config.Train.DDPG.FieldTau)
        utils.math.soft_update(self.actor_targ, self.actor, Tau)
        utils.math.soft_update(self.critic_targ, self.critic, Tau)

        # [optional] Update transition priority.
        if self.configs.get(config.Train.DDPG.PER.FieldEnabled):
            priorities = torch.abs(q_pred - q_targ).detach().numpy()
            priorities **= self.configs.get(config.Train.DDPG.PER.FieldAlpha)
            priorities *= self.configs.get(config.Train.DDPG.PER.FieldK)
            for i in range(len(sampled_trans)):
                sampled_trans[i].p = float(priorities[i][0])