import config
import numpy as np
import utils.print
import utils.string_utils
from envs import create_environment
from framework.agent import create_agent
from framework.configuration import global_configs as configs
from framework.evaluator import Evaluator

def main():
    # Initialize environment.
    sim, game = create_environment('gazebo')
    state = sim.reset()
    dim_action = sim.dim_action()
    dim_state = state.dim_state()

    # Load agent.
    agent = create_agent('sac', 'sac/l3', dim_state, dim_action, name='joint_solver')
    evaluator = Evaluator(agent)
    evaluator.load(enable_learning=False)

    # Progress.
    step_rewards = []
    epoch_rewards = []

    for _ in range(config.Testing.MaxEpoches):
        state = sim.reset()
        game.reset()
        sim.plot_reset()
        done = False
        iteration = 0
        epoch_reward = 0

        while not done and iteration < config.Testing.MaxIterations:
            iteration += 1
            action = agent.sample_action(state, deterministic=True)
            state = sim.step(action)
            reward, done = game.update(action, state)
            epoch_reward += reward
            step_rewards.append(reward)
            sim.plot_step()
        epoch_rewards.append(epoch_reward)
    sim.close()

    step_rewards = np.array(step_rewards, dtype=config.Common.DataType.Numpy)
    epoch_rewards = np.array(epoch_rewards, dtype=config.Common.DataType.Numpy).reshape(-1, 1)
    
    res = {}
    res['EpRwd'] = np.mean(epoch_rewards)
    res['EpRwdStd'] = np.std(epoch_rewards)
    res['StpRwd'] = np.mean(step_rewards)
    res['StpRwdStd'] = np.std(step_rewards)
    utils.print.put('[Test] ' + utils.string_utils.dict_to_str(res))