from cgitb import enable
import config
import numpy as np
import utils.print
import utils.string_utils
from framework.algorithm.ddpg import DDPG as Agent
from framework.evaluator import Evaluator
from simulator import Game, Simulator

def main():
    # Initialize environment.
    sim = Simulator()
    game = Game()
    state = sim.reset()
    dim_action = sim.dim_action()
    dim_state = state.dim_state()

    # Initialize agent.
    agent = Agent(dim_state, dim_action, 'ddpg/l5')

    # Load evaluator.
    evaluator = Evaluator(agent)
    evaluator.load(enable_learning=False)

    # Logging.
    step_rewards = []
    epoch_rewards = []

    for _ in range(config.Test.MaxEpoches):
        state = sim.reset()
        game.reset()
        sim.plot_reset()
        done = False
        iteration = 0
        epoch_reward = 0

        while not done and iteration < config.Test.MaxIterations:
            iteration += 1
            action = agent.sample_action(state)
            state = sim.step(action)
            reward, done = game.update(action, state)
            epoch_reward += reward
            step_rewards.append(reward)
            sim.plot_step()
        
        epoch_rewards.append(epoch_reward)

    sim.close()

    step_rewards = np.array(step_rewards, dtype=config.DataType.Numpy)
    epoch_rewards = np.array(epoch_rewards, dtype=config.DataType.Numpy).reshape(-1, 1)
    
    res = {}
    res['EpRwd'] = np.mean(epoch_rewards)
    res['EpRwdStd'] = np.std(epoch_rewards)
    res['StpRwd'] = np.mean(step_rewards)
    res['StpRwdStd'] = np.std(step_rewards)
    utils.print.put('[Test] ' + utils.string_utils.dict_to_str(res))