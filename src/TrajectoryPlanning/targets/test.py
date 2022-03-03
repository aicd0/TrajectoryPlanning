import config
from framework.ddpg import Agent
from framework.evaluator import Evaluator
from simulator.simulator import Simulator

def main():
    sim = Simulator()
    # sim.eng.simPlotInit(nargout=0)
    state = sim.reset()
    dim_action = state.dim_action()
    dim_state = state.dim_state()
    
    agent = Agent(dim_state, dim_action)
    agent.try_load(config.Model.CheckpointDir)
    
    evaluator = Evaluator(sim, 1)
    policy = lambda x: agent.sample_action(x, noise=config.Test.NoiseEnabled, detach=config.Test.DetachAgent)
    evaluator(policy, visualize=True, save=False)