import config
import utils.print
from envs.simulator import Simulator
from framework.planner import Planner

def benchmark(sim: Simulator, planner: Planner):
    total = config.Testing.MaxEpoches
    success = 0
    span = 0
    steps = 0

    for _ in range(total):
        state = sim.reset()
        sim.plot_reset()
        ret = planner.reach(state.desired, verbose=True)
        if ret:
            success += 1
            span += planner.span
            steps += planner.steps
            
    utils.print.put('Success rate: %.2f%%(%d/%d). Avg time used: %fms. Avg steps: %.2f.'
        % (success / total * 100, success, total, span * 1000 / success, steps / success))