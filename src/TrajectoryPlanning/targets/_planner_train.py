import config
import framework.algorithm.her
import numpy as np
import time
import utils.print
import utils.string_utils
from copy import copy
from envs import create_simulator
from envs.gazebo.state import GazeboState
from framework.agent import create_agent
from framework.configuration import global_configs as configs
from framework.evaluator import Evaluator
from framework.noise.ou import OrnsteinUhlenbeckProcess
from framework.noise.uniform import UniformNoise
from framework.replay_buffer import Transition

class PlannerState(GazeboState):
    def __init__(self, state: GazeboState, plan: np.ndarray=None):
        super().__init__()
        self.achieved = state.achieved
        self.desired = state.desired
        self.collision = state.collision
        self.joint_position = state.joint_position
        self.plan = copy(state.achieved) if plan is None else plan

    def _as_input(self) -> np.ndarray:
        rel_pos_to_achieved = self.plan - self.achieved
        rel_pos_to_desired = self.desired - self.plan
        return np.concatenate([
            super()._as_input(),
            self.plan,
            rel_pos_to_achieved,
            rel_pos_to_desired,
        ], dtype=config.Common.DataType.Numpy)

def main():
    # Load from configs.
    epsilon = configs.get(config.Agent.ActionNoise.Normal.Epsilon_)
    her_enabled = configs.get(config.Agent.HER.Enabled_)
    her_k = configs.get(config.Agent.HER.K_)
    load_from_previous = configs.get(config.Training.LoadFromPreviousSession_)
    max_epoches = configs.get(config.Training.MaxEpoches_)
    max_iters = configs.get(config.Environment.MaxIterations_)
    noise_enabled = configs.get(config.Agent.ActionNoise.Normal.Enabled_)
    protected_epoches = configs.get(config.Training.ProtectedEpoches_)
    warmup_steps = configs.get(config.Agent.Warmup_)

    # Initialize environment.
    sim = create_simulator('gazebo')
    game = sim.reward()

    state = sim.reset()
    dim_action = sim.dim_action()
    dim_state = state.dim_state()

    planner_state = PlannerState(state)
    dim_action_planner = len(planner_state.plan)
    dim_state_planner = planner_state.dim_state()

    # Initialize noise.
    warmup_noise = UniformNoise(dim_action_planner, -1, 1)
    normal_noise = OrnsteinUhlenbeckProcess(dim_action_planner,
        theta=configs.get(config.Agent.ActionNoise.Normal.Theta_),
        mu=configs.get(config.Agent.ActionNoise.Normal.Mu_),
        sigma=configs.get(config.Agent.ActionNoise.Normal.Sigma_))

    # Load agents.
    agent_joint_solver = create_agent('sac', 'sac/l3', dim_state, dim_action, name='joint_solver')
    assert agent_joint_solver.load(enable_learning=False)

    agent_planner = create_agent('sac', 'sac/l3', dim_state_planner, dim_action_planner, name='planner')
    evaluator = Evaluator(agent_planner)
    if load_from_previous:
        evaluator.load()

    # ~
    trained_epoches = 0
    last_update_time = time.time()
    last_update_step = 0
    
    while evaluator.epoches <= max_epoches:
        warmup = evaluator.steps < warmup_steps
        if not warmup:
            trained_epoches += 1

        epoch_replay_buffer: list[Transition] = []
        game.reset()
        state = sim.reset()
        plan = copy(state.achieved)
        done = False

        while not done and evaluator.iterations <= max_iters:
            # Sample an action from planner.
            state = sim.state()
            planner_state = PlannerState(state, copy(plan))
            if warmup:
                planner_action = warmup_noise.sample()
            else:
                planner_action = agent_planner.sample_action(planner_state, deterministic=False)
                if noise_enabled:
                    noise_amount = 1 - evaluator.steps / epsilon
                else:
                    noise_amount = 0
                noise_amount = max(noise_amount, 0)
                planner_action += normal_noise.sample() * noise_amount
            planner_action = planner_action.clip(-1, 1)
            plan += planner_action * 0.1
            plan = plan.clip([-1.2, -1.2, 0], [1.2, 1.2, 1.2])
            sim.place_marker('marker_green', plan)

            # Sample an action from joint solver.
            state.desired = plan
            action = agent_joint_solver.sample_action(state, deterministic=True)
            state = sim.step(action)
            next_planner_state = PlannerState(state, copy(plan))

            # Calculate reward and add to replay buffer.
            reward, done = game.update(action, state)
            trans = Transition(planner_state, planner_action, reward, next_planner_state)
            agent_planner.replay_buffer.append(trans)
            epoch_replay_buffer.append(trans)
    
            # [optional] Optimize & Save the agent.
            if not warmup:
                agent_planner.learn()

            # Evaluation & Logging.
            evaluator.step(reward)
            if time.time() - last_update_time > 1:
                utils.print.put('[Train] %s' %
                    utils.string_utils.dict_to_str(evaluator.summary(shortterm=True)), same_line=True)
                last_update_time = time.time()
        
        # [optional] Perform HER.
        if her_enabled:
            epoch_replay_buffer = framework.algorithm.her.her(epoch_replay_buffer, her_k, game)
            for trans in epoch_replay_buffer:
                agent_planner.replay_buffer.append(trans)

        # Evaluation & logging.
        evaluator.epoch(allow_save=trained_epoches > protected_epoches)
        if (evaluator.steps - last_update_step) >= config.Training.MinLogStepInterval:
            last_update_step = evaluator.steps
            utils.print.put('[Evaluate] ' + utils.string_utils.dict_to_str(evaluator.summary()))
    sim.close()