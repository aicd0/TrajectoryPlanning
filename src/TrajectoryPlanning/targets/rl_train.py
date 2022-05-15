import config
import framework.algorithm.her
import time
import utils.print
import utils.string_utils
from envs import create_simulator
from framework.agent import create_agent
from framework.configuration import global_configs as configs
from framework.evaluator import Evaluator
from framework.noise.ou import OrnsteinUhlenbeckProcess
from framework.noise.uniform import UniformNoise
from framework.replay_buffer import Transition

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

    # Initialize noise.
    warmup_noise = UniformNoise(dim_action, -1, 1)
    normal_noise = OrnsteinUhlenbeckProcess(dim_action,
        theta=configs.get(config.Agent.ActionNoise.Normal.Theta_),
        mu=configs.get(config.Agent.ActionNoise.Normal.Mu_),
        sigma=configs.get(config.Agent.ActionNoise.Normal.Sigma_))

    # Load agent.
    agent = create_agent('sac', 'sac/l4', dim_state, dim_action, name='rl')
    evaluator = Evaluator(agent)
    if load_from_previous:
        evaluator.load()

    # Training progress.
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
        done = False

        while not done and evaluator.iterations <= max_iters:
            # Sample an action from agent.
            if warmup:
                action = warmup_noise.sample()
            else:
                action = agent.sample_action(state, deterministic=False)
                if noise_enabled:
                    noise_amount = 1 - evaluator.steps / epsilon
                else:
                    noise_amount = 0
                noise_amount = max(noise_amount, 0)
                action += normal_noise.sample() * noise_amount
            action = action.clip(-1, 1)
            next_state = sim.step(action)

            # Calculate reward and add to replay buffer.
            reward, done = game.update(state, action, next_state)
            trans = Transition(state, action, reward, next_state)
            agent.replay_buffer.append(trans)
            epoch_replay_buffer.append(trans)
    
            # [optional] Optimize & Save the agent.
            if not warmup:
                agent.learn()

            # Move to the next state.
            state = next_state

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
                agent.replay_buffer.append(trans)

        # Evaluation & Logging.
        evaluator.epoch(allow_save=trained_epoches > protected_epoches)
        if (evaluator.steps - last_update_step) >= config.Training.MinLogStepInterval:
            last_update_step = evaluator.steps
            utils.print.put('[Evaluate] ' + utils.string_utils.dict_to_str(evaluator.summary()))
    sim.close()