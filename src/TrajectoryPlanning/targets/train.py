import config
import random
import time
from copy import copy
from framework.ddpg import Agent
from framework.evaluator import Evaluator
from framework.replay_buffer import Transition
from simulator.simulator import Game, Simulator

def augment_replay_buffered(replay_buffer: list[Transition]) -> list[Transition]:
    res: list[Transition] = []
    game = Game()
    game.reset()

    for i, trans in enumerate(replay_buffer):
        # Sample k transitions after this transition as new goals.
        sample_src = replay_buffer[i + 1:]
        sample_count = min(config.HER.K, len(sample_src))
        sampled_trans = random.sample(sample_src, sample_count)
        
        # Generate a new transition for each goal.
        for trans_goal in sampled_trans:
            new_state = copy(trans.state)
            new_state.desired = trans_goal.state.achieved
            new_state.update() # notify changes.

            new_next_state = copy(trans.next_state)
            new_next_state.desired = trans_goal.state.achieved
            new_next_state.update() # notify changes.

            new_action = trans.action
            reward, _, _ = game.update(new_action, new_next_state)
            new_trans = Transition(new_state, new_action, reward, new_next_state)
            res.append(new_trans)
    
    return res

def main():
    sim = Simulator()
    game = Game()
    
    # Reset simulator to get sizes of states and actions.
    state = sim.reset()
    dim_action = state.dim_action()
    dim_state = state.dim_state()

    # Initialize the agent.
    agent = Agent(dim_state, dim_action)
    # agent.try_load(config.Model.CheckpointDir)

    # Initialize evaluator.
    evaluator = Evaluator(sim, config.DDPG.Evaluation.MaxEpisodes)

    step = 0
    last_evaluation_step = 0
    last_log_time = time.time()

    for episode in range(1, config.DDPG.MaxEpisodes + 1):
        episode_replay_buffer: list[Transition] = []
        game.reset()
        state = sim.reset()
        iteration = 0

        while iteration < config.DDPG.MaxIterations:
            iteration += 1
            step += 1

            # Sample an action and execute.
            if step < config.DDPG.Warmup:
                action = agent.sample_random_action()
            else:
                action = agent.sample_action(state, noise=config.DDPG.NoiseEnabled)
            next_state = sim.step(action)

            # Calculate reward and add to replay buffer.
            reward, game_over, stage_over = game.update(action, next_state)
            trans = Transition(state, action, reward, next_state)
            agent.replay_buffer.append(trans)
            episode_replay_buffer.append(trans)
    
            # [optional] Optimize & save the agent.
            if step >= config.DDPG.Warmup:
                agent.learn()
            if step % config.Model.SaveStepInterval == 0:
                agent.save(config.Model.CheckpointDir)

            if game_over:
                break
            elif stage_over:
                next_state = sim.stage()
            state = next_state

            if time.time() - last_log_time > 1:
                print('Ep=%d, Iter=%d, Step=%d       \r' % (episode, iteration, step), end='')
                last_log_time = time.time()
        
        game.summary()
        
        # [optional] Perform HER.
        if config.HER.Enabled:
            episode_replay_buffer = augment_replay_buffered(episode_replay_buffer)
            for trans in episode_replay_buffer:
                agent.replay_buffer.append(trans)

        # [optional] Evaluate.
        if (step - last_evaluation_step) >= config.DDPG.Evaluation.MinStepInterval:
            last_evaluation_step = step
            policy = lambda x: agent.sample_action(x, noise=False)
            validate_reward = evaluator(policy, step=step)
            print('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))