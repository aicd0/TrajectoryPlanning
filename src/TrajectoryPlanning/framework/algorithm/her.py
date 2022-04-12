import random
from copy import copy
from framework.replay_buffer import Transition
from simulator import Game

def augment_replay_buffer(replay_buffer: list[Transition], k: int) -> list[Transition]:
    old_replay_buffer: list[Transition] = []
    new_replay_buffer: list[Transition] = []
    game = Game()

    for trans in replay_buffer:
        if trans.state.support_her():
            old_replay_buffer.append(trans)

    for i, trans in enumerate(old_replay_buffer):
        # Sample k transitions after this transition as new goals.
        target_replay_buffer = old_replay_buffer[i + 1:]
        sample_count = min(k, len(target_replay_buffer))
        sampled_trans = random.sample(target_replay_buffer, sample_count)
        
        # Generate a new transition for each goal.
        for trans_goal in sampled_trans:
            new_state = copy(trans.state)
            new_state.desired = trans_goal.state.achieved
            new_state.update() # notify changes.

            new_next_state = copy(trans.next_state)
            new_next_state.desired = trans_goal.state.achieved
            new_next_state.update() # notify changes.

            new_action = trans.action

            game.reset()
            reward, _ = game.update(new_action, new_next_state)

            new_trans = Transition(new_state, new_action, reward, new_next_state)
            new_replay_buffer.append(new_trans)
    
    return new_replay_buffer