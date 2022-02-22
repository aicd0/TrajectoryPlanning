import numpy as np
from framework.state import State

class GameState:
    def __init__(self) -> None:
        self.max_d2 = 1

    def update(self, action: np.ndarray, next_state: State, test=False) -> None:
        self.reward = 0.
        self.stage_over = False
        self.game_over = False

        # On self-collision.
        if next_state.self_collision:
            self.reward = -100.
            self.game_over = True
            return

        # On world-collision.
        if next_state.world_collision:
            self.reward = -100.
            self.game_over = True
            return

        # On goal achieved.
        d2 = np.square(next_state.achieved - next_state.desired).sum()

        if d2 < self.max_d2:
            self.reward = 100.
            self.stage_over = True
            return

        # Other cases.
        self.reward = -1.

    def __level_up(self):
        self.max_d2 = max(0.01, self.max_d2 * 0.95)