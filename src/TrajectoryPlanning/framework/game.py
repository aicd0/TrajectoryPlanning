import numpy as np
from framework.state import State

class GameState:
    def __init__(self) -> None:
        self.__max_d2 = 0.2**2

    def reset(self) -> None:
        self.total_reward = 0.
        self.total_self_collision = 0
        self.total_world_collision = 0
        self.total_goal_achived = 0

    def update(self, action: np.ndarray, next_state: State, test=False) -> None:
        self.reward = 0.
        self.stage_over = False
        self.game_over = False

        if next_state.self_collision:
            # On self-collision.
            self.reward = -100.
            self.total_self_collision += 1
            self.game_over = True
            
        elif next_state.world_collision:
            # On world-collision.
            self.reward = -100.
            self.total_world_collision += 1
            self.game_over = True
            
        else:
            d2 = np.square(next_state.achieved - next_state.desired).sum()
            if d2 < self.__max_d2:
                # On goal achieved.
                self.reward = 100.
                self.total_goal_achived += 1
                self.stage_over = True

            # Other cases.
            self.reward = -1.

        self.total_reward += self.reward

    def summary(self) -> str:
        return 'Total reward: %d (sc=%d, wc=%d, g=%d)' % (
            self.total_reward, self.total_self_collision,
            self.total_world_collision, self.total_goal_achived)

    def __level_up(self):
        self.__max_d2 = max(0.01, self.__max_d2 * 0.95)
        