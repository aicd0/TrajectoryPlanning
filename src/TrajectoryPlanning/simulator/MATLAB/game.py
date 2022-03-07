import config
import numpy as np
from math import sqrt
from simulator.MATLAB.game_state import GameState
from typing import Tuple

class Game:
    def __init__(self) -> None:
        pass

    def __distance2reward(self, d: float) -> float:
        return 10 / (d * 5 + 1)

    def __update(self, action: np.ndarray, next_state: GameState) -> None:
        self.__self_collision = False
        self.__world_collision = False
        self.__deadlock = False
        self.__goal_achieved = False
        self.__reward = 0
        
        if next_state.self_collision:
            # On self-collision.
            self.__self_collision = True
            return
            
        if next_state.world_collision:
            # On world-collision.
            self.__world_collision = True
            return

        if next_state.deadlock:
            # On deadlock.
            self.__deadlock = True
            return

        # Calculate the distance to the target point.
        d = sqrt(np.square(next_state.achieved - next_state.desired).sum())
        self.__reward = self.__distance2reward(d)

        if d < 0.1:
            # On goal achieved.
            self.__goal_achieved = True

    def reset(self) -> None:
        self.__rewards = []

    def update(self, action: np.ndarray, next_state: GameState) -> Tuple:
        self.__update(action, next_state)
        self.__rewards.append(self.__reward)
        self.__done = any([
            self.__self_collision,
            self.__world_collision,
            self.__deadlock,
            len(self.__rewards) >= 150
        ])
        return self.__reward, self.__done

    def summary(self) -> None:
        rewards = np.array(self.__rewards, dtype=config.DataType.Numpy)
        reward_sum = np.sum(rewards)
        reward_std = np.std(rewards)
        print('Rwd=%f, RwdStd=%f (sc=%d, wc=%d, dl=%d, g=%d)' %
            (reward_sum, reward_std, self.__self_collision,
            self.__world_collision, self.__deadlock, self.__goal_achieved))
