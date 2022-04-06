import numpy as np
from simulator.MATLAB.game_state import GameState
from typing import Tuple

class Game:
    def __init__(self) -> None:
        pass

    def __distance2reward(self, d: float) -> float:
        return 10 / (d * 5 + 1)

    def __update(self, action: np.ndarray, state: GameState) -> None:
        self.__reward = 0
        self.__done = False
        # self.__done = any([
        #     state.self_collision,
        #     state.world_collision,
        #     state.deadlock,
        # ])

        if state.collision:
            return

        # Calculate the distance to the target point.
        d = np.linalg.norm(state.achieved - state.desired)
        self.__reward = self.__distance2reward(d)

    def reset(self) -> None:
        self.__steps = 0

    def update(self, action: np.ndarray, state: GameState) -> Tuple:
        self.__update(action, state)
        self.__steps += 1
        if self.__steps >= 100:
            self.__done = True
        return self.__reward, self.__done