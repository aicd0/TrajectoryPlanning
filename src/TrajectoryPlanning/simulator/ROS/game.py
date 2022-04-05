import numpy as np
import utils.math
from simulator.ROS.game_state import GameState
from typing import Tuple

class Game:
    def __init__(self) -> None:
        pass

    def __update(self, action: np.ndarray, state: GameState) -> None:
        self.__reward = 0
        self.__done = False

        if self.__done:
            return

        if state.collision:
            self.__reward = 0
            return

        # Calculate the distance to the target point.
        d = utils.math.distance(state.achieved, state.desired)
        self.__reward = Game.__distance2reward(d)

    def reset(self) -> None:
        self.__steps = 0

    def update(self, action: np.ndarray, state: GameState) -> Tuple:
        self.__update(action, state)
        self.__steps += 1
        if self.__steps >= 100:
            self.__done = True
        return self.__reward, self.__done

    @staticmethod
    def __distance2reward(d: float) -> float:
        return 10 / (d * 5 + 1)