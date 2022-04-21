import config
import numpy as np
import utils.math
from .game_state import GameState
from framework.configuration import global_configs as configs
from typing import Tuple

class Game:
    def __init__(self) -> None:
        # Load configs
        self.max_iterations = configs.get(config.Environment.ROS.MaxIterations_)

    def __update(self, action: np.ndarray, state: GameState) -> None:
        self.__reward = 0
        self.__done = False

        if state.collision:
            return

        # Calculate the distance to the target point.
        d = utils.math.distance(state.achieved, state.desired)
        self.__reward = Game.__distance2reward(d)

    def reset(self) -> None:
        self.__steps = 0

    def update(self, action: np.ndarray, state: GameState) -> Tuple:
        self.__update(action, state)
        self.__steps += 1
        if self.__steps >= self.max_iterations:
            self.__done = True
        return self.__reward, self.__done

    @staticmethod
    def __distance2reward(d: float) -> float:
        return 10 / (d * 5 + 1)