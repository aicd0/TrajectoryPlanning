import numpy as np
from simulator.Pendulum_v1.game_state import GameState
from typing import Tuple

class Game:
    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def update(self, action: np.ndarray, next_state: GameState) -> Tuple:
        return next_state.reward, next_state.done, False

    def summary(self) -> None:
        pass