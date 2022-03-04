import config
import numpy as np
from math import sqrt
from simulator.MATLAB.game_state import GameState
from typing import Tuple

reward_self_collision = -1000
reward_world_collision = -1000
reward_deadlock = -1000
reward_goal_achieved = 1000

class Game:
    def __init__(self) -> None:
        pass

    def __distance2reward(self, d: float) -> float:
        return -d

    def __update(self, action: np.ndarray, next_state: GameState) -> None:
        self.__self_collision = False
        self.__world_collision = False
        self.__deadlock = False
        self.__goal_achived = False

        # Calculate the distance to the target point.
        d = sqrt(np.square(next_state.achieved - next_state.desired).sum())
        self.__reward = self.__distance2reward(d)
        
        if next_state.self_collision:
            # On self-collision.
            self.__reward += reward_self_collision
            self.__self_collision = True
            return
            
        if next_state.world_collision:
            # On world-collision.
            self.__reward += reward_world_collision
            self.__world_collision = True
            return

        if next_state.deadlock:
            # On deadlock
            self.__reward += reward_deadlock
            self.__deadlock = True
            return

        if d < 0.1:
            self.__target_point_hold_steps += 1
            if self.__target_point_hold_steps >= 20:
                self.__reward += reward_goal_achieved
                self.__goal_achived = True
        else:
            self.__target_point_hold_steps = 0

    def reset(self) -> None:
        self.__target_point_hold_steps = 0
        self.__rewards = []

    def update(self, action: np.ndarray, next_state: GameState) -> Tuple:
        self.__update(action, next_state)
        self.__done = self.__self_collision or self.__world_collision or self.__deadlock or self.__goal_achived
        self.__rewards.append(self.__reward)
        if len(self.__rewards) > 200:
            self.__done = True
        return self.__reward, self.__done

    def summary(self) -> None:
        rewards = np.array(self.__rewards, dtype=config.DataType.Numpy)
        reward_sum = np.sum(rewards)
        reward_std = np.std(rewards)
        print('Rwd=%f, RwdStd=%f (sc=%d, wc=%d, dl=%d, g=%d)' %
            (reward_sum, reward_std, self.__self_collision,
            self.__world_collision, self.__deadlock, self.__goal_achived))
