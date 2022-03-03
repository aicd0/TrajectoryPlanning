import numpy as np
from framework.state import State
from typing import Tuple

reward_self_collision = -1000
reward_world_collision = -1000
reward_deadlock = -100
reward_goal_achieved = 1000

class GameState:

    def __init__(self) -> None:
        self.__d2_to_reward = [
            [0.1, -1],
            [0.3, -2],
            [0.5, -3],
            [0.7, -4],
            [0.9, -5],
            [1.1, -6],
            [1.3, -7],
            [1.5, -8],
            [1.7, -9],
            [1.9, -10],
        ]
        for e in self.__d2_to_reward:
            e[0] *= e[0]

    def __level_up(self):
        self.__max_d2 = max(0.01, self.__max_d2 * 0.95)

    def __distance2reward(self, d2: float) -> Tuple[int, bool]:
        left = 0
        right = len(self.__d2_to_reward) - 1
        while (left != right):
            i = (left + right + 1) // 2
            if self.__d2_to_reward[i][0] <= d2:
                left = i
            else:
                right = i - 1
        if (d2 < self.__d2_to_reward[left][0]):
            return reward_goal_achieved, True
        return self.__d2_to_reward[left][1], False

    def __update(self, action: np.ndarray, next_state: State) -> None:
        self.reward = 0
        self.self_collision = False
        self.world_collision = False
        self.deadlock = False
        self.goal_achived = False

        if next_state.self_collision:
            # On self-collision.
            self.reward = reward_self_collision
            self.self_collision = True
            return
            
        if next_state.world_collision:
            # On world-collision.
            self.reward = reward_world_collision
            self.world_collision = True
            return

        if next_state.deadlock:
            # On deadlock
            self.reward = reward_deadlock
            self.deadlock = True
            return

        # Calculate the distance to the target point.
        d2 = np.square(next_state.achieved - next_state.desired).sum()
        self.reward, self.goal_achived = self.__distance2reward(d2)

    def reset(self) -> None:
        self.total_reward = 0.
        self.reward_counter = {}
        self.total_self_collision = 0
        self.total_world_collision = 0
        self.total_deadlock = 0
        self.total_goal_achived = 0

    def update(self, action: np.ndarray, next_state: State) -> None:
        self.__update(action, next_state)

        self.game_over = False
        self.stage_over = False
        if self.self_collision or self.world_collision or self.deadlock:
            self.game_over = True
        elif self.goal_achived:
            self.stage_over = True

        self.total_reward += self.reward
        self.reward_counter[self.reward] = self.reward_counter.get(self.reward, 0) + 1
        if self.self_collision:
            self.total_self_collision += 1
        if self.world_collision:
            self.total_world_collision += 1
        if self.deadlock:
            self.total_deadlock += 1
        if self.goal_achived:
            self.total_goal_achived += 1

    def summary(self) -> None:
        print('Total reward: %d (sc=%d, wc=%d, dl=%d, g=%d)' % (
            self.total_reward, self.total_self_collision, self.total_world_collision,
            self.total_deadlock, self.total_goal_achived))
        reward_count_str = []
        for key in sorted(self.reward_counter):
            reward_count_str.append(str(key) + '(' + str(self.reward_counter[key]) + ')')
        print('Rewards: [' + ', '.join(reward_count_str) + ']')
