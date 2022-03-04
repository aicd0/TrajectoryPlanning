class GameState:
    def __init__(self):
        pass

    def from_reset(self, dim_state, dim_action, state) -> None:
        self.__dim_state = dim_state
        self.__dim_action = dim_action
        self.state = state

    def from_step(self, state, reward, done) -> None:
        self.state = state
        self.reward = reward
        self.done = done

    def as_input(self):
        return self.state

    def dim_state(self) -> int:
        return self.__dim_state

    def dim_action(self) -> int:
        return self.__dim_action