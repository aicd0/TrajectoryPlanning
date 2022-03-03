class GameState:
    def __init__(self):
        pass

    def from_reset(self, state_space, action_space, state) -> None:
        self.state_space = state_space
        self.action_space = action_space
        self.state = state

    def from_step(self, state, reward, done) -> None:
        self.state = state
        self.reward = reward
        self.done = done

    def as_input(self):
        return self.state

    def dim_state(self) -> int:
        return self.state_space.shape[0]

    def dim_action(self) -> int:
        return self.action_space.shape[0]