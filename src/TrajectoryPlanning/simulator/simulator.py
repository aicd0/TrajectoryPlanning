import matlab
import numpy as np
from framework.state import State

def __state(eng) -> State:
    state = State()
    state.from_matlab(eng.workspace['state'])
    return state

def initialize(eng) -> None:
    eng.simInitialize(nargout=0)

def reset(eng) -> State:
    eng.simReset(nargout=0)
    return __state(eng)

def step(eng, action) -> State:
    action = action[:, np.newaxis].tolist() # row order
    eng.workspace['action'] = matlab.double(action)
    eng.simStep(nargout=0)
    return __state(eng)

def stage(eng) -> State:
    eng.simStage(nargout=0)
    return __state(eng)