from simulator.targets import Game, GameState, Simulator

def main():
    # sim = Simulator()
    # sim.state()
    b = 0
    a = 1

    def callback():
        nonlocal b
        b = a
    
    a = 2
    callback()
    print(b)