import config
from engine_connector import EngineConnector

def main():
    connector = EngineConnector()
    assert connector.connect(config.SessionName)
    eng = connector.engine()

    # Initialize simulator.
    eng.sim_initialize(nargout=0)
    
    
    print(eng.workspace)

    return 0

if __name__ == '__main__':
    _ = main()