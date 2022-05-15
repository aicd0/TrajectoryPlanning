import numpy as np
import matplotlib.pyplot as plt
from framework.configuration import global_configs as configs

def main():
    x = np.arange(0.15, 6, 0.01)
    y = np.array([i - 1 / i for i in x])
    plt.figure(figsize=(3, 3))
    plt.plot(x, y)
    plt.title('')
    plt.xlabel('Distance')
    plt.ylabel('Potential')
    plt.tight_layout()
    plt.savefig('output/export/plot.png', dpi=300)
    plt.close()