import numpy as np
import matplotlib.pyplot as plt
from framework.configuration import global_configs as configs

def main():
    data = np.load('output/ablation_study/sac_g99_k2_per/evaluators/agent_rl/plot_data/step_rewards.npz')
    x1 = data['x_step']
    y1 = data['y_avg_win']
    data = np.load('output/ablation_study/sac_g96_k2_per/evaluators/agent_rl/plot_data/step_rewards.npz')
    x2 = data['x_step']
    y2 = data['y_avg_win']
    plt.figure(figsize=(6, 4))
    plt.plot(x1, y1, label='99')
    plt.plot(x2, y2, label='96')
    plt.title('')
    plt.xlabel('Step')
    plt.ylabel('Step Reward')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig('output/export/plot.png', dpi=300)
    plt.close()