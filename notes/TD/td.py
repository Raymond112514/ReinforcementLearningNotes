import gym
import numpy as np
import time
import imageio
import matplotlib.pyplot as plt

class Policy:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
    
    def get_action(self, state):
        pass

def display_values(values):
    """
    Takes in the value function, represented as a grid of shape [width, height]
    And displays the grid_world with values
    """
    height, width = 4, 12
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xticks(np.arange(width+1))
    ax.set_yticks(np.arange(height+1))
    ax.set_xticklabels([])
    ax.set_yticklabels([]) 
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    for y in range(height):
        for x in range(width):
            if width * y + x in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
                ax.add_patch(plt.Rectangle((x, 3-y), 1, 1, fill=True, color="black"))
            else:
                ax.add_patch(plt.Rectangle((x, 3-y), 1, 1, fill=True, color="white"))
                ax.text(x + 0.25, 3-y + 0.45, f'{values[width * y + x]:.3f}', fontsize=10, color='black')
    ax.set_title(f"Value function for random policy")
    plt.show()
    
def display_policy(policy, q_values):
    height, width = 4, 12
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xticks(np.arange(width+1))
    ax.set_yticks(np.arange(height+1))
    ax.set_xticklabels([])
    ax.set_yticklabels([]) 
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    triangle_size = 0.1
    for x in range(width):
        for y in range(height):
            if width * y +x in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
                ax.add_patch(plt.Rectangle((x, 3-y), 1, 1, fill=True, color="black"))
            else:
                ax.add_patch(plt.Rectangle((x, 3-y), 1, 1, fill=True, color="white"))
                optimal_action = policy.get_action(width * y + x, q_values, epsilon=0.0)
                if optimal_action == 0:
                    triangle = Polygon([(x + 0.5, 3-y + 1 - 0.1), 
                                        (x + 0.5 - triangle_size, 3-y + 1 - triangle_size - 0.1), 
                                        (x + 0.5 + triangle_size, 3-y + 1 - triangle_size - 0.1)], closed=True, color='black')
                elif optimal_action == 2:
                    triangle = Polygon([(x + 0.5, 3-y + 0.1), 
                                        (x + 0.5 - triangle_size, 3-y + 0.1 + triangle_size), 
                                        (x + 0.5 + triangle_size, 3-y + 0.1 + triangle_size)], closed=True, color='black')
                elif optimal_action == 3:
                    triangle = Polygon([(x + 0.1, 3-y + 0.5), 
                                        (x + 0.1 + triangle_size, 3-y + 0.5 - triangle_size), 
                                        (x + 0.1 + triangle_size, 3-y + 0.5 + triangle_size)], closed=True, color='black')
                elif optimal_action == 1:
                    triangle = Polygon([(x + 1 - 0.1, 3-y + 0.5), 
                                        (x + 1 - 0.1 - triangle_size, 3-y + 0.5 - triangle_size), 
                                        (x + 1 - 0.1 - triangle_size, 3-y + 0.5 + triangle_size)], closed=True, color='black')
                ax.add_patch(triangle)

    ax.set_title(f"Optimal policy for cliff walking")
    plt.show()