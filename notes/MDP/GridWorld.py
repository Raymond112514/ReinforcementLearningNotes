import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

np.random.seed(0)

OFFSET = {(1, 0): (0.65, 0.45),
          (-1, 0): (0.05, 0.45),
          (0, 1): (0.375, 0.85),
          (0, -1): (0.375, 0.05)}

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# Implementation of GridWorld

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

class GridWorld:
    
    def __init__(self, height, width, exits, walls, rewards, initialize=None):
        """
        height, width: dimension of the grid world
        exits: dictionary with 'good_exit' and 'bad_exit' instance, stores the exits with different rewards
        walls: list of tuples representing the unavailable blocks
        rewards: dictionary with 'living_reward', 'win_reward', 'lose_reward' instance.
        initialize: default set as nearly-deterministic distribution, support uniform and random actions 
        """
        self.height = height
        self.width = width
        self.exits = exits
        self.walls = walls
        self.rewards = rewards
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.initialize = initialize
        self.states = None
        self.initialize_random_world()
        
    def initialize_random_world(self):
        """
        Initializes a world by assigning transition probabilities to each available grid
        Initializes transition_prob, which stores a dictionary with the following structure
        {state: {action1: {p11, p12, p13, ..., p1n}, 
                 action2: {p21, p22, p23, ..., p2n},
                 ...}}
        """
        self.transition_prob = dict()
        for x in range(self.width):
            for y in range(self.height):
                state = (x, y)
                if self.is_available(state):
                    prob = dict()
                    available_actions, successors = self.get_available_actions(state)
                    for i, action in enumerate(available_actions):
                        if self.initialize == "random":
                            prob[action] = np.random.randn(len(successors)) 
                        elif self.initialize == "uniform":
                            prob[action] = np.ones(len(successors)) 
                        else:
                            prob[action] = np.random.randn(len(successors)) 
                            prob[action][i] *= 10.0
                        prob[action] /= sum(prob[action])
                self.transition_prob[state] = prob
                    
    def is_available(self, state):
        """
        Checks whether a state tuple (x, y) is an available state
        """
        return 0 <= state[0] < self.width and 0 <= state[1] < self.height and state not in self.walls
    
    def get_available_actions(self, state):
        """
        Returns the list of available actions in state and its corresponding successors
        By default, at exit states, the only available action is do nothing (0, 0)
        """
        if not self.is_available(state):
            return [], []
        elif state in self.exits.values():
            return [(0, 0)], [state]
        available_actions = []
        successors = []
        for action in self.actions:
            successor = (state[0] + action[0], state[1] + action[1])
            if self.is_available(successor):
                available_actions.append(action)
                successors.append(successor)
        return available_actions, successors
                
    def get_transition_prob(self, state, action):
        """
        Returns the transition probability distribution p(s'|s, a)
        The returned distribution is stored as a dictionary with keys being the successor and value being the probability
        """
        available_actions, successors = self.get_available_actions(state)
        prob_dist = dict()
        for successor, prob in zip(successors, self.transition_prob[state][action]):
            prob_dist[successor] = prob
        return prob_dist
    
    def get_reward(self, state, successor):
        """
        Return the reward obtained by transitioning from state to successor
        """
        if state == self.exits['good_exit']:
            return self.rewards['win_reward']
        elif state == self.exits['bad_exit']:
            return self.rewards['lose_reward']
        else:
            return self.rewards['living_reward']
        
    def get_states(self):
        """ 
        Return the list of available states
        """
        if self.states is not None:
            return self.states
        self.states = []
        for x in range(self.width):
            for y in range(self.height):
                state = (x, y)
                if self.is_available(state):
                    self.states.append(state)
        return self.states
    
    def sample_trajectory(self, init_pos, policy):
        """
        Returns a trajectory based on init_pos following the polcy, 
        """
        
        
                
        
        
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# Utility functions for displaying GridWorld

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
        
def color(grid_world, state):
    """
    Returns the color of a state based on the type (exit, normal, wall, etc...)
    """
    if state in grid_world.walls:
        return [0, 0, 0]
    elif state == grid_world.exits['bad_exit']:
        return [181/255, 4/255, 4/255]
    elif state == grid_world.exits['good_exit']:
        return [0, 161/255, 5/255]
    return [255/255, 255/255, 255/255]

def display_values(grid_world, values):
    """
    Takes in the value function, represented as a grid of shape [width, height]
    And displays the grid_world with values
    """
    height, width = grid_world.height, grid_world.width
    fig, ax = plt.subplots(figsize=(1.5*width, 1.5*height))
    ax.set_xticks(np.arange(width+1))
    ax.set_yticks(np.arange(height+1))
    ax.set_xticklabels([])
    ax.set_yticklabels([]) 
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    for x in range(width):
        for y in range(height):
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, color=color(grid_world, (x, y))))
            if (x, y) not in grid_world.walls:
                ax.text(x + 0.325, y + 0.45, f'{values[x][y]:.3f}', fontsize=10, color='black')
            if (x, y) == grid_world.exits['good_exit'] or (x, y) == grid_world.exits['bad_exit']:
                ax.add_patch(plt.Rectangle((x+0.1, y+0.1), 0.8, 0.8, facecolor='none', edgecolor="white", linewidth=2))
                ax.text(x + 0.325, y + 0.45, f'{values[x][y]:.3f}', fontsize=10, color='white')

def display_qvalues(grid_world, q_values):
    """ 
    Takes in the q-function, represented as a dictionary with the following structure
    {state: {action1: q1, action2: q2,...}}
    And displays the grid world with q_values.
    """
    height, width = grid_world.height, grid_world.width
    fig, ax = plt.subplots(figsize=(1.5*width, 1.5*height))
    ax.set_xticks(np.arange(width+1))
    ax.set_yticks(np.arange(height+1))
    ax.set_xticklabels([])
    ax.set_yticklabels([]) 
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    for x in range(width):
        for y in range(height):
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, color=color(grid_world, (x, y))))
            q_value = q_values[(x, y)]
            if (x, y) != grid_world.exits['good_exit'] and (x, y) != grid_world.exits['bad_exit']:
                for action, q_val in q_value.items():
                    offset_x, offset_y = OFFSET[action]
                    ax.text(x + offset_x, y + offset_y, f'{q_val:.3f}', fontsize=8, color='black')
            else:
                ax.add_patch(plt.Rectangle((x+0.1, y+0.1), 0.8, 0.8, facecolor='none', edgecolor="white", linewidth=2))
                ax.text(x + 0.325, y + 0.45, f'{max(q_value.values()):.3f}', fontsize=10, color='white')
                
def display_policy(grid_world, policy):
    """ 
    Takes in a policy represented as a dictionary with key, value being state, action
    Displays the grid_world with the policy
    """
    height, width = grid_world.height, grid_world.width
    fig, ax = plt.subplots(figsize=(1.5*width, 1.5*height))
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
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, color=color(grid_world, (x, y))))
            if (x, y) in grid_world.walls:
                continue
            if (x, y) != grid_world.exits['good_exit'] and (x, y) != grid_world.exits['bad_exit']:
                optimal_action = policy[(x, y)]
                if optimal_action == (0, 1):
                    triangle = Polygon([(x + 0.5, y + 1 - 0.1), 
                                        (x + 0.5 - triangle_size, y + 1 - triangle_size - 0.1), 
                                        (x + 0.5 + triangle_size, y + 1 - triangle_size - 0.1)], closed=True, color='black')
                elif optimal_action == (0, -1):
                    triangle = Polygon([(x + 0.5, y + 0.1), 
                                        (x + 0.5 - triangle_size, y + 0.1 + triangle_size), 
                                        (x + 0.5 + triangle_size, y + 0.1 + triangle_size)], closed=True, color='black')
                elif optimal_action == (-1, 0):
                    triangle = Polygon([(x + 0.1, y + 0.5), 
                                        (x + 0.1 + triangle_size, y + 0.5 - triangle_size), 
                                        (x + 0.1 + triangle_size, y + 0.5 + triangle_size)], closed=True, color='black')
                elif optimal_action == (1, 0):
                    triangle = Polygon([(x + 1 - 0.1, y + 0.5), 
                                        (x + 1 - 0.1 - triangle_size, y + 0.5 - triangle_size), 
                                        (x + 1 - 0.1 - triangle_size, y + 0.5 + triangle_size)], closed=True, color='black')
                ax.add_patch(triangle)
            else:
                ax.add_patch(plt.Rectangle((x+0.1, y+0.1), 0.8, 0.8, facecolor='none', edgecolor="white", linewidth=2))
    plt.show()
