import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Policy:
    """
    A template for MDP policy
    Need to implement the policy method!
    """
    def __init__(self, states, actions, state_action_dict=None):
        """ 
        @param states: list
            A list of available states
        @param actions: list
            A list of available actions
        @param state_action_dict: dict (default: None)
            A dictionary with key=state, value=action. Mainly for deterministic policy.
            In the policy method, can simply call self.state_action_dict[state]
        """
        self.states = states
        self.actions = actions
        self.state_action_dict = state_action_dict
        
    def policy(self, state):
        """ 
        Need to implement by the user!
        @param state: an available state
        
        @return action: an available action
        """
        pass
        
    def sample(self, env, max_len=100):
        """ 
        Samples a trajectory
        @param env: Any
            The enviroment the agent is interacting
        @param max_len: int
            Maximum length of the episode
        
        @return trajectory: dict
            Takes in the form
            {(state1, action1): reward1,....}
        """
        trajectory = {}
        obs, _ = env.reset()
        for t in range(max_len):
            action = self.policy(obs)
            obs_, reward, done, info = env.step(action)[:4]
            trajectory[(obs, action)] = reward
            obs = obs_
            if done:
                break
        return trajectory



################################################################################################################################################
################################################################################################################################################
###    Functions for plotting                                                                                                          #########
################################################################################################################################################
################################################################################################################################################

def plotBlackJackOptimalStrategy(optimal_policy, states, epsilon=0.0):
    """ 
    Given a policy and a collection of states, plot the optimal policy for BlackJack
    """
    LABEL = {0: 'Stick', 1: 'Hit'}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    true_states = [state for state in states if state[-1] is True]
    xy = []
    action = []
    for state in true_states:
        xy.append(state[:2])
        action.append(optimal_policy.policy(state, epsilon))
    xy = np.array(xy)
    for label in np.unique(action):
        indices = action == label
        ax[0].scatter(xy[indices, 0], xy[indices, 1], label=f'{LABEL[label]}')
        ax[0].set_xlabel('Player hand')
        ax[0].set_ylabel('Dealer hand')
        ax[0].set_title('With usable ace')
        ax[0].legend()
    true_states = [state for state in states if state[-1] is False]
    xy = []
    action = []
    for state in true_states:
        xy.append(state[:2])
        action.append(optimal_policy.policy(state, epsilon))
    xy = np.array(xy)
    for label in np.unique(action):
        indices = action == label
        ax[1].scatter(xy[indices, 0], xy[indices, 1], label=f'{LABEL[label]}')
        ax[1].set_xlabel('Player hand')
        ax[1].set_ylabel('Dealer hand')
        ax[1].set_title('Without usable ace')
        ax[1].legend()
    plt.show()
    
def extract(q_values, usable_ace=True, action=0):
    """ 
    Returns the q_values with usable ace=True (default) and action=0 (default)
    """
    values = {}
    for state, actions in q_values.items():
        if state[-1] == usable_ace and state[0] >= 12 and state[0] <= 21:
            values[state[:2]] = actions[action]
    return values

def plot_values_helper(values, usable_ace):
    """
    Helper function for plot values, extract coordinates needed to plot the graphs
    """
    values = {state[:-1]: value for state, value in values.items() if state[-1] is usable_ace and 12 <= state[0] <= 21}
    x_values = np.array([k[0] for k in values.keys()])
    y_values = np.array([k[1] for k in values.keys()])
    max_x = np.max(x_values)
    max_y = np.max(y_values)
    min_x = np.min(x_values)
    min_y = np.min(y_values)
    X, Y = np.meshgrid(np.arange(min_y, max_y + 1), np.arange(min_x, max_x + 1))  
    Z = np.zeros(X.shape)
    for key, value in values.items():
        x, y = key
        Z[x-min_x, y-min_y] = value 
    return X, Y, Z

def plot_values(values):
    """
    Takes in values of the form
    {state1: values1, state2: values2, ...}
    and plot the value function in 3D. 
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={'projection': '3d'})
    for i, with_usable_ace in enumerate([True, False]):
        X, Y, Z = plot_values_helper(values, usable_ace=with_usable_ace)
        ax[i].plot_surface(X, Y, Z, alpha=0.9)
        ax[i].set_xlabel('Dealer hand')
        ax[i].set_ylabel('Player hand')
        ax[i].set_zlabel('Reward', labelpad=2) 
        ax[i].set_xlim(0, 10)
        ax[i].set_zlim(-1, 1)
        ax[i].view_init(elev=17, azim=-65)
    ax[0].set_title(f'With usable ace')
    ax[1].set_title(f'Without usable ace')
    plt.show()
    
def plot_q_values_heatmap_helper(q_values, usable_ace, action):
    """ 
    Helper function for extracting entries in q_values with usable_ace=False and action=0 (default)
    """
    values = {}
    for state, actions in q_values.items():
        if state[-1] == usable_ace and state[0] >= 12 and state[0] <= 21:
            values[state[:2]] = actions[action]
    keys = list(values.keys())
    x_coords = sorted(set(k[0] for k in keys))
    y_coords = sorted(set(k[1] for k in keys))
    heatmap = np.full((len(x_coords), len(y_coords)), np.nan)
    for (x, y), value in values.items():
        x_index = x_coords.index(x)
        y_index = y_coords.index(y)
        heatmap[x_index, y_index] = value
    return heatmap, x_coords, y_coords
    
def plot_q_values_heatmap(q_values):
    """ 
    Plot the q-values in Blackjack as heatmap
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    sns.set(font_scale=0.5)
    for i, usable_ace in enumerate([True, False]):
        for j, action in enumerate([0, 1]):
            heatmap, x_coords, y_coords = plot_q_values_heatmap_helper(q_values, usable_ace, action)
            sns.heatmap(heatmap, xticklabels=x_coords, yticklabels=y_coords, annot=True, cmap="coolwarm", center=0, ax=ax[i, j], vmin=-1, vmax=1)
            ax[i, j].set_xlabel('Player hand')
            ax[i, j].set_ylabel('Dealer hand')
    ax[0, 0].set_title('With usable ace, and stick')
    ax[1, 0].set_title('Without usable ace, and stick')
    ax[0, 1].set_title('With usable ace, and hit')
    ax[1, 1].set_title('Without usable ace, and hit')
    plt.show()