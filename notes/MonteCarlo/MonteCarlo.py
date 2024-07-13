import numpy as np
import matplotlib.pyplot as plt

class MonteCarloValue:
    """
    Monte Carlo method for estimating the value functions
    """
    def __init__(self, states, sample):
        """ 
        states: a list of possible states
        sample: a function when called sample() returns a trajectory of the form
        {(state1, action1): reward1, (state2, action2): reward2,...}
        """
        self.states = states
        self.sample = sample
        
    def estimate(self, max_iter, gamma=1, every_visit=False):
        """ 
        Estimates the value function
        max_iter: number of trajectories used
        gamma: discount factor
        every_visit: Uses every visit Monte Carlo if True, else uses first visit Monte Carlo
        Returns the q-values of the form
        {state1: {action1: value1, action2: value2,...},
         state2: {action1: value1, action2: value2,...},
         ....}
        """
        value = {state: 0 for state in self.states}
        count = {state: 0 for state in self.states}
        for i in range(max_iter):
            episode = self.sample()
            states = list([state[0] for state in episode.keys()])
            cum_reward = 0
            for t, (state_action, reward) in enumerate(list(episode.items())[::-1]):
                state, _ = state_action
                t = len(episode) - t - 1
                cum_reward += gamma * reward
                if every_visit or state not in states[:t]:
                    count[state] += 1
                    value[state] = value[state] + (cum_reward - value[state]) / count[state]
        return value

class MonteCarloQValue:
    """
    Monte Carlo method for estimating the Q value functions.
    The implementation is similar to the case of value function, except for small twists in the estimate function
    """
    def __init__(self, states, actions, sample):
        """ 
        states: a list of possible states
        actions: a list of possible actions
        sample: a function when called sample() returns a trajectory of the form
        {(state1, action1): reward1, (state2, action2): reward2,...}
        """
        self.states = states
        self.actions = actions
        self.sample = sample
        
    def estimate(self, max_iter, gamma=1, every_visit=False):
        """ 
        Estimates the Q value function
        max_iter: number of trajectories used
        gamma: discount factor
        every_visit: Uses every visit Monte Carlo if True, else uses first visit Monte Carlo
        """
        value = {state: {action: 0 for action in self.actions} for state in self.states}
        count = {state: {action: 0 for action in self.actions} for state in self.states}
        for i in range(max_iter):
            episode = self.sample()
            cum_reward = 0
            for t, (state_action, reward) in enumerate(list(episode.items())[::-1]):
                state, action = state_action
                t = len(episode) - t - 1
                cum_reward += gamma * reward
                if every_visit or state not in self.states[:t]:
                    count[state][action] += 1
                    value[state][action] = value[state][action] + (cum_reward - value[state][action]) / count[state][action]            
        return value

def plot_values(values, usable_ace=True):
    """
    Takes in values of the form
    {state1: values1, state2: values2, ...}
    and plot the value function. 
    """
    values = {state[:-1]: value for state, value in values.items() if state[-1] is usable_ace and state[0] <= 21}
    x_values = np.array([k[0] for k in values.keys()])
    y_values = np.array([k[1] for k in values.keys()])
    max_x = np.max(x_values)
    max_y = np.max(y_values)
    X, Y = np.meshgrid(np.arange(0, max_y + 1), np.arange(0, max_x + 1))  
    Z = np.zeros(X.shape)

    for key, value in values.items():
        x, y = key
        Z[x, y] = value  

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.9)
    ax.set_xlabel('Dealer hand')
    ax.set_ylabel('Player hand')
    ax.set_zlabel('Reward', labelpad=2) 
    ax.set_xlim(0, 10)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=17, azim=-65)
    plt.show()