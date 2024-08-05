import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random

################################################################################################################################################

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, final_layer=None):
        super(SimpleNeuralNetwork, self).__init__()
        self.net = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers):
            self.net.append(
              nn.Linear(hidden_dim, hidden_dim),
              nn.ReLU()
            )
        self.net.append(nn.Linear(hidden_dim, out_dim))
        if final_layer == "Softmax":
            self.net.append(nn.Softmax(dim=-1))
        elif final_layer == "Sigmoid":
            self.net.append(nn.Sigmoid())
        elif final_layer == "Tanh":
            self.net.append(nn.Tanh())
        self.net = nn.Sequential(*self.net)
        
    def forward(self, x):
        return self.net(x)
      
      
class Agent:
    def __init__(self):
        pass
      
    def learn_episode(self):
        pass
      
    def learn(self, n_episodes, *param):
        history = {"loss": [], "reward": []}
        for episode in range(n_episodes):
            loss, reward = self.learn_episode(*param)
            history["loss"].append(loss)
            history["reward"].append(reward)
        return history
      
    def sample_trajectory(self, env, gamma=1, max_steps=10000):
        state = env.reset()
        states = [state]
        actions = []
        rewards = []
        dones = []
        done = False
        step = 0
        while not done and step < max_steps:
            with torch.no_grad():
                action = self.policy.select_action(state)
            next_state, reward, done, _ = self.env.step(action.item())
            reward = reward ** step
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            state = next_state
            step += 1
        states = torch.tensor(states).float().to(self.device)
        next_states = states[1:]
        states = states[:-1]
        actions = torch.tensor(actions).float().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)
        return states, actions, rewards, next_states, dones
      
class ReplayBuffer:
    def __init__(self, env, process, policy, maxlen=10000, min_action=None, max_action=None, device="cpu"):
        self.env = env
        self.buffer = deque(maxlen=maxlen)
        self.process = process
        self.process.reset_states()
        self.maxlen = maxlen
        self.fill(policy, min_action, max_action)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def fill(self, policy, min_action, max_action):
        while len(self.buffer) < self.maxlen:
            state = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action = policy.select_action(state) + self.process.sample()
                if min_action or max_action:
                    action = action.clip(min_action, max_action)
                next_state, reward, done, _ = self.env.step(action.cpu().detach().numpy())
                self.buffer.append((state, action, reward, next_state, done))
                state = next_state

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, n_samples):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, n_samples))
        state = torch.tensor(state).float().to(self.device)
        action = torch.tensor(action).float().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        next_state = torch.tensor(next_state).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)
        return state, action, reward, next_state, done