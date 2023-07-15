import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
from scipy.special import softmax

import gym
from gym.utils import seeding
from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data, window_size):
        super(PortfolioEnv, self).__init__()
        self.data = data
        self.window_size = window_size

        self.day = window_size
        self.current_step = window_size
        # Define the action and observation space
        self.action_space = spaces.Box(
            low=-10, high=5, shape=(self.data.shape[1],), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(
            self.window_size, self.data.shape[1], self.data.shape[2]+1), dtype=np.float32)

        self.portfolio_value = 1e6
        self.balance = 1e6

        self.weights = np.zeros(self.data.shape[1])
        self.weights.fill(1/self.data.shape[1])
        self.memory = []
        self.weights_memory = []
        # Initialize the state
        self.reset()

    def step(self, action):
        # Execute one step of the environment
        action = softmax(action)  # Clip action values to [0, 1]
        self.weights = np.array(action).reshape(-1)
        self.weights_memory.append(self.weights)

        self.portfolio_value = self.balance
        self.memory.append(self.portfolio_value)

        # Update the state
        self.observation = self.get_observation()
        reward = 0
        for i in range(self.data.shape[1]):
            reward += ((self.data[self.day+1, i, 0]-self.data[self.day, i, 0]) /
                       self.data[self.day, i, 0])*self.weights[i]*self.balance

        self.balance += reward
        # Move to the next time step
        self.current_step += 1
        self.day += 1
        # Calculate the reward and done flag
        done = self.day >= self.data.shape[0] - 1
        print(self.observation[-1, :, 0])
        return self.observation, (reward/1e6), done, {}

    def reset(self):
        # Reset the environment to the initial state
        self.current_step = self.window_size
        self.day = self.window_size
        self.portfolio_value = 1e6
        self.balance = 1e6
        self.weights = np.zeros(self.data.shape[1])
        self.weights.fill(1/self.data.shape[1])
        self.memory = []
        self.weights_memory = []
        # Initialize the state
        self.observation = self.get_observation()

        return self.observation

    def get_observation(self):
        state = np.zeros(self.observation_space.shape)
        state[:, :, :-
              1] = np.array(self.data[self.day-self.window_size:self.day])
        state[:, :, -1] = np.repeat(self.weights.reshape(1, -1),
                                    self.observation_space.shape[0], axis=0)
        return state
