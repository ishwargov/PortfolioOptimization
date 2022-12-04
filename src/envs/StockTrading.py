import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import numpy as np
import random
import math
import time

import gym
from gym.utils import seeding
from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import objective_functions
from pypfopt.efficient_frontier import EfficientFrontier


max_trade = 50
balance = 10000
transaction_fee = 0.001


class StockEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, tickers):
        self.df = df
        self.tickers = tickers
        self.stock_dim = self.df.shape[1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(2*self.stock_dim+1,))
        self.data = self.df.iloc[0, :]
        self.state = [balance]+self.data.values.tolist()+[0]*self.stock_dim
        self.reward = 0
        self.memory = [balance]
        self.terminal = False
        self.day = 0
        self.sharpe = 0
        self.beta = 0
        self.act_memory = [[1/self.stock_dim]*self.stock_dim]

    def sell(self, index, action):
        action = np.floor(action)
        if self.state[index+self.stock_dim+1] > 0:
            self.state[0] += self.state[index+1] * \
                min(abs(action),
                    self.state[index+self.stock_dim+1]) * (1 - transaction_fee)
            self.state[index+self.stock_dim +
                       1] -= min(abs(action), self.state[index+self.stock_dim+1])

    def buy(self, index, action):
        action = np.floor(action)
        max_action = max(
            self.state[0]//(self.state[index+1] * (1 + transaction_fee)), action)
        self.state[0] -= self.state[index+1] * \
            max_action * (1 + transaction_fee)
        self.state[index+self.stock_dim+1] += max_action

    def step(self, actions):
        self.terminal = (self.day >= len(self.df.index.unique())-1)

        if self.terminal:
            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions*max_trade

            total_assets = self.state[0]
            total_assets += sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(
                self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            args = np.argsort(actions)

            sell_index = args[:np.where(actions < 0)[0].shape[0]]
            buy_index = args[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self.sell(index, actions[index])

            for index in buy_index:
                self.buy(index, actions[index])

            # force buy with balance (equal weight to stocks)
            if(self.state[0] > 0):
                cash_div = self.state[0]/self.stock_dim
                for i in range(self.stock_dim):
                    autoaction = cash_div//(self.state[i+1]
                                            * (1 + transaction_fee))
                    self.buy(i, autoaction)

            self.day += 1
            self.data = self.df.iloc[self.day, :]

            self.state = [self.state[0]]+self.data.values.tolist() + \
                list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])

            final_assets = self.state[0]
            final_assets += sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(
                self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))

            weights = self.normalize(
                np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))

            # if self.day >= 10:
            #     self.sharpe = self.get_sharpe(
            #         self.df.iloc[:self.day, :], weights)

            self.reward = self.beta*(self.sharpe) + \
                (1-self.beta)*(final_assets - total_assets)

            self.act_memory.append(weights.tolist())
        return self.state, self.reward, self.terminal, {}

    def normalize(self, actions):
        return actions/(np.sum(actions)+1e-10)

    def get_sharpe(self, data, weights):
        mu = mean_historical_return(data)
        S = CovarianceShrinkage(data).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        ef.set_weights({self.tickers[i]: weights[i]
                       for i in range(len(self.tickers))})
        return (ef.portfolio_performance()[2])

    def render(self, mode='human', close=False):
        return self.state

    def reset(self):
        self.day = 0
        self.data = self.df.iloc[0, :]
        self.terminal = False
        self.act_memory = [[1/self.stock_dim]*self.stock_dim]
        self.state = [balance]+self.data.values.tolist()+[0]*self.stock_dim
        self.reward = 0
        return self.state

    def get_action_memory(self):
        return self.act_memory
