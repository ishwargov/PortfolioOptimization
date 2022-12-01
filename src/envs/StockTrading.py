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
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3

max_trade = 30
balance = 10000
transaction_fee = 0.001

class StockEnv(gym.Env):
    def __init__():
        