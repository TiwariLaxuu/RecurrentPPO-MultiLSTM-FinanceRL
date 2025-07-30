import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class TradingEnvPPO(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.max_steps = len(df) - 1
        self.current_step = 0
        self.position = 0  
        self.initial_balance = 10000
        self.balance = self.initial_balance

        # Observation: past 10 candles (OHLCV) = (10, 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5), dtype=np.float32)

        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        self.current_step = 10
        self.position = 0
        self.balance = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        print(obs.shape)
        return obs 
        # return self.df.iloc[self.current_step-10:self.current_step].values.astype(np.float32)

    def step(self, action):
        done = False
        reward = 0
        price = self.df.iloc[self.current_step]['Close']

        # reward based on position and price movement
        if action == 1:  # buy
            if self.position == 0:
                self.entry_price = price
                self.position = 1
        elif action == 2:  # sell
            if self.position == 1:
                reward = price - self.entry_price
                self.balance += reward
                self.position = 0

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}")

class TradingEnvRecurrentPPO(gym.Env):
    def __init__(self, df):
        super(TradingEnvRecurrentPPO, self).__init__()
        self.df = df
        self.max_steps = len(df) - 1
        self.current_step = 0
        self.position = 0  
        self.initial_balance = 10000
        self.balance = self.initial_balance

        # Observation: past 10 candles (OHLCV) = (10, 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5), dtype=np.float32)

        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        self.current_step = 10
        self.position = 0
        self.balance = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        print(obs.shape)
        return obs 
        # return self.df.iloc[self.current_step-10:self.current_step].values.astype(np.float32)

    def step(self, action):
        done = False
        reward = 0
        price = self.df.iloc[self.current_step]['Close']

        # reward based on position and price movement
        if action == 1:  # buy
            if self.position == 0:
                self.entry_price = price
                self.position = 1
        elif action == 2:  # sell
            if self.position == 1:
                reward = price - self.entry_price
                self.balance += reward
                self.position = 0

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}")