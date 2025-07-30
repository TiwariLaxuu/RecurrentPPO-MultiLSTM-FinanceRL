import pandas as pd
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.trading_env import TradingEnv
from algorithm.ppo import train_agent, save_model, load_model, create_env

def main():
      # Adjust path accordingly
    # Load training data
    train_df = pd.read_csv('data/train_data.csv')[['Open', 'High', 'Low', 'Close', 'Volume']]
    test_df = pd.read_csv('data/test_data.csv')[['Open', 'High', 'Low', 'Close', 'Volume']]
    # Create training environment
    train_env = create_env(train_df, TradingEnv)
    # Train the agent
    model = train_agent(train_env, total_timesteps=10000)
    # Save the trained model
    save_model(model, "model/ppo_trading_agent")
    # Load the model if needed
    # model = load_model("model/ppo_trading_agent")
    # Create test environment
    test_env = TradingEnv(test_df)      
    obs, {} = test_env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
    print("✅ Test Completed. Total Reward:", total_reward)
if __name__ == "__main__":
    main()
# This script trains a PPO agent on a trading environment and tests it on a separate dataset.