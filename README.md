# RecurrentPPO-MultiLSTM-FinanceRL

**Enhancing Recurrent PPO with Multi-Input LSTM Feature Extraction and Time-Series Forecasting in Financial Markets**

This repository contains the implementation of a deep reinforcement learning (DRL) framework for financial trading, leveraging **Recurrent Proximal Policy Optimization (Recurrent PPO)** enhanced with a **multi-stream LSTM** architecture. The model is designed to process complex financial time-series data from different perspectives: price movements, technical indicators, and forecasted features.

---

## Overview

In this work, we propose a multi-input architecture combining **three LSTM modules**:

1. **Agent Stream LSTM** – RecurrentPPO itself has LSTM Architecture to handle temporal information of history of agent like after buying, we have to sell(such info is in hidden layer)
2. **Indicator Stream LSTM** – Processes raw historical price, volume and technical indicators data (e.g., RSI, MACD, EMA).
3. **Forecast Stream LSTM** – Learns patterns from short-term time-series forecasts(Supervised Training).

These feature streams are fused and passed to a recurrent policy network trained using **Recurrent PPO**, which enables sequential decision-making with memory, making it suitable for financial time-series.

---

## Key Features

- ✅ Multi-LSTM architecture for richer feature extraction  
- ✅ Recurrent PPO agent using Stable-Baselines3 with custom policy  
- ✅ Support for discrete or continuous action spaces (buy, sell, hold)  
- ✅ Modular code: plug in new features, signals, or market datasets  
- ✅ Designed for real-world financial forecasting and trading tasks

---




