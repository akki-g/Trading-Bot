# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a comprehensive algorithmic trading system with three main trading strategies:

### Core Components

1. **Traditional Technical Analysis Strategy** (`bot.py`)
   - Multi-indicator strategy using RSI, MACD, Bollinger Bands, Stochastic, and EMA200
   - Signal generation based on confluence of indicators
   - Position management with take profit (0.2%) and stop loss (0.1%)
   - Main entry point: `run_trading_algorithm(symbol, start_date, end_date)`

2. **Reinforcement Learning Framework** (`rl_env.py`)
   - Three RL agents: A2C (Actor-Critic), DQN (Deep Q-Network), PPO (Proximal Policy Optimization)
   - Custom trading environment (`RLTradingEnvironment`) with advanced state features
   - Complete pipeline from training to evaluation (`RLTradingPipeline`)

3. **Forex-Specific RL Implementation** (`model.py`)
   - Specialized for forex trading with session awareness (Sydney, Tokyo, London, NY)
   - Volatility regime detection and pip-based metrics
   - Leverage-aware position sizing and spread handling
   - Multi-timeframe analysis and currency strength indicators

4. **Data Management** (`data.py`)
   - Simple data fetching using yfinance
   - Currently focused on LUMN ticker

## Dependencies

The system requires these key Python packages (from `requirements.txt`):
- yfinance (market data)
- pandas, numpy (data manipulation)
- scikit-learn (preprocessing)
- torch (deep learning for RL)
- matplotlib (visualization)
- backtrader (alternative backtesting framework, not currently used)
- gym (RL environment)

## Common Development Commands

### Running the Traditional Strategy
```python
python bot.py
# Tests NVDA from 2022-01-01 with full performance analysis and plots
```

### Running RL Training and Evaluation
```python
python rl_env.py
# Trains A2C agent on NVDA with 100 episodes, then evaluates on test data
```

### Running Forex RL System
```python
python model.py
# Runs multi-currency forex backtest with specialized RL agents
```

### Data Fetching
```python
python data.py
# Downloads LUMN historical data for analysis
```

## Key Configuration Parameters

### Traditional Strategy
- Take profit: 0.2% (0.002)
- Stop loss: 0.1% (0.001)
- Transaction cost: 0.1% (0.001)

### RL Environment
- Initial capital: $10,000
- Lookback window: 30 days
- Action space: Discrete (Hold=0, Buy=1, Sell=2)
- State features: 23+ technical and position-based features

### Forex RL
- Leverage: 50x
- Transaction cost: 0.002% (2 pips)
- Minimum holding period: 5 periods
- Take profit: 0.2% (20 pips), Stop loss: 0.1% (10 pips)

## Virtual Environment

The project includes a Python virtual environment in `tradenv/` which contains all dependencies. Activate with:
```bash
source tradenv/bin/activate  # On macOS/Linux
```

## Model Persistence

- RL models are saved using PyTorch's state dict format
- Forex models use production naming: `production_{pair}_model.pth`
- Standard models: `{symbol}_rl_results.csv` for performance data

## Performance Visualization

All strategies generate comprehensive plots:
- Price charts with entry/exit signals
- Technical indicators (RSI, MACD)
- Cumulative P&L curves
- Trade distribution histograms
- Risk-return scatter plots (forex)

## Trading Signal Logic

The traditional strategy requires ALL conditions to be met simultaneously:
- **Buy**: Above EMA200 + RSI>50 + Near BB lower band + MACD bullish cross + Stochastic bullish cross
- **Sell**: Above EMA200 + RSI<50 + Near BB upper band + MACD bearish cross + Stochastic bearish cross

The RL strategies learn optimal entry/exit points through trial and reward optimization, potentially discovering patterns beyond traditional technical analysis.