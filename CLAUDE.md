# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a comprehensive algorithmic trading system with four main components: traditional technical analysis, reinforcement learning frameworks, forex-specific implementations, and a production-ready data management system.

### Core Components

1. **Traditional Technical Analysis Strategy** (`bot.py`)
   - Multi-indicator confluence strategy using RSI (6,14,24), MACD (8,21,5), Bollinger Bands (20), Stochastic (9,3,3), and EMA200
   - Signal generation requires ALL conditions met simultaneously for trade execution
   - Position management with configurable take profit (0.2%) and stop loss (0.1%)
   - Complete backtest pipeline with performance metrics and visualization
   - Main entry point: `run_trading_algorithm(symbol, start_date, end_date)`

2. **Reinforcement Learning Framework** (`rl_env.py`)
   - Three RL agents: A2C (Actor-Critic), DQN (Deep Q-Network), PPO (Proximal Policy Optimization)
   - Custom trading environment (`RLTradingEnvironment`) with 23+ advanced state features
   - Features include price ratios, technical indicators, volume analysis, market microstructure, and position information
   - Complete pipeline from training to evaluation (`RLTradingPipeline`)
   - Experience replay, target networks, GAE (Generalized Advantage Estimation)
   - Model saving/loading with comprehensive checkpointing

3. **Forex-Specific RL Implementation** (`model.py`)
   - Specialized `ForexRLEnvironment` with session awareness (Sydney, Tokyo, London, NY trading sessions)
   - Volatility regime detection (low/medium/high) and pip-based performance metrics
   - Leverage-aware position sizing (50x) and spread handling (2 pips)
   - Enhanced `ForexA2CAgent` with attention mechanism and deeper networks
   - Multi-timeframe analysis and currency strength indicators
   - Production-ready training with early stopping and best model selection
   - Comprehensive forex backtesting across major pairs with risk-return analysis

4. **Multi-Pair Forex Trading System** (`forex.py`)
   - Bulk data downloading with retry logic and rate limiting for multiple currency pairs
   - `MultiPairForexBot` class for simultaneous trading across multiple pairs
   - Portfolio-level position management with configurable position sizing
   - Real-time signal processing and execution across all major forex pairs
   - Parameter optimization framework for strategy tuning
   - Complete paper trading simulation with performance tracking

5. **Hybrid Data Management System** (`forex_db/`)
   - TimescaleDB-based high-performance time-series database
   - **CSV-first approach**: Bulk historical data from pre-downloaded CSV files (5-year daily + 1-year 1-minute data)
   - **Incremental updates**: yfinance API for new data since last database update
   - Mixed resolution support: 1-minute and daily data in unified schema
   - Database schema optimized for forex data with hypertables and compression
   - Automated gap detection and data validation

6. **Data Visualization Dashboard** (`visualization/`)
   - **React Frontend**: Interactive web-based dashboard for data exploration
   - **FastAPI Backend**: High-performance API server with filtering and pagination
   - **Multi-view Interface**: Table view for detailed data and chart view for visual analysis
   - **Advanced Filtering**: Filter by currency pair, time interval, date range, and record limits
   - **Real-time Statistics**: Summary statistics and data range information
   - **Interactive Charts**: OHLC line charts, candlestick visualizations, and volume analysis
   - **Responsive Design**: Mobile-friendly interface with modern UI components

## Dependencies

The system requires these key Python packages:

### Core Trading Libraries
- `yfinance` - Market data fetching
- `pandas`, `numpy` - Data manipulation and analysis
- `scikit-learn` - Data preprocessing and scaling
- `matplotlib` - Visualization and plotting

### Machine Learning & RL
- `torch` - Deep learning framework for RL agents
- `gymnasium` (gym) - RL environment framework
- `backtrader` - Alternative backtesting framework

### Database & API
- `psycopg2-binary` - PostgreSQL adapter
- `timescaledb` - Time-series database extension
- `sqlalchemy` - Database ORM
- `fastapi`, `uvicorn` - API server framework
- `aiohttp` - Async HTTP client

### Data Sources & Utilities
- `alpha-vantage` - Financial data API
- `requests` - HTTP requests
- `python-dotenv` - Environment variable management
- `schedule` - Task scheduling
- `retry` - Retry logic for API calls

## Common Development Commands

### Virtual Environment
```bash
# Activate the trading environment
source tradenv/bin/activate  # On macOS/Linux
# or
tradenv\Scripts\activate     # On Windows
```

### Traditional Strategy
```python
# Run full backtest on NVDA with visualization
python bot.py

# Test specific symbol and date range
from bot import run_trading_algorithm
data, metrics = run_trading_algorithm('AAPL', '2023-01-01', '2024-01-01')
```

### Reinforcement Learning
```python
# Train and evaluate A2C agent on NVDA
python rl_env.py

# Train specific agent type
from rl_env import RLTradingPipeline
pipeline = RLTradingPipeline('NVDA', agent_type='PPO')
history = pipeline.train(episodes=200)
results = pipeline.evaluate()
```

### Forex RL System
```python
# Run comprehensive forex RL backtest
python model.py

# Train single forex pair
from model import ForexRLPipeline
pipeline = ForexRLPipeline('EURUSD=X')
pipeline.train_forex(episodes=100)
```

### Multi-Pair Forex Trading
```python
# Run multi-pair forex analysis
python forex.py

# Backtest specific pairs
from forex import backtest_forex_pairs
results, summary = backtest_forex_pairs(['EURUSD=X', 'GBPUSD=X'])
```

### Data Visualization Dashboard
```bash
# Test system components
cd visualization
python test_system.py

# Start backend API server (Terminal 1)
cd visualization
python start_backend.py

# Start React frontend (Terminal 2)
cd visualization
./start_frontend.sh

# Access dashboard
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### CSV-Based Database System (New Approach)
```bash
# Initial setup - import your CSV files into database
cd forex_db
python setup_csv_system.py

# Check current data status
python incremental_updater.py --status

# Update all pairs with latest data
python incremental_updater.py --update-all

# Update specific pair only
python incremental_updater.py --pair EURUSD

# Run scheduled updates (continuous)
python incremental_updater.py --schedule

# Manual CSV import (if needed)
python csv_data_manager.py --data-dir ..
```

### Legacy Database System (Original)
```bash
# Full system with historical data collection
python forex_db/forex_system.py

# Skip historical data collection  
python forex_db/forex_system.py --no-historical

# Test database connectivity
python forex_db/test_connection.py
```

## Key Configuration Parameters

### Traditional Strategy (bot.py)
- Take profit: 0.2% (0.002)
- Stop loss: 0.1% (0.001)
- Transaction cost: 0.1% (0.001)
- Technical indicators: RSI(6,14,24), MACD(8,21,5), BB(20), Stochastic(9,3,3), EMA(200)

### RL Environment (rl_env.py)
- Initial capital: $10,000
- Lookback window: 30 days
- Action space: Discrete (Hold=0, Buy=1, Sell=2)
- State features: 23+ (returns, technical indicators, volume, position info)
- Reward scaling: 100x
- Experience replay buffer: 10,000 experiences (DQN)

### Forex RL (model.py)
- Leverage: 50x
- Transaction cost: 0.002% (2 pips)
- Take profit: 0.2% (20 pips), Stop loss: 0.1% (10 pips)
- Minimum holding period: 5 periods
- Hidden layers: 512 neurons with 3 layers
- Learning rate: 0.00005 (reduced for stability)

### Multi-Pair Trading (forex.py)
- Major pairs: EURUSD, USDJPY, GBPUSD, USDCHF, AUDUSD, USDCAD, NZDUSD
- Position sizing: 10-20% of available capital per position
- Bulk download with exponential backoff retry logic

### CSV-Based Database System (forex_db/)
- **Historical data**: Pre-loaded from CSV files (5-year daily + 1-year 1-minute resolution)
- **Incremental updates**: yfinance API for new data since last update
- **Update frequency**: 5 minutes for 1-minute data, daily for daily data
- **Batch size**: 10,000 records for CSV import, 1,000 for incremental updates
- **Data validation**: OHLC validation, gap detection, duplicate prevention
- **Schema**: Mixed resolution support (1d, 1m intervals) with TimescaleDB optimization

## Trading Signal Logic

### Traditional Strategy Signal Requirements
All conditions must be met simultaneously:

**Buy Signal:**
- Above EMA200 (trend filter)
- RSI14 > 50 (momentum)
- Price within 5% of Bollinger Bands lower band (support)
- MACD bullish crossover (MACD > Signal Line)
- Stochastic bullish crossover (K > D)

**Sell Signal:**
- Above EMA200 (trend filter maintained)
- RSI14 < 50 (momentum reversal)
- Price within 5% of Bollinger Bands upper band (resistance)
- MACD bearish crossover (MACD < Signal Line)
- Stochastic bearish crossover (K < D)

### RL Strategy Learning
- A2C: Actor-critic with Generalized Advantage Estimation
- DQN: Deep Q-Network with experience replay and target networks
- PPO: Proximal Policy Optimization with clipped surrogate objective
- Forex RL: Session-aware with volatility regime detection

## Model Persistence & Results

### File Naming Conventions
- RL models: `{symbol}_rl_model.pth` (standard), `best_{pair}_model.pth` (forex)
- Production models: `production_{pair}_model.pth`
- Results: `{symbol}_trading_results.csv`, `forex_backtest_summary.csv`
- Plots: `{symbol}_trading_performance.png`, `forex_rl_backtest_summary.png`

### Model Components Saved
- PyTorch state dictionaries for networks
- Optimizer states for resuming training
- Training history and hyperparameters
- Environment configuration and feature scalers

## Performance Visualization

All strategies generate comprehensive analysis:

### Traditional Strategy
- Price charts with entry/exit signals and Bollinger Bands
- Technical indicator plots (RSI with overbought/oversold levels)
- Cumulative P&L curves with drawdown analysis
- Trade distribution histograms

### RL Strategies
- Portfolio value evolution over time
- Action distribution (buy/sell/hold signals)
- Trade P&L distribution analysis
- Training convergence plots (reward, loss, entropy)

### Forex Analysis
- Multi-pair performance comparison
- Risk-return scatter plots
- Sharpe ratio and drawdown analysis
- Currency pair correlation matrices

## Database Schema (forex_db/)

### Enhanced Schema for Mixed Resolution Data
```sql
CREATE TABLE forex_data (
    timestamp       TIMESTAMPTZ NOT NULL,
    pair           VARCHAR(10) NOT NULL,
    interval_type  VARCHAR(5) NOT NULL,  -- '1m', '1d', etc.
    open           DECIMAL(20,8) NOT NULL,
    high           DECIMAL(20,8) NOT NULL,
    low            DECIMAL(20,8) NOT NULL,
    close          DECIMAL(20,8) NOT NULL,
    volume         BIGINT DEFAULT 0,
    spread         DECIMAL(10,8),
    tick_volume    INTEGER,
    source         VARCHAR(50) DEFAULT 'CSV_IMPORT',
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_ohlc CHECK (
        high >= low AND high >= open AND high >= close AND 
        low <= open AND low <= close
    ),
    CONSTRAINT valid_interval CHECK (
        interval_type IN ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M')
    )
);

-- Unique constraint to prevent duplicates
CREATE UNIQUE INDEX idx_forex_data_unique 
    ON forex_data (timestamp, pair, interval_type);
```

### Performance Optimization
- TimescaleDB hypertables partitioned by timestamp
- Automatic compression for data older than 7 days
- Multi-column indexes for pair + interval + timestamp queries
- Materialized view (`forex_latest`) for latest data per pair/interval
- Connection pooling for concurrent access

## Testing & Validation

### CSV Data Import & Validation
```python
# Complete system setup from CSV files
cd forex_db
python setup_csv_system.py

# Manual CSV data validation
from forex_db.csv_data_manager import CSVDataManager
manager = CSVDataManager()
df = manager.load_csv_file('../fx_data_EURUSD.csv')
```

### Incremental Update Testing
```python
# Test incremental updates
from forex_db.incremental_updater import IncrementalUpdater
updater = IncrementalUpdater()
updater.print_data_status()  # Check data freshness
success = updater.update_pair_data('EURUSD')  # Test single pair update
```

### Strategy Testing
```python
# Unit tests for trading logic
python -m pytest forex_db/tests/

# Validate strategy signals with new data
from bot import calculate_indicators, generate_signals
```

### Database Performance Testing
```python
# Query performance with mixed resolution data
from forex_db.database_manager import DatabaseManager
db = DatabaseManager()

# Test query performance
import pandas as pd
query = """
SELECT * FROM forex_data 
WHERE pair = 'EURUSD' AND interval_type = '1m' 
AND timestamp >= NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC
"""
df = pd.read_sql(query, db.get_connection())
```

## CSV-First Data Workflow

### Current Data Structure
The system now uses a hybrid approach with pre-downloaded CSV files containing:
- **5-year daily data**: 2020-2025 with `interval=1d` for long-term analysis
- **1-year 1-minute data**: 2024-2025 with `interval=1m` for high-frequency trading
- **7 major forex pairs**: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD

### CSV File Format
```csv
datetime,symbol,interval,open,high,low,close,volume
2024-01-01 17:00:00,EURUSD,1m,1.10427,1.10429,1.10425,1.10429,0.0
2020-06-29 00:00:00,EURUSD,1d,1.1224,1.1289,1.1220,1.1225,0.0
```

### Recommended Workflow

#### 1. Initial Setup (One-time)
```bash
# Setup database and import CSV data
cd forex_db
python setup_csv_system.py
```

#### 2. Daily Operations
```bash
# Check data status
python incremental_updater.py --status

# Update with latest data
python incremental_updater.py --update-all
```

#### 3. Automated Updates
```bash
# Run continuous scheduled updates
python incremental_updater.py --schedule

# Or add to crontab for periodic updates:
# */5 * * * * cd /path/to/forex_db && python incremental_updater.py --update-all
```

#### 4. Data Access for Trading Systems
```python
# Access data for trading algorithms
from forex_db.database_manager import DatabaseManager
import pandas as pd

db = DatabaseManager()
query = """
SELECT timestamp, open, high, low, close 
FROM forex_data 
WHERE pair = 'EURUSD' AND interval_type = '1m'
AND timestamp >= NOW() - INTERVAL '1 week'
ORDER BY timestamp
"""
data = pd.read_sql(query, db.get_connection())
```

### Advantages of CSV-First Approach
- **Reliability**: No API rate limiting for historical data
- **Speed**: Bulk import much faster than API calls
- **Completeness**: Guaranteed historical coverage
- **Cost-effective**: No API costs for historical data
- **Incremental**: Only fetch new data as needed