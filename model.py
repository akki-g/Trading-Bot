"""
RL Implementation specifically for Forex Trading
Optimized for 24/7 markets and currency pair characteristics
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

# Import base RL components
from rl_env import RLTradingEnvironment, A2CAgent, RLTradingPipeline
from bot import calculate_indicators, generate_signals

class ForexRLEnvironment(RLTradingEnvironment):
    """Specialized RL environment for Forex trading"""
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        # Forex-specific default configuration
        forex_config = {
            'initial_capital': 10000,
            'max_position': 1.0,
            'transaction_cost': 0.00002,  # 2 pips typical spread
            'stop_loss': 0.001,  # 0.1% (10 pips for majors)
            'take_profit': 0.002,  # 0.2% (20 pips for majors)
            'leverage': 50,  # Typical forex leverage
            'lookback_window': 30,
            'reward_scaling': 1000,  # Higher scaling for smaller forex moves
            'use_position_sizing': True,
            'min_holding_period': 5,  # Minimum 5 periods to avoid overtrading
        }
        if config:
            forex_config.update(config)
        
        super().__init__(data, forex_config)
        
        # Forex-specific features
        self.session_times = self._identify_trading_sessions()
        self.volatility_regime = self._calculate_volatility_regime()
        
    def _identify_trading_sessions(self):
        """Identify major forex trading sessions"""
        sessions = pd.DataFrame(index=self.data.index)
        
        # Assuming data has datetime index
        hour = self.data.index.hour
        
        # Define sessions (in UTC)
        sessions['Sydney'] = ((hour >= 21) | (hour < 6)).astype(int)
        sessions['Tokyo'] = ((hour >= 0) & (hour < 9)).astype(int)
        sessions['London'] = ((hour >= 7) & (hour < 16)).astype(int)
        sessions['NewYork'] = ((hour >= 12) & (hour < 21)).astype(int)
        
        # Overlap sessions (high volatility)
        sessions['London_NY_Overlap'] = (
            sessions['London'] & sessions['NewYork']
        ).astype(int)
        
        return sessions
    
    def _calculate_volatility_regime(self):
        """Calculate current volatility regime"""
        returns = self.data['Close'].pct_change()
        volatility = returns.rolling(20).std()
        
        # Define regimes
        vol_percentiles = volatility.quantile([0.33, 0.67])
        regime = pd.Series(index=self.data.index, dtype=int)
        
        regime[volatility <= vol_percentiles[0.33]] = 0  # Low vol
        regime[(volatility > vol_percentiles[0.33]) & 
               (volatility <= vol_percentiles[0.67])] = 1  # Medium vol
        regime[volatility > vol_percentiles[0.67]] = 2  # High vol
        
        return regime
    
    def _calculate_features(self, step: int) -> np.ndarray:
        """Calculate forex-specific features"""
        # Get base features
        features = super()._calculate_features(step)
        
        # Add forex-specific features
        forex_features = []
        
        # Trading session features
        for session in ['Sydney', 'Tokyo', 'London', 'NewYork', 'London_NY_Overlap']:
            forex_features.append(self.session_times[session].iloc[step])
        
        # Volatility regime
        forex_features.append(self.volatility_regime.iloc[step] / 2)  # Normalize
        
        # Currency strength (simplified - using moving average divergence)
        ma_short = self.data['Close'].rolling(10).mean().iloc[step]
        ma_long = self.data['Close'].rolling(50).mean().iloc[step]
        currency_strength = (ma_short - ma_long) / ma_long
        forex_features.append(currency_strength)
        
        # Pip movement in last hour (assuming 5-min data)
        if step >= 12:
            pip_movement = (self.data['Close'].iloc[step] - 
                          self.data['Close'].iloc[step-12]) * 10000  # For majors
            forex_features.append(pip_movement / 100)  # Normalize
        else:
            forex_features.append(0)
        
        # Combine all features
        all_features = np.concatenate([features, forex_features])
        
        return all_features
    
    def _calculate_reward(self, prev_value, current_value, action):
        """Forex-specific reward calculation"""
        # Base reward
        reward = super()._calculate_reward(prev_value, current_value, action)
        
        # Additional forex-specific rewards
        current_price = self.data['Close'].iloc[self.current_step]
        
        # Reward for trading during high-volume sessions
        if action != 0:  # Any trade
            session_multiplier = 1.0
            if self.session_times['London_NY_Overlap'].iloc[self.current_step]:
                session_multiplier = 1.2  # 20% bonus for overlap trading
            elif (self.session_times['London'].iloc[self.current_step] or 
                  self.session_times['NewYork'].iloc[self.current_step]):
                session_multiplier = 1.1  # 10% bonus for major sessions
            
            reward *= session_multiplier
        
        # Penalty for trading in low volatility
        if self.volatility_regime.iloc[self.current_step] == 0 and action != 0:
            reward -= 0.005 * self.config['reward_scaling']
        
        # Reward for respecting minimum holding period
        if self.holding_period > 0 and self.holding_period < self.config['min_holding_period']:
            if action != 0:  # Trying to close position early
                reward -= 0.01 * self.config['reward_scaling']
        
        return reward


class ForexA2CAgent(A2CAgent):
    """A2C Agent optimized for Forex trading"""
    
    def __init__(self, state_size, action_size, config=None):
        # Forex-optimized configuration
        forex_config = {
            'learning_rate': 0.00005,  # Lower LR for stability
            'gamma': 0.995,  # Higher gamma for longer-term rewards
            'tau': 0.97,
            'entropy_coef': 0.02,  # Higher entropy for exploration
            'value_loss_coef': 0.5,
            'max_grad_norm': 0.5,
            'hidden_size': 512,  # Larger network for complex patterns
            'num_layers': 3
        }
        if config:
            forex_config.update(config)
        
        super().__init__(state_size, action_size, forex_config)
        
    def _build_actor_critic(self):
        """Build deeper network for forex patterns"""
        class ForexActorCritic(nn.Module):
            def __init__(self, state_size, action_size, hidden_size, num_layers):
                super().__init__()
                
                # Shared layers with residual connections
                layers = []
                input_size = state_size
                
                for i in range(num_layers):
                    layers.append(nn.Linear(input_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.1))
                    layers.append(nn.BatchNorm1d(hidden_size))
                    input_size = hidden_size
                
                self.shared = nn.Sequential(*layers)
                
                # Attention mechanism for important features
                self.attention = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 4, hidden_size),
                    nn.Sigmoid()
                )
                
                # Actor head with temperature
                self.actor = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, action_size)
                )
                self.temperature = nn.Parameter(torch.ones(1))
                
                # Critic head
                self.critic = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1)
                )
                
            def forward(self, state):
                # Ensure state has batch dimension
                if len(state.shape) == 1:
                    state = state.unsqueeze(0)
                
                features = self.shared(state)
                
                # Apply attention
                attention_weights = self.attention(features)
                features = features * attention_weights
                
                # Actor output with temperature scaling
                logits = self.actor(features) / self.temperature
                policy = torch.softmax(logits, dim=-1)
                
                # Critic output
                value = self.critic(features)
                
                return policy, value
        
        return ForexActorCritic(
            self.state_size, 
            self.action_size, 
            self.config['hidden_size'],
            self.config['num_layers']
        )


class ForexRLPipeline(RLTradingPipeline):
    """Complete pipeline for Forex RL trading"""
    
    def __init__(self, currency_pair: str):
        super().__init__(currency_pair, agent_type='ForexA2C')
        self.currency_pair = currency_pair
        self.pip_value = self._calculate_pip_value()
        
    def _calculate_pip_value(self):
        """Calculate pip value for the currency pair"""
        # Simplified pip values for majors
        pip_values = {
            'EURUSD=X': 0.0001,
            'GBPUSD=X': 0.0001,
            'USDJPY=X': 0.01,
            'USDCHF=X': 0.0001,
            'AUDUSD=X': 0.0001,
            'USDCAD=X': 0.0001,
            'NZDUSD=X': 0.0001
        }
        return pip_values.get(self.currency_pair, 0.0001)
    
    def prepare_forex_data(self, start_date, end_date=None):
        """Prepare forex data with additional features"""
        # Get base data
        data = self.prepare_data(start_date, end_date)
        
        # Add forex-specific indicators
        
        # 1. Multiple timeframe analysis
        data['SMA_5'] = data['Close'].rolling(5).mean()
        data['SMA_15'] = data['Close'].rolling(15).mean()
        data['SMA_60'] = data['Close'].rolling(60).mean()
        
        # 2. Currency strength index (simplified)
        data['Strength'] = (data['Close'] - data['Close'].rolling(20).mean()) / data['Close'].rolling(20).std()
        
        # 3. Average True Range for volatility
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        data['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        
        # 4. Spread estimation (simplified)
        data['Spread'] = (data['High'] - data['Low']) / data['Close'] * 10000  # In pips
        
        return data
    
    def train_forex(self, episodes=200, use_multi_timeframe=True):
        """Train with forex-specific features"""
        # Prepare training data
        train_data = self.prepare_forex_data("2020-01-01", "2023-01-01")
        
        # Create forex environment
        env_config = {
            'initial_capital': 10000,
            'max_position': 1.0,
            'transaction_cost': 0.00002,  # 2 pips
            'stop_loss': 0.001,  # 10 pips
            'take_profit': 0.002,  # 20 pips
            'leverage': 50,
            'min_holding_period': 5
        }
        
        self.env = ForexRLEnvironment(train_data, env_config)
        
        # Create forex-optimized agent
        self.agent = ForexA2CAgent(
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n
        )
        
        # Training with early stopping
        best_sharpe = -np.inf
        patience = 20
        patience_counter = 0
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_data = []
            
            while True:
                action, _, _ = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_data.append((state, action, reward, next_state, done))
                state = next_state
                
                if done:
                    # Update agent
                    if len(episode_data) > 0:
                        states = [e[0] for e in episode_data]
                        actions = [e[1] for e in episode_data]
                        rewards = [e[2] for e in episode_data]
                        next_states = [e[3] for e in episode_data]
                        dones = [e[4] for e in episode_data]
                        
                        self.agent.update(states, actions, rewards, next_states, dones)
                    
                    # Calculate metrics
                    portfolio_return = (info['portfolio_value'] - env_config['initial_capital']) / env_config['initial_capital'] * 100
                    
                    if len(self.env.equity_curve) > 1:
                        returns = pd.Series(self.env.equity_curve).pct_change().dropna()
                        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    else:
                        sharpe = 0
                    
                    # Early stopping check
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        patience_counter = 0
                        # Save best model
                        self.save_model(f'best_{self.currency_pair}_model.pth')
                    else:
                        patience_counter += 1
                    
                    # Store history
                    self.training_history['episode_rewards'].append(episode_reward)
                    self.training_history['episode_returns'].append(portfolio_return)
                    self.training_history['episode_sharpe'].append(sharpe)
                    
                    # Print progress
                    if episode % 10 == 0:
                        avg_sharpe = np.mean(self.training_history['episode_sharpe'][-10:])
                        print(f"Episode {episode}: Avg Sharpe: {avg_sharpe:.3f}, "
                              f"Best Sharpe: {best_sharpe:.3f}")
                    
                    # Early stopping
                    if patience_counter >= patience:
                        print(f"Early stopping at episode {episode}")
                        # Load best model
                        self.load_model(f'best_{self.currency_pair}_model.pth')
                        break
                    
                    break
        
        return self.training_history
    
    def calculate_forex_metrics(self, results):
        """Calculate forex-specific performance metrics"""
        metrics = {
            'total_return_pct': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'max_drawdown_pct': results['max_drawdown']
        }
        
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            
            # Pip-based metrics
            avg_win_pips = trades_df[trades_df['pnl'] > 0]['pnl'].mean() / self.pip_value if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss_pips = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) / self.pip_value if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            
            metrics['avg_win_pips'] = avg_win_pips * 10000  # Convert to pips
            metrics['avg_loss_pips'] = avg_loss_pips * 10000
            metrics['profit_factor'] = (avg_win_pips / avg_loss_pips) if avg_loss_pips > 0 else np.inf
            
            # Holding period analysis
            metrics['avg_holding_period'] = trades_df['holding_period'].mean()
            metrics['max_holding_period'] = trades_df['holding_period'].max()
            
        return metrics


def run_forex_rl_backtest(currency_pairs=None, training_period='2Y', test_period='6M'):
    """Run RL backtest on multiple forex pairs"""
    
    if currency_pairs is None:
        currency_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 
            'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X'
        ]
    
    results = {}
    
    # Calculate dates
    end_date = datetime.now()
    test_start = end_date - pd.Timedelta(test_period)
    train_start = test_start - pd.Timedelta(training_period)
    
    print(f"Training period: {train_start.date()} to {test_start.date()}")
    print(f"Testing period: {test_start.date()} to {end_date.date()}")
    print("="*60)
    
    for pair in currency_pairs:
        print(f"\nProcessing {pair}...")
        
        try:
            # Create pipeline
            pipeline = ForexRLPipeline(pair)
            
            # Train
            print("Training agent...")
            pipeline.train_forex(episodes=100)
            
            # Evaluate
            print("Evaluating performance...")
            eval_results = pipeline.evaluate(
                start_date=test_start.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Calculate forex metrics
            forex_metrics = pipeline.calculate_forex_metrics(eval_results)
            
            results[pair] = forex_metrics
            
            # Print summary
            print(f"\n{pair} Results:")
            print(f"  Return: {forex_metrics['total_return_pct']:.2f}%")
            print(f"  Sharpe: {forex_metrics['sharpe_ratio']:.2f}")
            print(f"  Win Rate: {forex_metrics['win_rate']:.1f}%")
            print(f"  Avg Win: {forex_metrics['avg_win_pips']:.1f} pips")
            print(f"  Avg Loss: {forex_metrics['avg_loss_pips']:.1f} pips")
            print(f"  Profit Factor: {forex_metrics['profit_factor']:.2f}")
            
        except Exception as e:
            print(f"Error processing {pair}: {str(e)}")
            continue
    
    # Create summary report
    summary_df = pd.DataFrame(results).T
    summary_df = summary_df.sort_values('sharpe_ratio', ascending=False)
    
    print("\n" + "="*80)
    print("FOREX RL BACKTEST SUMMARY")
    print("="*80)
    print(summary_df.to_string())
    
    # Save results
    summary_df.to_csv('forex_rl_backtest_results.csv')
    
    # Plot comparison
    plot_forex_results(summary_df)
    
    return results, summary_df


def plot_forex_results(summary_df):
    """Plot forex backtest results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Returns comparison
    ax1 = axes[0, 0]
    summary_df['total_return_pct'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Total Returns by Currency Pair')
    ax1.set_ylabel('Return (%)')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Sharpe ratios
    ax2 = axes[0, 1]
    summary_df['sharpe_ratio'].plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('Sharpe Ratios by Currency Pair')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Sharpe = 1')
    ax2.legend()
    
    # 3. Win rates and profit factors
    ax3 = axes[1, 0]
    x = range(len(summary_df))
    width = 0.35
    ax3.bar([i - width/2 for i in x], summary_df['win_rate'], width, label='Win Rate (%)', color='gold')
    ax3.bar([i + width/2 for i in x], summary_df['profit_factor'] * 10, width, label='Profit Factor (x10)', color='orange')
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary_df.index)
    ax3.set_title('Win Rate and Profit Factor')
    ax3.legend()
    
    # 4. Risk metrics
    ax4 = axes[1, 1]
    ax4.scatter(summary_df['max_drawdown_pct'].abs(), summary_df['total_return_pct'], s=100)
    for i, pair in enumerate(summary_df.index):
        ax4.annotate(pair.replace('=X', ''), 
                    (summary_df['max_drawdown_pct'].abs().iloc[i], 
                     summary_df['total_return_pct'].iloc[i]))
    ax4.set_xlabel('Max Drawdown (%)')
    ax4.set_ylabel('Total Return (%)')
    ax4.set_title('Risk-Return Profile')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('forex_rl_backtest_summary.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("FOREX RL TRADING SYSTEM")
    print("="*60)
    
    # 1. Run backtest on major pairs
    results, summary = run_forex_rl_backtest(
        currency_pairs=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'],
        training_period='1Y',
        test_period='3M'
    )
    
    # 2. Train and deploy best performing pair
    if len(summary) > 0:
        best_pair = summary.index[0]  # Highest Sharpe ratio
        print(f"\nDeploying RL strategy for best pair: {best_pair}")
        
        pipeline = ForexRLPipeline(best_pair)
        pipeline.train_forex(episodes=200)
        
        # Save for production
        pipeline.save_model(f'production_{best_pair}_model.pth')
        print(f"Model saved for production trading: {best_pair}")