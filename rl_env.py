import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from typing import Dict, List, Tuple
import pickle

# Import your existing functions
from bot import calculate_indicators, generate_signals

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class RLTradingEnvironment(gym.Env):
    """Enhanced RL Trading Environment with advanced features"""
    
    def __init__(self, data: pd.DataFrame, config: Dict = None):
        super().__init__()
        
        # Default configuration
        self.config = {
            'initial_capital': 10000,
            'max_position': 1.0,
            'transaction_cost': 0.001,
            'stop_loss': 0.01,  # 1% stop loss
            'take_profit': 0.02,  # 2% take profit
            'lookback_window': 30,  # Days of history for state
            'reward_scaling': 100,
            'use_position_sizing': True
        }
        if config:
            self.config.update(config)
        
        self.data = data
        self.n_steps = len(data)
        
        # Episode variables
        self.reset()
        
        # Action space: Discrete - Hold(0), Buy(1), Sell(2)
        # Or Continuous - Position size from -1 to 1
        self.discrete_actions = True
        if self.discrete_actions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # State features
        self.state_features = self._define_state_features()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.state_features),), 
            dtype=np.float32
        )
        
        # Feature normalization
        self.scaler = StandardScaler()
        self._fit_scaler()
    
    def _define_state_features(self) -> List[str]:
        """Define which features to include in state"""
        features = [
            # Price features
            'returns_1', 'returns_5', 'returns_20',
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_ema200',
            
            # Technical indicators
            'rsi_14', 'rsi_6', 'rsi_24',
            'macd_signal', 'macd_histogram',
            'bb_position', 'bb_width',
            'stoch_k', 'stoch_d', 'stoch_cross',
            
            # Volume features
            'volume_ratio', 'volume_trend',
            
            # Market microstructure
            'high_low_ratio', 'close_open_ratio',
            
            # Position information
            'position_held', 'position_pnl', 'holding_period',
            
            # Strategy signals (from your existing strategy)
            'strategy_signal', 'signal_strength'
        ]
        return features
    
    def _fit_scaler(self):
        """Fit scaler on historical features"""
        all_features = []
        for i in range(self.config['lookback_window'], len(self.data)):
            features = self._calculate_features(i)
            all_features.append(features)
        
        if all_features:
            self.scaler.fit(all_features)
    
    def _calculate_features(self, step: int) -> np.ndarray:
        """Calculate state features for given step"""
        features = []
        
        # Price returns
        features.append(self.data['Return'].iloc[step] if 'Return' in self.data else 0)
        features.append(self.data['Close'].iloc[step] / self.data['Close'].iloc[step-5] - 1)
        features.append(self.data['Close'].iloc[step] / self.data['Close'].iloc[step-20] - 1)
        
        # Price vs moving averages
        features.append(self.data['Close'].iloc[step] / self.data['SMA'].iloc[step] - 1 if 'SMA' in self.data else 0)
        features.append(self.data['Close'].iloc[step] / self.data['Close'].rolling(50).mean().iloc[step] - 1)
        features.append(self.data['Close'].iloc[step] / self.data['EMA200'].iloc[step] - 1 if 'EMA200' in self.data else 0)
        
        # RSI values (normalized)
        features.append((self.data['RSI14'].iloc[step] - 50) / 50 if 'RSI14' in self.data else 0)
        features.append((self.data['RSI6'].iloc[step] - 50) / 50 if 'RSI6' in self.data else 0)
        features.append((self.data['RSI24'].iloc[step] - 50) / 50 if 'RSI24' in self.data else 0)
        
        # MACD
        macd_signal = 1 if self.data['MACD'].iloc[step] > self.data['Signal_Line'].iloc[step] else -1
        features.append(macd_signal)
        features.append(self.data['MACD_Histogram'].iloc[step] / self.data['Close'].iloc[step] if 'MACD_Histogram' in self.data else 0)
        
        # Bollinger Bands
        if 'UpperBand' in self.data and 'LowerBand' in self.data:
            bb_pos = (self.data['Close'].iloc[step] - self.data['LowerBand'].iloc[step]) / (self.data['UpperBand'].iloc[step] - self.data['LowerBand'].iloc[step])
            bb_width = (self.data['UpperBand'].iloc[step] - self.data['LowerBand'].iloc[step]) / self.data['Close'].iloc[step]
        else:
            bb_pos, bb_width = 0.5, 0.1
        features.append(bb_pos)
        features.append(bb_width)
        
        # Stochastic
        features.append(self.data['Stoch_K'].iloc[step] / 100 if 'Stoch_K' in self.data else 0.5)
        features.append(self.data['Stoch_D'].iloc[step] / 100 if 'Stoch_D' in self.data else 0.5)
        stoch_cross = 1 if self.data['Stoch_K'].iloc[step] > self.data['Stoch_D'].iloc[step] else -1
        features.append(stoch_cross)
        
        # Volume
        vol_ratio = self.data['Volume'].iloc[step] / self.data['Volume'].rolling(20).mean().iloc[step]
        features.append(vol_ratio)
        vol_trend = self.data['Volume'].rolling(5).mean().iloc[step] / self.data['Volume'].rolling(20).mean().iloc[step]
        features.append(vol_trend)
        
        # Market microstructure
        features.append(self.data['High'].iloc[step] / self.data['Low'].iloc[step] - 1)
        features.append(self.data['Close'].iloc[step] / self.data['Open'].iloc[step] - 1)
        
        # Position information
        features.append(1 if self.position > 0 else (-1 if self.position < 0 else 0))
        features.append(self.unrealized_pnl / self.config['initial_capital'] if self.position != 0 else 0)
        features.append(self.holding_period / 100)  # Normalized holding period
        
        # Your strategy signals
        strategy_signal = self.data['Signal'].iloc[step] if 'Signal' in self.data else 0
        features.append(strategy_signal)
        
        # Signal strength (how many conditions are met)
        signal_strength = 0
        if 'RSI14' in self.data:
            if self.data['RSI14'].iloc[step] > 50: signal_strength += 0.2
            if self.data['Close'].iloc[step] > self.data['EMA200'].iloc[step]: signal_strength += 0.2
            if 'MACD' in self.data and self.data['MACD'].iloc[step] > self.data['Signal_Line'].iloc[step]: signal_strength += 0.2
            if 'Stoch_K' in self.data and self.data['Stoch_K'].iloc[step] > self.data['Stoch_D'].iloc[step]: signal_strength += 0.2
            if bb_pos < 0.2: signal_strength += 0.2  # Near lower band
        features.append(signal_strength)
        
        return np.array(features, dtype=np.float32)
    
    def reset(self):
        """Reset environment"""
        self.current_step = self.config['lookback_window']
        self.capital = self.config['initial_capital']
        self.position = 0
        self.entry_price = 0
        self.holding_period = 0
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        self.trades = []
        self.equity_curve = [self.capital]
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state"""
        features = self._calculate_features(self.current_step)
        # Normalize features
        if hasattr(self.scaler, 'mean_'):
            features = self.scaler.transform([features])[0]
        return features
    
    def step(self, action):
        """Execute action and return new state"""
        current_price = self.data['Close'].iloc[self.current_step]
        prev_value = self._get_portfolio_value()
        
        # Execute action
        if self.discrete_actions:
            self._execute_discrete_action(action, current_price)
        else:
            self._execute_continuous_action(action, current_price)
        
        # Update unrealized P&L
        if self.position != 0:
            self.unrealized_pnl = (current_price - self.entry_price) * abs(self.position)
            self.holding_period += 1
        
        # Check stop loss / take profit
        if self.position != 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            if pnl_pct <= -self.config['stop_loss'] or pnl_pct >= self.config['take_profit']:
                self._close_position(current_price, 'SL/TP')
        
        # Calculate reward
        current_value = self._get_portfolio_value()
        reward = self._calculate_reward(prev_value, current_value, action)
        
        # Update state
        self.current_step += 1
        self.equity_curve.append(current_value)
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        if done and self.position != 0:
            self._close_position(current_price, 'EOD')
        
        # Get new state
        state = self._get_state()
        
        info = {
            'portfolio_value': current_value,
            'position': self.position,
            'trades': len(self.trades),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl
        }
        
        return state, reward, done, info
    
    def _execute_discrete_action(self, action, price):
        """Execute discrete action (0=Hold, 1=Buy, 2=Sell)"""
        if action == 1 and self.position <= 0:  # Buy
            if self.position < 0:  # Close short
                self._close_position(price, 'Signal')
            # Open long
            position_size = self.capital * self.config['max_position']
            shares = position_size / price
            cost = shares * price * (1 + self.config['transaction_cost'])
            
            if cost <= self.capital:
                self.position = shares
                self.capital -= cost
                self.entry_price = price
                self.holding_period = 0
                
        elif action == 2 and self.position >= 0:  # Sell
            if self.position > 0:  # Close long
                self._close_position(price, 'Signal')
            # Open short (if allowed)
            # position_size = self.capital * self.config['max_position']
            # shares = -position_size / price
            # self.position = shares
            # self.entry_price = price
            # self.holding_period = 0
    
    def _close_position(self, price, reason):
        """Close current position"""
        if self.position > 0:  # Long position
            revenue = self.position * price * (1 - self.config['transaction_cost'])
            pnl = revenue - (self.position * self.entry_price)
            self.capital += revenue
        else:  # Short position
            cost = abs(self.position) * price * (1 + self.config['transaction_cost'])
            pnl = abs(self.position) * self.entry_price - cost
            self.capital += abs(self.position) * self.entry_price - cost
        
        self.realized_pnl += pnl
        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': price,
            'position': self.position,
            'pnl': pnl,
            'holding_period': self.holding_period,
            'reason': reason
        })
        
        self.position = 0
        self.entry_price = 0
        self.holding_period = 0
        self.unrealized_pnl = 0
    
    def _get_portfolio_value(self):
        """Calculate total portfolio value"""
        if self.position > 0:
            return self.capital + self.position * self.data['Close'].iloc[self.current_step]
        elif self.position < 0:
            return self.capital + abs(self.position) * self.entry_price - abs(self.position) * self.data['Close'].iloc[self.current_step]
        return self.capital
    
    def _calculate_reward(self, prev_value, current_value, action):
        """Advanced reward calculation"""
        # Base reward: portfolio return
        returns = (current_value - prev_value) / prev_value
        reward = returns * self.config['reward_scaling']
        
        # Risk-adjusted reward (Sharpe ratio component)
        if len(self.equity_curve) > 20:
            recent_returns = pd.Series(self.equity_curve).pct_change().dropna().tail(20)
            if len(recent_returns) > 1 and recent_returns.std() > 0:
                sharpe = recent_returns.mean() / recent_returns.std()
                reward += sharpe * 0.1
        
        # Profit factor bonus
        if len(self.trades) >= 5:
            winning_trades = [t['pnl'] for t in self.trades if t['pnl'] > 0]
            losing_trades = [abs(t['pnl']) for t in self.trades if t['pnl'] < 0]
            if losing_trades:
                profit_factor = sum(winning_trades) / sum(losing_trades)
                reward += (profit_factor - 1) * 0.05
        
        # Penalty for excessive trading
        if action != 0:  # Any trade
            reward -= 0.001 * self.config['reward_scaling']
        
        # Reward for following good signals
        if 'Signal' in self.data:
            strategy_signal = self.data['Signal'].iloc[self.current_step]
            if (action == 1 and strategy_signal == 1) or (action == 2 and strategy_signal == -1):
                reward += 0.01 * self.config['reward_scaling']
        
        return reward


class A2CAgent:
    """Advantage Actor-Critic Agent"""
    
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default config
        self.config = {
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'tau': 0.95,  # GAE parameter
            'entropy_coef': 0.01,
            'value_loss_coef': 0.5,
            'max_grad_norm': 0.5,
            'hidden_size': 256
        }
        if config:
            self.config.update(config)
        
        # Networks
        self.actor_critic = self._build_actor_critic().to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.config['learning_rate'])
        
    def _build_actor_critic(self):
        """Build shared actor-critic network"""
        class ActorCritic(nn.Module):
            def __init__(self, state_size, action_size, hidden_size):
                super().__init__()
                # Shared layers
                self.shared = nn.Sequential(
                    nn.Linear(state_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Actor head
                self.actor = nn.Sequential(
                    nn.Linear(hidden_size, action_size),
                    nn.Softmax(dim=-1)
                )
                
                # Critic head
                self.critic = nn.Linear(hidden_size, 1)
                
            def forward(self, state):
                features = self.shared(state)
                policy = self.actor(features)
                value = self.critic(features)
                return policy, value
        
        return ActorCritic(self.state_size, self.action_size, self.config['hidden_size'])
    
    def act(self, state, deterministic=False):
        """Select action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value = self.actor_critic(state_tensor)
        
        if deterministic:
            action = torch.argmax(policy, dim=1).item()
        else:
            dist = Categorical(policy)
            action = dist.sample().item()
        
        return action, policy[0, action].item(), value.item()
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0
            else:
                next_value = next_values[i]
            
            delta = rewards[i] + self.config['gamma'] * next_value - values[i]
            gae = delta + self.config['gamma'] * self.config['tau'] * gae * (1 - dones[i])
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, states, actions, rewards, next_states, dones):
        """Update actor-critic networks"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Forward pass
        policies, values = self.actor_critic(states)
        _, next_values = self.actor_critic(next_states)
        
        # Compute advantages
        advantages = self.compute_gae(
            rewards.cpu().numpy(),
            values.squeeze().cpu().numpy(),
            next_values.squeeze().cpu().numpy(),
            dones.cpu().numpy()
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = advantages + values.squeeze()
        
        # Actor loss
        dist = Categorical(policies)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Critic loss
        critic_loss = nn.MSELoss()(values.squeeze(), returns.detach())
        
        # Total loss
        loss = actor_loss - self.config['entropy_coef'] * entropy + self.config['value_loss_coef'] * critic_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()
        
        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()


class DQNAgent:
    """Deep Q-Network Agent with Experience Replay and Target Network"""
    
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default config
        self.config = {
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update_freq': 100,
            'hidden_size': 256,
            'double_dqn': True,
            'dueling_dqn': True
        }
        if config:
            self.config.update(config)
        
        # Networks
        self.q_network = self._build_q_network().to(self.device)
        self.target_network = self._build_q_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])
        
        # Experience replay
        self.memory = deque(maxlen=self.config['memory_size'])
        
        # Training variables
        self.epsilon = self.config['epsilon']
        self.update_count = 0
        
        # Initialize target network
        self.update_target_network()
    
    def _build_q_network(self):
        """Build Q-network with optional dueling architecture"""
        if self.config['dueling_dqn']:
            class DuelingQNetwork(nn.Module):
                def __init__(self, state_size, action_size, hidden_size):
                    super().__init__()
                    # Shared layers
                    self.shared = nn.Sequential(
                        nn.Linear(state_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                    
                    # Value stream
                    self.value_stream = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size // 2, 1)
                    )
                    
                    # Advantage stream
                    self.advantage_stream = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size // 2, action_size)
                    )
                
                def forward(self, state):
                    features = self.shared(state)
                    value = self.value_stream(features)
                    advantage = self.advantage_stream(features)
                    
                    # Combine value and advantage
                    q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
                    return q_values
            
            return DuelingQNetwork(self.state_size, self.action_size, self.config['hidden_size'])
        else:
            class QNetwork(nn.Module):
                def __init__(self, state_size, action_size, hidden_size):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(state_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size // 2, action_size)
                    )
                
                def forward(self, state):
                    return self.network(state)
            
            return QNetwork(self.state_size, self.action_size, self.config['hidden_size'])
    
    def act(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy"""
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def replay(self, batch_size=None):
        """Train the network on a batch of experiences"""
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.config['double_dqn']:
                # Double DQN: use main network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.config['gamma'] * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.config['epsilon_min']:
            self.epsilon *= self.config['epsilon_decay']
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.config['target_update_freq'] == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())


class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default config
        self.config = {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'value_loss_coef': 0.5,
            'max_grad_norm': 0.5,
            'ppo_epochs': 4,
            'mini_batch_size': 64,
            'hidden_size': 256,
            'normalize_advantages': True
        }
        if config:
            self.config.update(config)
        
        # Networks
        self.actor_critic = self._build_actor_critic().to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.config['learning_rate'])
        
        # Storage for rollouts
        self.rollout_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'advantages': [],
            'returns': []
        }
    
    def _build_actor_critic(self):
        """Build actor-critic network"""
        class ActorCritic(nn.Module):
            def __init__(self, state_size, action_size, hidden_size):
                super().__init__()
                # Shared layers
                self.shared = nn.Sequential(
                    nn.Linear(state_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Actor head
                self.actor = nn.Linear(hidden_size, action_size)
                
                # Critic head
                self.critic = nn.Linear(hidden_size, 1)
            
            def forward(self, state):
                features = self.shared(state)
                return features
            
            def get_action_and_value(self, state):
                features = self.forward(state)
                logits = self.actor(features)
                value = self.critic(features)
                
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                return action, log_prob, dist.entropy(), value
            
            def get_value(self, state):
                features = self.forward(state)
                return self.critic(features)
            
            def evaluate_actions(self, state, action):
                features = self.forward(state)
                logits = self.actor(features)
                value = self.critic(features)
                
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                
                return log_prob, entropy, value
        
        return ActorCritic(self.state_size, self.action_size, self.config['hidden_size'])
    
    def get_action(self, state, eval_mode=False):
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if eval_mode:
                # Deterministic action for evaluation
                features = self.actor_critic(state_tensor)
                logits = self.actor_critic.actor(features)
                action = torch.argmax(logits, dim=-1).item()
                return action, None
            else:
                # Stochastic action for training
                action, log_prob, entropy, value = self.actor_critic.get_action_and_value(state_tensor)
                return action.item(), log_prob.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in rollout buffer"""
        self.rollout_buffer['states'].append(state)
        self.rollout_buffer['actions'].append(action)
        self.rollout_buffer['rewards'].append(reward)
        self.rollout_buffer['values'].append(value)
        self.rollout_buffer['log_probs'].append(log_prob)
        self.rollout_buffer['dones'].append(done)
    
    def compute_gae(self, next_value=0):
        """Compute Generalized Advantage Estimation"""
        rewards = self.rollout_buffer['rewards']
        values = self.rollout_buffer['values']
        dones = self.rollout_buffer['dones']
        
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value_i = next_value
            else:
                next_value_i = values[i + 1]
            
            delta = rewards[i] + self.config['gamma'] * next_value_i * (1 - dones[i]) - values[i]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        self.rollout_buffer['advantages'] = advantages
        self.rollout_buffer['returns'] = [adv + val for adv, val in zip(advantages, values)]
    
    def update(self):
        """Update policy using PPO"""
        if len(self.rollout_buffer['states']) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(self.rollout_buffer['states']).to(self.device)
        actions = torch.LongTensor(self.rollout_buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.rollout_buffer['log_probs']).to(self.device)
        advantages = torch.FloatTensor(self.rollout_buffer['advantages']).to(self.device)
        returns = torch.FloatTensor(self.rollout_buffer['returns']).to(self.device)
        
        # Normalize advantages
        if self.config['normalize_advantages']:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(self.config['ppo_epochs']):
            # Create mini-batches
            batch_size = len(states)
            mini_batch_size = min(self.config['mini_batch_size'], batch_size)
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, entropy, values = self.actor_critic.evaluate_actions(batch_states, batch_actions)
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon']) * batch_advantages
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = actor_loss + self.config['value_loss_coef'] * critic_loss - self.config['entropy_coef'] * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                
                total_loss += loss.item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_loss.item()
        
        # Clear buffer
        for key in self.rollout_buffer:
            self.rollout_buffer[key] = []
        
        num_updates = self.config['ppo_epochs'] * (batch_size // mini_batch_size)
        return (total_loss / num_updates, total_actor_loss / num_updates, 
                total_critic_loss / num_updates, total_entropy / num_updates)


class RLTradingPipeline:
    """Complete RL trading pipeline with training, evaluation, and deployment"""
    
    def __init__(self, symbol: str, agent_type: str = 'A2C'):
        self.symbol = symbol
        self.agent_type = agent_type
        self.agent = None
        self.env = None
        self.training_history = {
            'episode_rewards': [],
            'episode_returns': [],
            'episode_trades': [],
            'episode_sharpe': []
        }
    
    def prepare_data(self, start_date="2020-01-01", end_date=None):
        """Download and prepare data with indicators"""
        print(f"Downloading data for {self.symbol}...")
        data = yf.download(self.symbol, start=start_date, end=end_date)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Calculate all indicators
        data = calculate_indicators(data)
        
        # Generate signals from your strategy
        data = generate_signals(data)
        
        # Add returns
        data['Return'] = data['Close'].pct_change()
        
        return data
    
    def train(self, episodes=200, save_path=None):
        """Train the RL agent"""
        # Prepare training data
        train_data = self.prepare_data("2020-01-01", "2023-01-01")
        
        # Create environment
        env_config = {
            'initial_capital': 10000,
            'max_position': 1.0,
            'transaction_cost': 0.001,
            'stop_loss': 0.01,
            'take_profit': 0.02
        }
        self.env = RLTradingEnvironment(train_data, env_config)
        
        # Create agent
        if self.agent_type == 'A2C':
            self.agent = A2CAgent(
                state_size=self.env.observation_space.shape[0],
                action_size=self.env.action_space.n
            )
        elif self.agent_type == 'DQN':
            self.agent = DQNAgent(
                state_size=self.env.observation_space.shape[0],
                action_size=self.env.action_space.n
            )
        elif self.agent_type == 'PPO':
            self.agent = PPOAgent(
                state_size=self.env.observation_space.shape[0],
                action_size=self.env.action_space.n
            )
        
        print(f"\nTraining {self.agent_type} agent for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_data = []
            
            while True:
                # Select action
                if self.agent_type == 'A2C':
                    action, _, _ = self.agent.act(state)
                elif self.agent_type == 'DQN':
                    action = self.agent.act(state)
                elif self.agent_type == 'PPO':
                    action, log_prob = self.agent.get_action(state)
                    # Get value for PPO
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    with torch.no_grad():
                        value = self.agent.actor_critic.get_value(state_tensor).item()
                
                # Step environment
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                # Store experience
                if self.agent_type == 'PPO':
                    # For PPO, store in rollout buffer
                    self.agent.store_transition(state, action, reward, value, log_prob, done)
                else:
                    # For A2C and DQN
                    episode_data.append((state, action, reward, next_state, done))
                
                state = next_state
                
                if done:
                    # Calculate episode metrics
                    portfolio_return = (info['portfolio_value'] - env_config['initial_capital']) / env_config['initial_capital'] * 100
                    
                    # Calculate Sharpe ratio
                    if len(self.env.equity_curve) > 1:
                        returns = pd.Series(self.env.equity_curve).pct_change().dropna()
                        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    else:
                        sharpe = 0
                    
                    # Store metrics
                    self.training_history['episode_rewards'].append(episode_reward)
                    self.training_history['episode_returns'].append(portfolio_return)
                    self.training_history['episode_trades'].append(len(self.env.trades))
                    self.training_history['episode_sharpe'].append(sharpe)
                    
                    # Update agent
                    if self.agent_type == 'A2C' and len(episode_data) > 0:
                        states = [e[0] for e in episode_data]
                        actions = [e[1] for e in episode_data]
                        rewards = [e[2] for e in episode_data]
                        next_states = [e[3] for e in episode_data]
                        dones = [e[4] for e in episode_data]
                        
                        self.agent.update(states, actions, rewards, next_states, dones)
                    
                    elif self.agent_type == 'DQN':
                        # Store experiences in replay buffer
                        for exp in episode_data:
                            self.agent.remember(*exp)
                        
                        # Train on batch
                        if len(self.agent.memory) > 32:
                            for _ in range(10):
                                self.agent.replay(32)
                        
                        # Update target network
                        if episode % 10 == 0:
                            self.agent.update_target_network()
                    
                    elif self.agent_type == 'PPO':
                        # For PPO, compute GAE and update policy
                        # Get final value for GAE computation
                        if done:
                            next_value = 0
                        else:
                            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.agent.device)
                            with torch.no_grad():
                                next_value = self.agent.actor_critic.get_value(next_state_tensor).item()
                        
                        self.agent.compute_gae(next_value)
                        self.agent.update()
                    
                    # Print progress
                    if episode % 10 == 0:
                        avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                        avg_return = np.mean(self.training_history['episode_returns'][-10:])
                        avg_sharpe = np.mean(self.training_history['episode_sharpe'][-10:])
                        
                        print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                              f"Avg Return: {avg_return:.2f}%, Avg Sharpe: {avg_sharpe:.2f}")
                    
                    break
        
        # Save model if requested
        if save_path:
            self.save_model(save_path)
        
        return self.training_history
    
    def evaluate(self, start_date="2023-01-01", end_date=None):
        """Evaluate the trained agent on test data"""
        if self.agent is None:
            raise ValueError("No trained agent found. Please train first.")
        
        # Prepare test data
        test_data = self.prepare_data(start_date, end_date)
        
        # Create test environment
        test_env = RLTradingEnvironment(test_data, self.env.config)
        
        # Run evaluation
        state = test_env.reset()
        total_reward = 0
        actions_taken = []
        
        while True:
            # Select action (deterministic for evaluation)
            if self.agent_type == 'A2C':
                action, _, _ = self.agent.act(state, deterministic=True)
            elif self.agent_type == 'DQN':
                action = self.agent.act(state, eval_mode=True)
            elif self.agent_type == 'PPO':
                action, _ = self.agent.get_action(state, eval_mode=True)
            
            actions_taken.append(action)
            
            # Step
            state, reward, done, info = test_env.step(action)
            total_reward += reward
            
            if done:
                break
        
        # Calculate performance metrics
        initial_capital = test_env.config['initial_capital']
        final_value = info['portfolio_value']
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate Sharpe ratio
        equity_curve = pd.Series(test_env.equity_curve)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Win rate
        if test_env.trades:
            winning_trades = [t for t in test_env.trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(test_env.trades) * 100
        else:
            win_rate = 0
        
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(test_env.trades),
            'win_rate': win_rate,
            'final_value': final_value,
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'trades': test_env.trades,
            'equity_curve': test_env.equity_curve,
            'actions': actions_taken
        }
        
        return results
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown"""
        cumulative = pd.Series(equity_curve)
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def save_model(self, filepath):
        """Save trained model"""
        save_dict = {
            'agent_type': self.agent_type,
            'symbol': self.symbol,
            'training_history': self.training_history,
            'env_config': self.env.config if self.env else None
        }
        
        if self.agent_type == 'A2C':
            save_dict['model_state'] = self.agent.actor_critic.state_dict()
            save_dict['optimizer_state'] = self.agent.optimizer.state_dict()
            save_dict['agent_config'] = self.agent.config
        elif self.agent_type == 'DQN':
            save_dict['q_network_state'] = self.agent.q_network.state_dict()
            save_dict['target_network_state'] = self.agent.target_network.state_dict()
            save_dict['optimizer_state'] = self.agent.optimizer.state_dict()
            save_dict['epsilon'] = self.agent.epsilon
        elif self.agent_type == 'PPO':
            save_dict['model_state'] = self.agent.actor_critic.state_dict()
            save_dict['optimizer_state'] = self.agent.optimizer.state_dict()
            save_dict['agent_config'] = self.agent.config
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        
        self.agent_type = checkpoint['agent_type']
        self.symbol = checkpoint['symbol']
        self.training_history = checkpoint['training_history']
        
        # Recreate agent
        if self.agent_type == 'A2C':
            self.agent = A2CAgent(
                state_size=20,  # You might need to adjust this
                action_size=3,
                config=checkpoint.get('agent_config')
            )
            self.agent.actor_critic.load_state_dict(checkpoint['model_state'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        elif self.agent_type == 'DQN':
            self.agent = DQNAgent(
                state_size=20,
                action_size=3,
                config=checkpoint.get('agent_config')
            )
            self.agent.q_network.load_state_dict(checkpoint['q_network_state'])
            self.agent.target_network.load_state_dict(checkpoint['target_network_state'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.agent.epsilon = checkpoint['epsilon']
        elif self.agent_type == 'PPO':
            self.agent = PPOAgent(
                state_size=20,
                action_size=3,
                config=checkpoint.get('agent_config')
            )
            self.agent.actor_critic.load_state_dict(checkpoint['model_state'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"Model loaded from {filepath}")
    
    def plot_results(self, results):
        """Plot evaluation results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Equity curve
        axes[0].plot(results['equity_curve'])
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].grid(True)
        
        # Actions taken
        actions_df = pd.DataFrame({'action': results['actions']})
        actions_df['buy'] = (actions_df['action'] == 1).astype(int)
        actions_df['sell'] = (actions_df['action'] == 2).astype(int)
        
        axes[1].plot(actions_df['buy'].cumsum(), label='Buy signals', color='green')
        axes[1].plot(actions_df['sell'].cumsum(), label='Sell signals', color='red')
        axes[1].set_title('Cumulative Trading Signals')
        axes[1].set_ylabel('Number of Signals')
        axes[1].legend()
        axes[1].grid(True)
        
        # Trade P&L distribution
        if results['trades']:
            trade_pnls = [t['pnl'] for t in results['trades']]
            axes[2].hist(trade_pnls, bins=20, alpha=0.7, color='blue')
            axes[2].axvline(x=0, color='red', linestyle='--')
            axes[2].set_title('Trade P&L Distribution')
            axes[2].set_xlabel('P&L ($)')
            axes[2].set_ylabel('Frequency')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_rl_results.png', dpi=150)
        plt.show()
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Final Portfolio Value: ${results['final_value']:.2f}")


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = RLTradingPipeline('NVDA', agent_type='A2C')
    
    # Train agent
    training_history = pipeline.train(episodes=100)
    
    # Evaluate on test data
    results = pipeline.evaluate(start_date="2023-01-01")
    
    # Plot results
    pipeline.plot_results(results)
    
    # Compare with your original strategy
    print("\nComparing with original strategy...")
    from bot import run_trading_algorithm
    
    original_data, original_metrics = run_trading_algorithm('NVDA', start_date="2023-01-01")
    print(f"\nOriginal Strategy Win Rate: {original_metrics['Win Rate']:.2f}%")
    print(f"RL Strategy Win Rate: {results['win_rate']:.2f}%")