"""
Agent-based model module for the Unit Zero Labs Tokenomics Engine.
Provides implementations of different agent types and behaviors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
import random


class Agent:
    """Base class for all agents in the simulation."""
    
    def __init__(self, agent_id: int, initial_tokens: float = 0, initial_cash: float = 0):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_tokens: Initial token holdings
            initial_cash: Initial cash holdings (in USD)
        """
        self.agent_id = agent_id
        self.tokens = initial_tokens
        self.cash = initial_cash
        self.type = "base"
        self.history = []  # Transaction history
    
    def decide(self, state: Dict[str, Any], params: Dict[str, Any]) -> Tuple[str, float]:
        """
        Make a decision based on the current state.
        
        Args:
            state: Current state of the simulation
            params: Simulation parameters
            
        Returns:
            Tuple of (action, amount)
            action: "buy", "sell", "hold", "stake", "unstake"
            amount: Amount of tokens to buy/sell/stake/unstake
        """
        # Base agent just holds
        return "hold", 0
    
    def record_transaction(self, action: str, tokens: float, price: float):
        """
        Record a transaction in the agent's history.
        
        Args:
            action: Type of transaction ("buy", "sell", "stake", "unstake")
            tokens: Amount of tokens involved
            price: Token price at the time of transaction
        """
        self.history.append({
            "action": action,
            "tokens": tokens,
            "price": price,
            "cash_impact": -tokens * price if action == "buy" else tokens * price if action == "sell" else 0
        })
    
    def get_portfolio_value(self, token_price: float) -> float:
        """
        Calculate the agent's portfolio value.
        
        Args:
            token_price: Current token price
            
        Returns:
            Total portfolio value (tokens + cash)
        """
        return self.tokens * token_price + self.cash


class RandomTrader(Agent):
    """Agent that makes random trading decisions."""
    
    def __init__(self, agent_id: int, initial_tokens: float = 0, initial_cash: float = 1000):
        """Initialize a random trader agent."""
        super().__init__(agent_id, initial_tokens, initial_cash)
        self.type = "random_trader"
        
        # Agent parameters
        self.trade_probability = 0.3  # Probability of making a trade
        self.max_position_pct = 0.5  # Maximum percentage of portfolio in tokens
    
    def decide(self, state: Dict[str, Any], params: Dict[str, Any]) -> Tuple[str, float]:
        """Make a random trading decision."""
        # Only trade with some probability
        if random.random() > self.trade_probability:
            return "hold", 0
        
        # Get current token price
        token_price = state["token_price"]
        
        # Calculate portfolio value
        portfolio_value = self.get_portfolio_value(token_price)
        
        # Calculate current token position as percentage of portfolio
        token_position_pct = (self.tokens * token_price) / portfolio_value if portfolio_value > 0 else 0
        
        # Decide whether to buy or sell
        if token_position_pct < self.max_position_pct and self.cash > 0:
            # Buy tokens
            action = "buy"
            max_tokens_to_buy = self.cash / token_price
            amount = random.uniform(0, max_tokens_to_buy)
        elif self.tokens > 0:
            # Sell tokens
            action = "sell"
            amount = random.uniform(0, self.tokens)
        else:
            # Hold
            action = "hold"
            amount = 0
        
        return action, amount


class TrendFollower(Agent):
    """Agent that follows price trends."""
    
    def __init__(self, agent_id: int, initial_tokens: float = 0, initial_cash: float = 1000):
        """Initialize a trend follower agent."""
        super().__init__(agent_id, initial_tokens, initial_cash)
        self.type = "trend_follower"
        
        # Agent parameters
        self.lookback_periods = 5  # Number of periods to look back for trend
        self.position_size_pct = 0.2  # Position size as percentage of portfolio
        self.price_history = []  # Track price history
    
    def decide(self, state: Dict[str, Any], params: Dict[str, Any]) -> Tuple[str, float]:
        """Make a trend-following decision."""
        # Get current token price
        token_price = state["token_price"]
        
        # Update price history
        self.price_history.append(token_price)
        if len(self.price_history) > self.lookback_periods:
            self.price_history.pop(0)
        
        # Need enough price history to detect a trend
        if len(self.price_history) < self.lookback_periods:
            return "hold", 0
        
        # Calculate price trend
        start_price = self.price_history[0]
        end_price = self.price_history[-1]
        price_change = (end_price - start_price) / start_price if start_price > 0 else 0
        
        # Calculate portfolio value
        portfolio_value = self.get_portfolio_value(token_price)
        
        # Decide based on trend
        if price_change > 0.05:  # Uptrend
            # Buy tokens
            action = "buy"
            target_token_value = portfolio_value * self.position_size_pct
            current_token_value = self.tokens * token_price
            if current_token_value < target_token_value and self.cash > 0:
                tokens_to_buy = (target_token_value - current_token_value) / token_price
                amount = min(tokens_to_buy, self.cash / token_price)
            else:
                action = "hold"
                amount = 0
        elif price_change < -0.05:  # Downtrend
            # Sell tokens
            action = "sell"
            if self.tokens > 0:
                amount = self.tokens * 0.5  # Sell half of tokens
            else:
                action = "hold"
                amount = 0
        else:
            # Hold
            action = "hold"
            amount = 0
        
        return action, amount


class StakingAgent(Agent):
    """Agent that focuses on staking tokens."""
    
    def __init__(self, agent_id: int, initial_tokens: float = 100, initial_cash: float = 500):
        """Initialize a staking agent."""
        super().__init__(agent_id, initial_tokens, initial_cash)
        self.type = "staking_agent"
        
        # Agent parameters
        self.staked_tokens = 0
        self.target_staking_pct = 0.8  # Target percentage of tokens to stake
        self.min_apr_threshold = 0.03  # Minimum APR to start staking
        self.unstake_apr_threshold = 0.01  # APR below which to unstake
    
    def decide(self, state: Dict[str, Any], params: Dict[str, Any]) -> Tuple[str, float]:
        """Make a staking decision."""
        # Get current staking APR
        staking_apr = state["staking_apr"]
        
        # Calculate current staking percentage
        total_tokens = self.tokens + self.staked_tokens
        staking_pct = self.staked_tokens / total_tokens if total_tokens > 0 else 0
        
        if staking_apr >= self.min_apr_threshold:
            # APR is good, consider staking
            if staking_pct < self.target_staking_pct and self.tokens > 0:
                # Stake more tokens
                action = "stake"
                target_staked = total_tokens * self.target_staking_pct
                amount = min(self.tokens, target_staked - self.staked_tokens)
            else:
                # Already at target staking percentage
                action = "hold"
                amount = 0
        elif staking_apr < self.unstake_apr_threshold and self.staked_tokens > 0:
            # APR is below threshold, unstake
            action = "unstake"
            amount = self.staked_tokens
        else:
            # Hold
            action = "hold"
            amount = 0
        
        return action, amount
    
    def stake_tokens(self, amount: float):
        """Stake tokens."""
        amount = min(amount, self.tokens)
        self.tokens -= amount
        self.staked_tokens += amount
    
    def unstake_tokens(self, amount: float):
        """Unstake tokens."""
        amount = min(amount, self.staked_tokens)
        self.staked_tokens -= amount
        self.tokens += amount
    
    def collect_staking_rewards(self, apr: float, time_period: float = 1/12):
        """
        Collect staking rewards.
        
        Args:
            apr: Annual percentage rate
            time_period: Fraction of a year (default is 1/12 for monthly)
        """
        rewards = self.staked_tokens * apr * time_period
        self.tokens += rewards


class AgentBasedModel:
    """Agent-based model for tokenomics simulation."""
    
    def __init__(self, initial_state: Dict[str, Any], params: Dict[str, Any]):
        """
        Initialize the agent-based model.
        
        Args:
            initial_state: Initial state variables
            params: Simulation parameters
        """
        self.state = initial_state.copy()
        self.params = params.copy()
        self.agents = []
        self.agent_types = {
            "random_trader": RandomTrader,
            "trend_follower": TrendFollower,
            "staking_agent": StakingAgent
        }
        self.results = []
    
    def create_agents(self, agent_counts: Dict[str, int]):
        """
        Create agents based on specified counts for each type.
        
        Args:
            agent_counts: Dictionary mapping agent types to counts
        """
        agent_id = 0
        
        for agent_type, count in agent_counts.items():
            if agent_type in self.agent_types:
                agent_class = self.agent_types[agent_type]
                for _ in range(count):
                    agent = agent_class(agent_id)
                    self.agents.append(agent)
                    agent_id += 1
    
    def process_agent_decisions(self):
        """Process all agent decisions and update the model state."""
        # Get current token price
        token_price = self.state["token_price"]
        
        # Track total buy/sell pressure
        total_buy_tokens = 0
        total_sell_tokens = 0
        
        # Track new staking/unstaking
        total_new_staked = 0
        total_new_unstaked = 0
        
        # Process each agent's decision
        for agent in self.agents:
            action, amount = agent.decide(self.state, self.params)
            
            if action == "buy":
                # Check if agent has enough cash
                max_tokens = agent.cash / token_price
                amount = min(amount, max_tokens)
                
                if amount > 0:
                    # Execute buy
                    cost = amount * token_price
                    agent.tokens += amount
                    agent.cash -= cost
                    agent.record_transaction("buy", amount, token_price)
                    total_buy_tokens += amount
            
            elif action == "sell":
                # Check if agent has enough tokens
                amount = min(amount, agent.tokens)
                
                if amount > 0:
                    # Execute sell
                    revenue = amount * token_price
                    agent.tokens -= amount
                    agent.cash += revenue
                    agent.record_transaction("sell", amount, token_price)
                    total_sell_tokens += amount
            
            elif action == "stake" and isinstance(agent, StakingAgent):
                # Check if agent has enough tokens
                amount = min(amount, agent.tokens)
                
                if amount > 0:
                    # Execute stake
                    agent.stake_tokens(amount)
                    agent.record_transaction("stake", amount, token_price)
                    total_new_staked += amount
            
            elif action == "unstake" and isinstance(agent, StakingAgent):
                # Check if agent has enough staked tokens
                amount = min(amount, agent.staked_tokens)
                
                if amount > 0:
                    # Execute unstake
                    agent.unstake_tokens(amount)
                    agent.record_transaction("unstake", amount, token_price)
                    total_new_unstaked += amount
        
        # Calculate net buy/sell pressure
        net_token_demand = total_buy_tokens - total_sell_tokens
        
        # Update token price based on demand
        liquidity_factor = self.params.get("liquidity_factor", 0.0001)
        price_impact = net_token_demand * liquidity_factor
        
        # Cap the price change to prevent extreme swings
        max_price_change = self.params.get("max_price_change_pct", 0.1)
        price_impact = max(min(price_impact, max_price_change), -max_price_change)
        
        # Update token price
        self.state["token_price"] *= (1 + price_impact)
        
        # Ensure price doesn't go below minimum
        min_price = self.params.get("min_token_price", 0.001)
        self.state["token_price"] = max(self.state["token_price"], min_price)
        
        # Update circulating supply based on staking
        self.state["staked_tokens"] += total_new_staked - total_new_unstaked
        
        # Update effective circulating supply (excluding staked tokens)
        self.state["effective_circulating_supply"] = self.state["circulating_supply"] - self.state["staked_tokens"]
        
        # Update market cap
        self.state["market_cap"] = self.state["circulating_supply"] * self.state["token_price"]
    
    def distribute_staking_rewards(self):
        """Distribute staking rewards to staking agents."""
        staking_apr = self.state["staking_apr"]
        time_period = 1/12  # Monthly
        
        for agent in self.agents:
            if isinstance(agent, StakingAgent) and agent.staked_tokens > 0:
                agent.collect_staking_rewards(staking_apr, time_period)
    
    def update_staking_apr(self):
        """Update staking APR based on total staked tokens."""
        # Simple model: APR decreases as more tokens are staked
        base_apr = self.params.get("base_staking_apr", 0.05)
        staking_ratio = self.state["staked_tokens"] / self.state["circulating_supply"] if self.state["circulating_supply"] > 0 else 0
        
        # APR decreases as staking ratio increases
        apr_scaling_factor = self.params.get("apr_scaling_factor", 0.5)
        self.state["staking_apr"] = base_apr * (1 - staking_ratio * apr_scaling_factor)
        
        # Ensure APR doesn't go below minimum
        min_apr = self.params.get("min_staking_apr", 0.01)
        self.state["staking_apr"] = max(self.state["staking_apr"], min_apr)
    
    def run_simulation(self, timesteps: int) -> pd.DataFrame:
        """
        Run the agent-based simulation for the specified number of timesteps.
        
        Args:
            timesteps: Number of timesteps to simulate
            
        Returns:
            DataFrame with simulation results
        """
        self.results = []
        
        # Initial state
        self.results.append(self.state.copy())
        
        # Run simulation for each timestep
        for t in range(1, timesteps):
            # Process agent decisions
            self.process_agent_decisions()
            
            # Distribute staking rewards
            self.distribute_staking_rewards()
            
            # Update staking APR
            self.update_staking_apr()
            
            # Update time step
            self.state["time_step"] = t
            
            # Update date if available
            if "date" in self.state:
                # Move date forward by one month
                # Assuming date is a pandas Timestamp or datetime object
                try:
                    from dateutil.relativedelta import relativedelta
                    self.state["date"] += relativedelta(months=1)
                except (ImportError, TypeError):
                    # If dateutil is not available or date is not a datetime
                    pass
            
            # Store results
            self.results.append(self.state.copy())
        
        # Convert results to DataFrame
        result_df = pd.DataFrame(self.results)
        return result_df
    
    def get_agent_data(self) -> pd.DataFrame:
        """
        Get data about agent holdings and portfolio values.
        
        Returns:
            DataFrame with agent data
        """
        agent_data = []
        
        # Get current token price
        token_price = self.state["token_price"]
        
        for agent in self.agents:
            data = {
                "agent_id": agent.agent_id,
                "type": agent.type,
                "tokens": agent.tokens,
                "cash": agent.cash,
                "portfolio_value": agent.get_portfolio_value(token_price)
            }
            
            # Add staked tokens if it's a staking agent
            if isinstance(agent, StakingAgent):
                data["staked_tokens"] = agent.staked_tokens
            else:
                data["staked_tokens"] = 0
            
            agent_data.append(data)
        
        return pd.DataFrame(agent_data)