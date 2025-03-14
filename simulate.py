import pandas as pd
import numpy as np
import random
from typing import Dict, List, Any, Tuple

# Import radCAD instead of cadCAD
from radcad import Model, Simulation, Experiment
from radcad.engine import Engine, Backend

class TokenomicsSimulation:
    """
    A class for running tokenomics simulations using radCAD.
    """
    
    def __init__(self, data):
        """
        Initialize the simulation with tokenomics data.
        
        Args:
            data: TokenomicsData object containing the parsed data
        """
        self.data = data
        self.initial_state = {}
        self.params = {}
        self.timesteps = len(data.dates) if data.dates is not None else 60
        
    def setup_initial_state(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set up the initial state for the simulation.
        
        Args:
            params: Dictionary of simulation parameters
            
        Returns:
            Dictionary representing the initial state
        """
        # Get initial values from the data
        initial_total_supply = self.data.static_params.get("Initial Total Supply of Tokens", 888000000)
        
        # Calculate initial circulating supply (sum of all buckets except Liquidity Pool)
        initial_circulating_supply = 0
        if not self.data.vesting_cumulative.empty and "Liquidity Pool" in self.data.vesting_cumulative.index:
            initial_circulating_supply = self.data.vesting_cumulative.drop("Liquidity Pool", errors='ignore').sum().iloc[0]
        
        # Get initial liquidity pool tokens
        initial_lp_tokens = 0
        if not self.data.vesting_cumulative.empty and "Liquidity Pool" in self.data.vesting_cumulative.index:
            initial_lp_tokens = self.data.vesting_cumulative.loc["Liquidity Pool"].iloc[0]
        
        # Get initial token price
        initial_token_price = params.get("token_price", self.data.token_price)
        if isinstance(initial_token_price, list):
            initial_token_price = initial_token_price[0] if initial_token_price else 0.03
        
        # Get initial staking APR
        initial_staking_apr = 0.05  # Default value
        if not self.data.staking_apr.empty:
            initial_staking_apr = self.data.staking_apr.loc["Staking APR"].iloc[0]
        
        # Set up the initial state
        self.initial_state = {
            "token_supply": initial_total_supply,
            "circulating_supply": initial_circulating_supply,
            "staked_tokens": 0,  # Start with 0 staked tokens
            "liquidity_pool_tokens": initial_lp_tokens,
            "token_price": initial_token_price,
            "market_cap": initial_circulating_supply * initial_token_price,
            "staking_apr": initial_staking_apr,
            "time_step": 0
        }
        
        return self.initial_state
    
    def p_vesting_schedule(self, params, substep, state_history, prev_state):
        """
        Policy function for token vesting schedule.
        
        Args:
            params: Simulation parameters
            substep: Current substep
            state_history: History of states
            prev_state: Previous state
            
        Returns:
            Dictionary with newly vested tokens
        """
        # Get current time step
        time_step = prev_state["time_step"]
        
        # Calculate newly vested tokens based on vesting schedule
        newly_vested = {}
        for bucket in self.data.vesting_series.index:
            if bucket != "SUM" and bucket != "Liquidity Pool":
                if time_step < len(self.data.vesting_series.columns):
                    newly_vested[bucket] = self.data.vesting_series.loc[bucket].iloc[time_step]
                else:
                    newly_vested[bucket] = 0
        
        return {"newly_vested": sum(newly_vested.values())}
    
    def p_staking(self, params, substep, state_history, prev_state):
        """
        Policy function for staking behavior.
        
        Args:
            params: Simulation parameters
            substep: Current substep
            state_history: History of states
            prev_state: Previous state
            
        Returns:
            Dictionary with staking delta
        """
        # Get staking share parameter
        staking_share = params["staking_share"]
        
        # Get current circulating supply and staked tokens
        circulating_supply = prev_state["circulating_supply"]
        current_staked = prev_state["staked_tokens"]
        
        # Calculate target staked amount based on staking share
        target_staked = circulating_supply * staking_share
        
        # Calculate staking delta (how many tokens to stake or unstake)
        staking_delta = target_staked - current_staked
        
        # Apply constraints (can't stake more than available)
        if staking_delta > 0:
            max_new_staking = min(staking_delta, circulating_supply - current_staked)
        else:
            max_new_staking = staking_delta  # Unstaking
        
        return {"staking_delta": max_new_staking}
    
    def p_token_price(self, params, substep, state_history, prev_state):
        """
        Policy function for token price dynamics.
        
        Args:
            params: Simulation parameters
            substep: Current substep
            state_history: History of states
            prev_state: Previous state
            
        Returns:
            Dictionary with price delta
        """
        # Get current price and market conditions
        current_price = prev_state["token_price"]
        market_volatility = params.get("market_volatility", 0.2)
        
        # Calculate staking ratio (staked tokens / circulating supply)
        staked_ratio = prev_state["staked_tokens"] / prev_state["circulating_supply"] if prev_state["circulating_supply"] > 0 else 0
        
        # Price increases with higher staking ratio (basic model)
        price_multiplier = 1 + (staked_ratio - 0.5) * 0.1
        
        # Add some randomness based on market volatility
        random_factor = 1 + (random.random() - 0.5) * market_volatility
        
        # Calculate new price
        new_price = current_price * price_multiplier * random_factor
        
        # Ensure price doesn't go below a minimum value
        min_price = 0.001
        new_price = max(new_price, min_price)
        
        return {"price_delta": new_price - current_price}
    
    def s_update_supply(self, params, substep, state_history, prev_state, policy_input):
        """
        State update function for circulating supply.
        
        Args:
            params: Simulation parameters
            substep: Current substep
            state_history: History of states
            prev_state: Previous state
            policy_input: Policy function outputs
            
        Returns:
            New circulating supply value
        """
        # Update circulating supply based on vesting
        newly_vested_total = policy_input["newly_vested"]
        
        return ("circulating_supply", prev_state["circulating_supply"] + newly_vested_total)
    
    def s_update_staking(self, params, substep, state_history, prev_state, policy_input):
        """
        State update function for staked tokens.
        
        Args:
            params: Simulation parameters
            substep: Current substep
            state_history: History of states
            prev_state: Previous state
            policy_input: Policy function outputs
            
        Returns:
            New staked tokens value
        """
        return ("staked_tokens", prev_state["staked_tokens"] + policy_input["staking_delta"])
    
    def s_update_price(self, params, substep, state_history, prev_state, policy_input):
        """
        State update function for token price.
        
        Args:
            params: Simulation parameters
            substep: Current substep
            state_history: History of states
            prev_state: Previous state
            policy_input: Policy function outputs
            
        Returns:
            New token price value
        """
        return ("token_price", prev_state["token_price"] + policy_input["price_delta"])
    
    def s_update_market_cap(self, params, substep, state_history, prev_state, policy_input):
        """
        State update function for market cap.
        
        Args:
            params: Simulation parameters
            substep: Current substep
            state_history: History of states
            prev_state: Previous state
            policy_input: Policy function outputs
            
        Returns:
            New market cap value
        """
        return ("market_cap", prev_state["circulating_supply"] * prev_state["token_price"])
    
    def s_update_time(self, params, substep, state_history, prev_state, policy_input):
        """
        State update function for time step.
        
        Args:
            params: Simulation parameters
            substep: Current substep
            state_history: History of states
            prev_state: Previous state
            policy_input: Policy function outputs
            
        Returns:
            New time step value
        """
        return ("time_step", prev_state["time_step"] + 1)
    
    def run_simulation(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Run a radCAD simulation with the given parameters.
        
        Args:
            params: Dictionary of simulation parameters
            
        Returns:
            DataFrame with simulation results
        """
        # Set up initial state
        initial_state = self.setup_initial_state(params)
        
        # Define state update blocks
        state_update_blocks = [
            {
                'policies': {
                    'vesting': self.p_vesting_schedule,
                    'staking': self.p_staking,
                    'price': self.p_token_price
                },
                'variables': {
                    'circulating_supply': self.s_update_supply,
                    'staked_tokens': self.s_update_staking,
                    'token_price': self.s_update_price,
                    'market_cap': self.s_update_market_cap,
                    'time_step': self.s_update_time
                }
            }
        ]
        
        # Create model
        model = Model(
            initial_state=initial_state,
            state_update_blocks=state_update_blocks,
            params=params
        )
        
        # Create simulation
        simulation = Simulation(
            model=model,
            timesteps=self.timesteps,
            runs=1  # Number of monte carlo runs
        )
        
        # Create experiment
        experiment = Experiment(simulation)
        
        # Run experiment
        result = experiment.run()
        
        # Convert to DataFrame
        result_df = pd.DataFrame(result)
        
        # Add dates
        result_df['date'] = [self.data.dates[i] if i < len(self.data.dates) else None for i in result_df['time_step']]
        
        return result_df
    
    def run_scenario_comparison(self, scenarios: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Run multiple simulations for scenario comparison.
        
        Args:
            scenarios: List of dictionaries, each containing a scenario name and parameters
            
        Returns:
            Dictionary mapping scenario names to simulation results
        """
        results = {}
        
        for scenario in scenarios:
            name = scenario.get("name", f"Scenario {len(results) + 1}")
            params = scenario.get("params", {})
            
            # Run simulation for this scenario
            result = self.run_simulation(params)
            
            # Store results
            results[name] = result
        
        return results 