import pandas as pd
import numpy as np
import random
import gc
from typing import Dict, List, Any, Tuple, Union
from radcad import Model, Simulation, Experiment
from radcad.engine import Engine, Backend
import streamlit as st

from logic.agent_model import AgentBasedModel
from logic.parameter_registry import parameter_registry
from logic.policy_factory import create_policy_factory

########################################################
####### UNIT ZERO LABS TOKEN SIMULATION ENGINE #########
########################################################


class TokenomicsSimulation:
    """
    A class for running tokenomics simulations using radCAD or agent-based modeling.
    Enhanced with dynamic policy creation based on available parameters.
    
    Supports both single-run (num_runs=1) and Monte Carlo analysis (num_runs>1) seamlessly.
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
        
        # Initialize dynamic policy system
        self.policy_factory = create_policy_factory(parameter_registry)
        self.dynamic_policies = {}
        self.consumed_params = []
        
    def setup_initial_state(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set up the initial state for the simulation with enhanced state variables.
        
        Args:
            params: Dictionary of simulation parameters
            
        Returns:
            Dictionary representing the initial state
        """
        # Get initial values from the data - try both keys
        initial_total_supply = self.data.static_params.get("initial_total_supply")
        
        # If not found with lowercase key, try the capitalized key
        if initial_total_supply is None:
            initial_total_supply = self.data.static_params.get("Initial Total Supply of Tokens")
            
        # If still not found, use default value
        if initial_total_supply is None:
            initial_total_supply = 888000000
            
        # Calculate initial circulating supply (sum of all buckets except Liquidity Pool)
        initial_circulating_supply = 0
        if not self.data.vesting_cumulative.empty:
            if "Liquidity Pool" in self.data.vesting_cumulative.index:
                circulating_sum = self.data.vesting_cumulative.drop("Liquidity Pool", errors='ignore').sum()
                initial_circulating_supply = circulating_sum.iloc[0] if len(circulating_sum) > 0 else 0
            else:
                total_sum = self.data.vesting_cumulative.sum()
                initial_circulating_supply = total_sum.iloc[0] if len(total_sum) > 0 else 0
        
        # Get initial liquidity pool tokens
        initial_lp_tokens = 0
        if not self.data.vesting_cumulative.empty and "Liquidity Pool" in self.data.vesting_cumulative.index:
            lp_row = self.data.vesting_cumulative.loc["Liquidity Pool"]
            initial_lp_tokens = lp_row.iloc[0] if len(lp_row) > 0 else 0
        
        # Get initial token price
        initial_token_price = params.get("token_price", self.data.token_price)
        if isinstance(initial_token_price, list):
            initial_token_price = initial_token_price[0] if initial_token_price else 0.03
        
        # Get initial staking APR
        initial_staking_apr = 0.05  # Default value
        if not self.data.staking_apr.empty and "Staking APR" in self.data.staking_apr.index:
            apr_row = self.data.staking_apr.loc["Staking APR"]
            initial_staking_apr = apr_row.iloc[0] if len(apr_row) > 0 else 0.05
        
        # Set up the enhanced initial state
        self.initial_state = {
            "token_supply": initial_total_supply,
            "circulating_supply": initial_circulating_supply,
            "staked_tokens": 0,  # Start with 0 staked tokens
            "effective_circulating_supply": initial_circulating_supply,
            "liquidity_pool_tokens": initial_lp_tokens,
            "token_price": initial_token_price,
            "market_cap": initial_circulating_supply * initial_token_price,
            "staking_apr": initial_staking_apr,
            "time_step": 0,
            # Enhanced state variables for dynamic policies
            "total_points_issued": 0,
            "total_tokens_converted": 0,
            "total_tokens_burned": 0,
            "staking_rewards": 0
        }
        
        # Add date if available
        if hasattr(self.data, "dates") and self.data.dates is not None and len(self.data.dates) > 0:
            self.initial_state["date"] = self.data.dates[0]
        
        return self.initial_state
    
    def _discover_and_setup_policies(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Discover and setup dynamic policies based on available parameters.
        
        Args:
            params: Dictionary of simulation parameters
            
        Returns:
            Tuple of (policies, state_updates) dictionaries
        """
        # Update parameter registry with current simulation parameters
        self._update_parameter_registry(params)
        
        # Get dynamic policies from factory
        self.dynamic_policies = self.policy_factory.discover_and_create_policies(self.data)
        
        # Get corresponding state updates - only for state variables that exist
        available_state_vars = [
            'circulating_supply', 'staked_tokens', 'effective_circulating_supply', 
            'token_price', 'market_cap', 'time_step', 'total_points_issued', 
            'total_tokens_converted', 'total_tokens_burned', 'staking_rewards'
        ]
        
        all_state_updates = self.policy_factory.create_state_updates()
        state_updates = {k: v for k, v in all_state_updates.items() if k in available_state_vars}
        
        # Track which parameters are consumed
        self.consumed_params = self._track_consumed_parameters()
        
        return self.dynamic_policies, state_updates
    
    def _update_parameter_registry(self, params: Dict[str, Any]) -> None:
        """Update parameter registry with current simulation parameters."""
        # Make sure simulation parameters are in the registry
        for param_name, param_value in params.items():
            if param_name not in parameter_registry.parameters:
                # Add missing parameters to registry
                parameter_registry.register_parameters_from_csv({param_name: param_value})
    
    def _track_consumed_parameters(self) -> List[str]:
        """Track which parameters are actually consumed by policies."""
        consumed = []
        
        # Add base parameters that are always consumed
        base_consumed = [
            "initial_total_supply", "Initial Total Supply of Tokens",
            "token_price", "staking_share", "market_volatility"
        ]
        consumed.extend([p for p in base_consumed if p in self.data.static_params])
        
        # Add parameters consumed by discovered policies
        from logic.parameter_registry import ParameterCategory
        
        if "points_campaign" in self.dynamic_policies:
            points_params = parameter_registry.get_parameters_by_category(ParameterCategory.POINTS_CAMPAIGN)
            consumed.extend(points_params.keys())
        
        if "custom_utility" in self.dynamic_policies:
            utility_params = parameter_registry.get_parameters_by_category(ParameterCategory.UTILITY)
            consumed.extend(utility_params.keys())
        
        return consumed
    
    def run_simulation(self, params: Dict[str, Any], num_runs: int = 1,
                      simulation_type: str = "stochastic") -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Run a simulation with the given parameters using dynamic policy system.
        
        Args:
            params: Dictionary of simulation parameters
            num_runs: Number of simulation runs (1 = single run, >1 = Monte Carlo analysis)
            simulation_type: Type of simulation to run ("stochastic" or "agent")
            
        Returns:
            If num_runs=1: DataFrame with simulation results
            If num_runs>1: Dictionary with raw data and statistical measures (mean, std_dev, conf_intervals, percentiles)
        """
        if simulation_type == "agent":
            return self.run_agent_simulation(params, num_runs)
        else:
            return self.run_stochastic_simulation(params, num_runs)
    
    def run_stochastic_simulation(self, params: Dict[str, Any], num_runs: int = 1) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Run a stochastic simulation with the given parameters.
        
        Args:
            params: Dictionary of simulation parameters
            num_runs: Number of simulation runs (1 = single run, >1 = Monte Carlo analysis)
            
        Returns:
            If num_runs=1: DataFrame with simulation results
            If num_runs>1: Dictionary with raw data and statistical measures
        """
        try:
            # Input validation
            if not params:
                raise ValueError("Parameters dictionary cannot be empty")
            
            # Limit number of runs to prevent memory issues
            safe_num_runs = min(num_runs, 200)  # Cap at 200 runs to avoid memory issues
            if safe_num_runs < num_runs:
                st.warning(f"Number of runs limited to {safe_num_runs} to prevent memory issues")
                
            # Validate and clean parameters
            cleaned_params = self._validate_and_clean_parameters(params)
            
            # Set up initial state
            initial_state = self.setup_initial_state(cleaned_params)
            
            # Check if we have valid initial state
            if not initial_state:
                raise ValueError("Failed to create valid initial state from parameters")
            
            # Discover and setup policies based on parameters
            policies, state_updates = self._discover_and_setup_policies(cleaned_params)
            
            if not policies:
                st.warning("No valid policies could be created from the provided parameters")
                
            # Track consumed parameters for validation
            self.consumed_params = self._track_consumed_parameters()
            
            # Define state update blocks with dynamic policies
            state_update_blocks = [
                {
                    'policies': policies,
                    'variables': state_updates
                }
            ]
            
            # Create radCAD model with proper workflow
            model = Model(
                initial_state=initial_state,
                state_update_blocks=state_update_blocks,
                params=cleaned_params
            )
            
            # Create simulation
            simulation = Simulation(
                model=model,
                timesteps=self.timesteps,
                runs=safe_num_runs
            )
            
            # Create experiment
            experiment = Experiment(simulation)
            
            # Run experiment
            result = experiment.run()
            
            # Process results based on number of runs
            if safe_num_runs > 1:
                processed_result = self._process_monte_carlo_results(result, safe_num_runs)
            else:
                processed_result = self._process_single_run_result(result)
                
            # Force garbage collection
            del model, simulation, experiment, result
            gc.collect()
            
            return processed_result
                
        except Exception as e:
            st.error(f"Error in stochastic simulation: {str(e)}")
            # Return minimal result to prevent downstream errors
            if num_runs == 1:
                return pd.DataFrame({'timestep': [0], 'date': [None]})
            else:
                return {
                    'raw_data': pd.DataFrame({'timestep': [0], 'date': [None]}),
                    'mean': {},
                    'std_dev': {},
                    'conf_intervals': {},
                    'percentiles': {}
                }
            
        finally:
            # Validate parameter usage
            self._validate_parameter_coverage()
    
    def _validate_and_clean_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean parameters to ensure they have proper values.
        
        Args:
            params: Raw parameters dictionary
            
        Returns:
            Dictionary with validated and cleaned parameters
        """
        cleaned_params = {}
        
        for param_name, param_value in params.items():
            # Only set defaults for truly None/missing values
            if param_value is None:
                # Use specific defaults for critical parameters
                if "conversion_rate" in param_name.lower():
                    cleaned_params[param_name] = 0.5  # Default conversion rate
                elif "price" in param_name.lower() and "token" in param_name.lower():
                    cleaned_params[param_name] = 0.03  # Default token price
                elif param_name in ["staking_share", "staking_utility_share"]:
                    cleaned_params[param_name] = 0.75  # Default staking share
                elif param_name == "market_volatility":
                    cleaned_params[param_name] = 0.2  # Default volatility
                else:
                    # For other None parameters, set minimal defaults
                    cleaned_params[param_name] = 0
                    
                # Only warn about important missing parameters
                if "conversion_rate" in param_name.lower() or "price" in param_name.lower():
                    st.warning(f"⚠️ Parameter '{param_name}' was missing, using default value: {cleaned_params[param_name]}")
            else:
                # Ensure numeric parameters are proper numbers
                if isinstance(param_value, (int, float)):
                    # Check for NaN or infinite values
                    if pd.isna(param_value) or not np.isfinite(param_value):
                        cleaned_params[param_name] = 0
                        st.warning(f"⚠️ Parameter '{param_name}' had invalid value, using 0")
                    else:
                        cleaned_params[param_name] = float(param_value)
                else:
                    cleaned_params[param_name] = param_value
        
        # Only add truly missing critical parameters
        critical_defaults = {
            "staking_share": 0.75,
            "token_price": 0.03,
            "market_volatility": 0.2
        }
        
        for param_name, default_value in critical_defaults.items():
            if param_name not in cleaned_params:
                cleaned_params[param_name] = default_value
        
        return cleaned_params
    
    def _process_monte_carlo_results(self, result, num_runs: int) -> Dict[str, Any]:
        """Process Monte Carlo simulation results."""
        try:
            # Convert to DataFrame
            result_df = pd.DataFrame(result)
            
            # Safely add dates with error handling
            if hasattr(self.data, 'dates') and self.data.dates is not None and len(self.data.dates) > 0:
                try:
                    result_df['date'] = [
                        self.data.dates[i] if i < len(self.data.dates) else None 
                        for i in result_df['timestep']
                    ]
                except (IndexError, KeyError) as e:
                    # Log warning but continue without dates
                    st.warning(f"Could not add dates to simulation results: {str(e)}")
                    result_df['date'] = None
            else:
                # No dates available, set to None
                result_df['date'] = None
            
            # Create a dictionary to store the processed results
            processed_results = {
                'raw_data': result_df,
                'mean': {},
                'std_dev': {},
                'conf_intervals': {},
                'percentiles': {}
            }
            
            # Get unique timesteps - ensure they exist
            if 'timestep' not in result_df.columns:
                st.error("Missing 'timestep' column in simulation results")
                return processed_results
                
            timesteps = sorted(result_df['timestep'].unique())
            
            # Enhanced state variables to analyze (including new dynamic ones)
            state_vars = ['token_supply', 'circulating_supply', 'staked_tokens', 
                        'effective_circulating_supply', 'token_price', 'market_cap', 
                        'staking_apr', 'total_points_issued', 'total_tokens_converted', 
                        'total_tokens_burned']
            
            # For each state variable
            for var in state_vars:
                # Skip if variable not in results
                if var not in result_df.columns:
                    continue
                    
                # Initialize DataFrames for statistics
                processed_results['mean'][var] = pd.DataFrame(index=timesteps)
                processed_results['std_dev'][var] = pd.DataFrame(index=timesteps)
                processed_results['conf_intervals'][var] = pd.DataFrame(index=timesteps, columns=['lower', 'upper'])
                processed_results['percentiles'][var] = pd.DataFrame(index=timesteps, columns=[5, 25, 50, 75, 95])
                
                # Calculate statistics for each timestep
                for t in timesteps:
                    # Get data for this timestep across all runs
                    timestep_data = result_df[(result_df['timestep'] == t)][var]
                    
                    if len(timestep_data) == 0:
                        continue
                    
                    # Calculate mean and standard deviation
                    mean_val = timestep_data.mean()
                    std_val = timestep_data.std()
                    
                    # Calculate 95% confidence interval
                    conf_interval = 1.96 * std_val / np.sqrt(num_runs) if num_runs > 1 else std_val
                    
                    # Calculate percentiles safely
                    try:
                        percentiles = timestep_data.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
                    except Exception:
                        percentiles = np.array([mean_val] * 5)
                    
                    # Store results
                    processed_results['mean'][var].loc[t] = mean_val
                    processed_results['std_dev'][var].loc[t] = std_val
                    processed_results['conf_intervals'][var].loc[t] = [mean_val - conf_interval, mean_val + conf_interval]
                    processed_results['percentiles'][var].loc[t] = percentiles
            
            # Add dates to the statistical results safely
            if 'date' in result_df.columns and result_df['date'].notna().any():
                try:
                    for var in state_vars:
                        if var in processed_results['mean']:
                            date_index = []
                            valid_timesteps = []
                            
                            for i in timesteps:
                                if i < len(self.data.dates):
                                    date_index.append(self.data.dates[i])
                                    valid_timesteps.append(i)
                                else:
                                    # Skip timesteps beyond available dates
                                    break
                            
                            if len(date_index) > 0:
                                # Only update if we have valid dates
                                processed_results['mean'][var] = processed_results['mean'][var].loc[valid_timesteps]
                                processed_results['mean'][var].index = date_index
                                processed_results['std_dev'][var] = processed_results['std_dev'][var].loc[valid_timesteps]
                                processed_results['std_dev'][var].index = date_index
                                processed_results['conf_intervals'][var] = processed_results['conf_intervals'][var].loc[valid_timesteps]
                                processed_results['conf_intervals'][var].index = date_index
                                processed_results['percentiles'][var] = processed_results['percentiles'][var].loc[valid_timesteps]
                                processed_results['percentiles'][var].index = date_index
                except Exception as e:
                    st.warning(f"Could not add dates to statistical results: {str(e)}")
            
            # Force garbage collection to free memory
            gc.collect()
            
            return processed_results
            
        except Exception as e:
            st.error(f"Error processing Monte Carlo results: {str(e)}")
            return {
                'raw_data': pd.DataFrame(),
                'mean': {},
                'std_dev': {},
                'conf_intervals': {},
                'percentiles': {}
            }
    
    def _process_single_run_result(self, result) -> pd.DataFrame:
        """Process single run simulation result."""
        try:
            result_df = pd.DataFrame(result)
            
            # Debug: Log what columns we actually have
            if len(result_df.columns) > 0:
                st.info(f"Simulation generated columns: {list(result_df.columns)}")
            else:
                st.warning("Simulation result DataFrame is empty or has no columns")
            
            # Safely add dates with error handling
            if hasattr(self.data, 'dates') and self.data.dates is not None and len(self.data.dates) > 0:
                try:
                    result_df['date'] = [
                        self.data.dates[i] if i < len(self.data.dates) else None 
                        for i in result_df['timestep']
                    ]
                except (IndexError, KeyError) as e:
                    # Log warning but continue without dates
                    st.warning(f"Could not add dates to simulation results: {str(e)}")
                    result_df['date'] = None
            else:
                # No dates available, set to None
                result_df['date'] = None
            
            return result_df
            
        except Exception as e:
            st.error(f"Error processing single run result: {str(e)}")
            # Return a minimal DataFrame to prevent further errors
            return pd.DataFrame({'timestep': [0], 'date': [None]})
    
    def _validate_parameter_coverage(self):
        """Validate that parameters are being properly consumed."""
        if parameter_registry.parameters:
            unused, missing = parameter_registry.validate_parameter_coverage(self.consumed_params)
            
            if unused and len(unused) > 0:
                # Only show warning for significant unused parameters
                significant_unused = [p for p in unused if not any(skip in p.lower() for skip in ['comment', 'unit', 'step'])]
                if significant_unused:
                    st.warning(f"⚠️ Some parameters from your CSV are not being used: {', '.join(significant_unused[:3])}{'...' if len(significant_unused) > 3 else ''}")
    
    def run_agent_simulation(self, params: Dict[str, Any], num_runs: int = 1) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Run an agent-based simulation with the given parameters.
        
        Args:
            params: Dictionary of simulation parameters
            num_runs: Number of simulation runs (1 = single run, >1 = Monte Carlo analysis)
            
        Returns:
            If num_runs=1: DataFrame with simulation results
            If num_runs>1: Dictionary with raw data and statistical measures (mean, std_dev, conf_intervals, percentiles)
        """
        try:
            # Limit number of runs to prevent memory issues
            safe_num_runs = min(num_runs, 100)  # Cap at 100 runs to avoid memory issues
            if safe_num_runs < num_runs:
                st.warning(f"Number of runs limited to {safe_num_runs} to prevent memory issues")
                
            # Single run case for direct agent data access
            if safe_num_runs == 1:
                # Set up initial state
                initial_state = self.setup_initial_state(params)
                
                # Create agent based model
                abm = AgentBasedModel(initial_state, params)
                
                # Create agents
                agent_counts = {
                    "random_trader": params.get("random_trader_count", 20),
                    "trend_follower": params.get("trend_follower_count", 15),
                    "staking_agent": params.get("staking_agent_count", 10)
                }
                abm.create_agents(agent_counts)
                
                # Run simulation
                result_df = abm.run_simulation(self.timesteps)
                
                # Add agent data at the end
                agent_data = abm.get_agent_data()
                
                # Store agent data as an auxiliary result
                self.agent_data = agent_data
                
                return result_df
            
            # Multiple runs case for Monte Carlo simulation
            else:
                # Initialize list to store all runs
                all_runs = []
                
                for run in range(safe_num_runs):
                    # Set up initial state
                    initial_state = self.setup_initial_state(params)
                    
                    # Create agent based model
                    abm = AgentBasedModel(initial_state, params)
                    
                    # Create agents
                    agent_counts = {
                        "random_trader": params.get("random_trader_count", 20),
                        "trend_follower": params.get("trend_follower_count", 15),
                        "staking_agent": params.get("staking_agent_count", 10)
                    }
                    abm.create_agents(agent_counts)
                    
                    # Run simulation
                    run_result = abm.run_simulation(self.timesteps)
                    run_result['run'] = run
                    all_runs.append(run_result)
                
                # Combine all runs
                combined_df = pd.concat(all_runs, ignore_index=True)
                
                # Process similar to stochastic Monte Carlo
                return self._process_agent_monte_carlo_results(combined_df, safe_num_runs)
            
        except Exception as e:
            st.error(f"Error in agent-based simulation: {str(e)}")
            raise

    def _process_agent_monte_carlo_results(self, combined_df: pd.DataFrame, num_runs: int) -> Dict[str, Any]:
        """Process Monte Carlo results for agent-based simulations."""
        # Create a dictionary to store the processed results
        processed_results = {
            'raw_data': combined_df,
            'mean': {},
            'std_dev': {},
            'conf_intervals': {},
            'percentiles': {}
        }
        
        # Get unique timesteps
        timesteps = sorted(combined_df['timestep'].unique())
        
        # State variables to analyze for agent-based simulations
        state_vars = ['token_price', 'total_tokens', 'staked_tokens', 'market_cap', 'staking_apr', 'total_portfolio_value']
        
        # For each state variable
        for var in state_vars:
            # Skip if variable not in results
            if var not in combined_df.columns:
                continue
                
            # Initialize DataFrames for statistics
            processed_results['mean'][var] = pd.DataFrame(index=timesteps)
            processed_results['std_dev'][var] = pd.DataFrame(index=timesteps)
            processed_results['conf_intervals'][var] = pd.DataFrame(index=timesteps, columns=['lower', 'upper'])
            processed_results['percentiles'][var] = pd.DataFrame(index=timesteps, columns=[5, 25, 50, 75, 95])
            
            # Calculate statistics for each timestep
            for t in timesteps:
                # Get data for this timestep across all runs
                timestep_data = combined_df[(combined_df['timestep'] == t)][var]
                
                if len(timestep_data) == 0:
                    continue
                
                # Calculate mean and standard deviation
                mean_val = timestep_data.mean()
                std_val = timestep_data.std()
                
                # Calculate 95% confidence interval
                conf_interval = 1.96 * std_val / np.sqrt(num_runs) if num_runs > 1 else std_val
                
                # Calculate percentiles
                try:
                    percentiles = timestep_data.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
                except Exception:
                    percentiles = np.array([mean_val] * 5)
                
                # Store results
                processed_results['mean'][var].loc[t] = mean_val
                processed_results['std_dev'][var].loc[t] = std_val
                processed_results['conf_intervals'][var].loc[t] = [mean_val - conf_interval, mean_val + conf_interval]
                processed_results['percentiles'][var].loc[t] = percentiles
        
        # Add dates to the statistical results
        for var in state_vars:
            if var in processed_results['mean']:
                date_index = [self.data.dates[i] if i < len(self.data.dates) else None for i in timesteps]
                processed_results['mean'][var].index = date_index
                processed_results['std_dev'][var].index = date_index
                processed_results['conf_intervals'][var].index = date_index
                processed_results['percentiles'][var].index = date_index
        
        # Force garbage collection to free memory
        gc.collect()
        
        return processed_results

    def run_scenario_comparison(self, scenarios: List[Dict[str, Any]], num_runs: int = 1,
                              simulation_type: str = "stochastic") -> Dict[str, Any]:
        """
        Run multiple simulations for scenario comparison.
        
        Args:
            scenarios: List of dictionaries, each containing a scenario name and parameters
            num_runs: Number of Monte Carlo runs for each scenario (default: 1)
            simulation_type: Type of simulation to run ("stochastic" or "agent")
            
        Returns:
            Dictionary mapping scenario names to simulation results
        """
        results = {}
        
        for scenario in scenarios:
            name = scenario.get("name", f"Scenario {len(results) + 1}")
            params = scenario.get("params", {})
            
            # Run simulation for this scenario with specified number of runs
            result = self.run_simulation(params, num_runs=num_runs, simulation_type=simulation_type)
            
            # Store results
            results[name] = result
        
        return results
    
    def create_monte_carlo_visualization(self, mc_results: Dict[str, Any], variable: str, title: str = None) -> Dict[str, Any]:
        """
        Create visualization data for Monte Carlo simulation results.
        
        Args:
            mc_results: Results from a Monte Carlo simulation run
            variable: The state variable to visualize (e.g., 'token_price', 'market_cap')
            title: Optional title for the visualization
            
        Returns:
            Dictionary with data for visualization:
                - x_values: Dates or timesteps
                - mean: Mean values
                - lower_ci: Lower confidence interval
                - upper_ci: Upper confidence interval
                - percentiles: Percentile values (5th, 25th, 50th, 75th, 95th)
        """
        if not title:
            title = f"{variable.replace('_', ' ').title()} Monte Carlo Simulation"
            
        # Get the data
        mean_values = mc_results['mean'][variable].iloc[:, 0].values
        conf_intervals = mc_results['conf_intervals'][variable].values
        lower_ci = [ci[0] for ci in conf_intervals]
        upper_ci = [ci[1] for ci in conf_intervals]
        
        # Get percentiles
        percentiles = mc_results['percentiles'][variable].values
        
        # Get x values (dates or timesteps)
        x_values = mc_results['mean'][variable].index
        
        # Create visualization data
        viz_data = {
            'title': title,
            'x_values': x_values,
            'mean': mean_values,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'percentiles': percentiles
        }
        
        return viz_data
    
    def get_distribution_at_timestep(self, mc_results: Dict[str, Any], variable: str, timestep: int) -> Dict[str, Any]:
        """
        Get probability distribution data for a specific variable at a specific timestep.
        
        Args:
            mc_results: Results from a Monte Carlo simulation run
            variable: The state variable to analyze (e.g., 'token_price', 'market_cap')
            timestep: The timestep to analyze
            
        Returns:
            Dictionary with distribution data:
                - values: All values from different runs
                - mean: Mean value
                - median: Median value
                - std_dev: Standard deviation
                - percentiles: Key percentiles (5th, 25th, 50th, 75th, 95th)
        """
        # Get all values for this variable at this timestep across all runs
        raw_data = mc_results['raw_data']
        values = raw_data[(raw_data['timestep'] == timestep)][variable].values
        
        # Calculate statistics
        mean = np.mean(values)
        median = np.median(values)
        std_dev = np.std(values)
        percentiles = np.percentile(values, [5, 25, 50, 75, 95])
        
        # Create distribution data
        dist_data = {
            'values': values,
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'percentiles': percentiles
        }
        
        return dist_data