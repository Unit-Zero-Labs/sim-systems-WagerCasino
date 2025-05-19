"""
Monte Carlo simulation module for the Unit Zero Labs Tokenomics Engine.
Provides optimized implementation of Monte Carlo simulations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
import concurrent.futures
import streamlit as st
from datetime import datetime
import multiprocessing

from utils.error_handler import ErrorHandler


class MonteCarloSimulator:
    """
    Monte Carlo simulator for tokenomics simulations.
    Optimizes performance through parallelization.
    """
    
    def __init__(self, simulation_engine):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            simulation_engine: A TokenomicsSimulation instance
        """
        self.simulation_engine = simulation_engine
        self.max_workers = min(multiprocessing.cpu_count(), 4)  # Use at most 4 cores
    
    def run_parallel_simulations(
        self, 
        params: Dict[str, Any], 
        num_runs: int, 
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations in parallel.
        
        Args:
            params: Simulation parameters
            num_runs: Number of simulation runs
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with Monte Carlo simulation results
        """
        start_time = time.time()
        
        # Create a list of simulation settings with different random seeds
        simulation_settings = [
            {"params": params, "run_num": i, "seed": np.random.randint(0, 10000)}
            for i in range(num_runs)
        ]
        
        # Create empty results object
        raw_results = []
        processed_results = None
        
        try:
            # Use a ProcessPoolExecutor for parallel processing
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all simulation runs
                future_to_run = {
                    executor.submit(self._run_single_simulation, setting): i 
                    for i, setting in enumerate(simulation_settings)
                }
                
                # Process results as they complete
                completed = 0
                for future in concurrent.futures.as_completed(future_to_run):
                    run_idx = future_to_run[future]
                    try:
                        result = future.result()
                        raw_results.append(result)
                    except Exception as e:
                        ErrorHandler.log_error(e, f"Monte Carlo run {run_idx}")
                    
                    # Update progress
                    completed += 1
                    if progress_callback:
                        progress_callback(completed / num_runs)
            
            # Process the raw results
            processed_results = self._process_simulation_results(raw_results)
            
        except Exception as e:
            ErrorHandler.log_error(e, "Monte Carlo simulation")
            raise
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if processed_results:
            processed_results['execution_info'] = {
                'num_runs': num_runs,
                'execution_time': execution_time,
                'runs_per_second': num_runs / execution_time,
                'parallel_workers': self.max_workers
            }
        
        return processed_results
    
    def _run_single_simulation(self, settings: Dict[str, Any]) -> pd.DataFrame:
        """
        Run a single simulation with the given settings.
        
        Args:
            settings: Dictionary containing simulation parameters and run info
            
        Returns:
            DataFrame with simulation results
        """
        params = settings["params"]
        run_num = settings["run_num"]
        seed = settings["seed"]
        
        # Set random seed for this run
        np.random.seed(seed)
        
        # Run the simulation
        result = self.simulation_engine.run_simulation(params, num_runs=1)
        
        # Add run information to the result
        result['run'] = run_num
        result['seed'] = seed
        
        return result
    
    def _process_simulation_results(self, raw_results: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Process raw simulation results into statistical measures.
        
        Args:
            raw_results: List of DataFrames from individual simulation runs
            
        Returns:
            Dictionary with processed Monte Carlo results
        """
        if not raw_results:
            st.error("No simulation results to process. Please check for errors in the simulation.")
            return None
            
        # Combine all raw results
        raw_data = pd.concat(raw_results, ignore_index=True)
        
        # Get unique timesteps
        timesteps = sorted(raw_data['timestep'].unique())
        
        # State variables to analyze
        state_vars = ['token_supply', 'circulating_supply', 'staked_tokens', 
                      'token_price', 'market_cap', 'staking_apr']
        
        # Initialize result containers
        processed_results = {
            'raw_data': raw_data,
            'mean': {},
            'std_dev': {},
            'conf_intervals': {},
            'percentiles': {}
        }
        
        # For each state variable
        for var in state_vars:
            # Make sure the variable exists in the raw data
            if var not in raw_data.columns:
                continue
                
            # Initialize DataFrames for statistics
            processed_results['mean'][var] = pd.DataFrame(index=timesteps)
            processed_results['std_dev'][var] = pd.DataFrame(index=timesteps)
            processed_results['conf_intervals'][var] = pd.DataFrame(index=timesteps, columns=['lower', 'upper'])
            processed_results['percentiles'][var] = pd.DataFrame(index=timesteps, columns=[5, 25, 50, 75, 95])
            
            # Calculate statistics for each timestep
            for t in timesteps:
                # Get data for this timestep across all runs
                timestep_data = raw_data[(raw_data['timestep'] == t)][var]
                
                if len(timestep_data) == 0:
                    # Skip if no data for this timestep
                    continue
                
                # Calculate mean and standard deviation
                mean_val = timestep_data.mean()
                std_val = timestep_data.std()
                
                # Calculate 95% confidence interval
                conf_interval = 1.96 * std_val / np.sqrt(len(raw_results))  # 95% CI
                
                # Calculate percentiles
                try:
                    percentiles = timestep_data.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
                except Exception as e:
                    percentiles = np.array([mean_val] * 5)  # Fallback to mean value
                
                # Store results
                processed_results['mean'][var].loc[t] = mean_val
                processed_results['std_dev'][var].loc[t] = std_val
                processed_results['conf_intervals'][var].loc[t] = [mean_val - conf_interval, mean_val + conf_interval]
                processed_results['percentiles'][var].loc[t] = percentiles
        
        # Add dates to the statistical results
        if 'date' in raw_data.columns:
            dates_by_timestep = {}
            for t in timesteps:
                # Make sure there's data for this timestep
                timestep_rows = raw_data[raw_data['timestep'] == t]
                if len(timestep_rows) > 0:
                    # Get the date for this timestep (same across all runs)
                    date = timestep_rows['date'].iloc[0]
                    dates_by_timestep[t] = date
                else:
                    dates_by_timestep[t] = None
            
            # Replace timestep indices with dates
            for var in state_vars:
                if var in processed_results['mean']:  # Only process variables that were found in the data
                    date_index = [dates_by_timestep[t] for t in timesteps if t in dates_by_timestep]
                    if len(date_index) > 0:
                        valid_timestamps = [t for t in timesteps if t in dates_by_timestep]
                        processed_results['mean'][var] = processed_results['mean'][var].loc[valid_timestamps]
                        processed_results['mean'][var].index = date_index
                        processed_results['std_dev'][var] = processed_results['std_dev'][var].loc[valid_timestamps]
                        processed_results['std_dev'][var].index = date_index
                        processed_results['conf_intervals'][var] = processed_results['conf_intervals'][var].loc[valid_timestamps]
                        processed_results['conf_intervals'][var].index = date_index
                        processed_results['percentiles'][var] = processed_results['percentiles'][var].loc[valid_timestamps]
                        processed_results['percentiles'][var].index = date_index
        
        return processed_results
    
    @staticmethod
    def get_distribution_at_timestep(
        mc_results: Dict[str, Any], 
        variable: str, 
        timestep: int
    ) -> Dict[str, Any]:
        """
        Get probability distribution data for a specific variable at a specific timestep.
        
        Args:
            mc_results: Results from a Monte Carlo simulation run
            variable: The state variable to analyze (e.g., 'token_price', 'market_cap')
            timestep: The timestep to analyze
            
        Returns:
            Dictionary with distribution data
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
            'percentiles': percentiles,
            'cv': std_dev / mean if mean != 0 else float('inf')  # Coefficient of variation
        }
        
        return dist_data
    
    @staticmethod
    def run_monte_carlo_with_progress(
        simulator, 
        params: Dict[str, Any], 
        num_runs: int
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with a progress bar.
        
        Args:
            simulator: MonteCarloSimulator instance
            params: Simulation parameters
            num_runs: Number of simulation runs
            
        Returns:
            Dictionary with Monte Carlo simulation results
        """
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Define progress callback
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Running Monte Carlo simulation: {int(progress * 100)}% complete")
        
        try:
            # Run the simulations
            results = simulator.run_parallel_simulations(params, num_runs, update_progress)
            
            # Complete the progress bar
            progress_bar.progress(1.0)
            status_text.text("Monte Carlo simulation complete!")
            
            return results
        except Exception as e:
            # Show error
            progress_bar.progress(1.0)
            status_text.text(f"Error in Monte Carlo simulation: {str(e)}")
            raise
        finally:
            # Small delay to show completion
            time.sleep(0.5)
            
            # Clear the progress elements
            progress_bar.empty()
            status_text.empty()