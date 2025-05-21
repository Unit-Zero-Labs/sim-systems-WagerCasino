"""
Monte Carlo simulation module for the Unit Zero Labs Tokenomics Engine.
Provides a simplified, robust implementation of Monte Carlo simulations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import concurrent.futures
import streamlit as st
from datetime import datetime
import multiprocessing
import math
import gc

from utils.error_handler import ErrorHandler


class MonteCarloSimulator:
    """
    Monte Carlo simulator for tokenomics simulations.
    Provides an easy-to-use interface for running stochastic simulations.
    """
    
    def __init__(self, simulation_engine):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            simulation_engine: A TokenomicsSimulation instance
        """
        self.simulation_engine = simulation_engine
        # Limit workers based on available CPU cores but cap at 4
        self.max_workers = min(multiprocessing.cpu_count(), 4)  
        self.results = None  # Store the most recent simulation results
        # Maximum runs per batch to avoid memory issues
        self.max_batch_size = 20
    
    def run_simulations(
        self, 
        params: Dict[str, Any], 
        num_runs: int = 50,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations with the given parameters.
        Simplified interface that handles progress display internally.
        
        Args:
            params: Simulation parameters
            num_runs: Number of simulation runs (default: 50)
            show_progress: Whether to show a progress bar (default: True)
            
        Returns:
            Dictionary with Monte Carlo simulation results
        """
        if show_progress:
            # Use the progress bar method
            self.results = self.run_simulations_with_progress(params, num_runs)
        else:
            # Run without progress bar
            self.results = self._run_parallel_simulations(params, num_runs)
            
        return self.results
    
    def run_simulations_with_progress(
        self, 
        params: Dict[str, Any], 
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations with a progress bar.
        
        Args:
            params: Simulation parameters
            num_runs: Number of simulation runs (default: 50)
            
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
            self.results = self._run_parallel_simulations(params, num_runs, update_progress)
            
            # Complete the progress bar
            progress_bar.progress(1.0)
            status_text.text("Monte Carlo simulation complete!")
            
            return self.results
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
    
    def _run_parallel_simulations(
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
        
        # Create empty results list
        raw_results = []
        processed_results = None
        
        # Create batch processing for larger run counts
        batch_size = min(self.max_batch_size, num_runs)
        num_batches = math.ceil(num_runs / batch_size)
        
        # Keep track of completion
        completed = 0
        
        try:
            # Process each batch
            for batch_idx in range(num_batches):
                # Calculate start and end indices for this batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_runs)
                batch_runs = end_idx - start_idx
                
                # Create a list of simulation settings for this batch with different random seeds
                simulation_settings = [
                    {"params": params, "run_num": i + start_idx, "seed": np.random.randint(0, 100000)}
                    for i in range(batch_runs)
                ]
                
                # Use a ThreadPoolExecutor for parallelization
                # ThreadPoolExecutor shares memory space, so it should be more efficient
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all simulation runs for this batch
                    future_to_run = {
                        executor.submit(self._run_single_simulation, setting): i 
                        for i, setting in enumerate(simulation_settings)
                    }
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_run):
                        run_idx = future_to_run[future]
                        try:
                            result = future.result()
                            if result is not None:
                                raw_results.append(result)
                        except Exception as e:
                            ErrorHandler.log_error(e, f"Monte Carlo run {run_idx + start_idx}")
                        
                        # Update progress
                        completed += 1
                        if progress_callback:
                            progress_callback(completed / num_runs)
                
                # Force garbage collection after each batch
                executor.shutdown(wait=True)
                gc.collect()
            
            # Process the raw results
            if raw_results:
                processed_results = self._process_simulation_results(raw_results)
            else:
                raise ValueError("No successful simulation runs were completed")
            
        except Exception as e:
            ErrorHandler.log_error(e, "Monte Carlo simulation")
            raise
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if processed_results:
            processed_results['execution_info'] = {
                'num_runs': len(raw_results),  # Use actual completed runs
                'execution_time': execution_time,
                'runs_per_second': len(raw_results) / execution_time if execution_time > 0 else 0,
                'parallel_workers': self.max_workers
            }
        
        return processed_results
    
    def _run_single_simulation(self, settings: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Run a single simulation with the given settings.
        
        Args:
            settings: Dictionary containing simulation parameters and run info
            
        Returns:
            DataFrame with simulation results or None if an error occurred
        """
        params = settings["params"]
        run_num = settings["run_num"]
        seed = settings["seed"]
        
        try:
            # Set random seed for this run
            np.random.seed(seed)
            
            # Run the simulation
            result = self.simulation_engine.run_simulation(params, num_runs=1)
            
            # Add run information to the result
            result['run'] = run_num
            result['seed'] = seed
            
            return result
        except Exception as e:
            ErrorHandler.log_error(e, f"Single simulation run {run_num}")
            return None
    
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
                # Use actual number of runs for this timestep, not the original list length
                n_runs = len(timestep_data)
                conf_interval = 1.96 * std_val / np.sqrt(n_runs) if n_runs > 1 else std_val
                
                # Calculate percentiles
                try:
                    percentiles = timestep_data.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
                except Exception as e:
                    # Fallback to mean value if percentile calculation fails
                    percentiles = np.array([mean_val] * 5)
                
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
    
    def get_distribution_at_timestep(
        self, 
        variable: str, 
        timestep: int
    ) -> Dict[str, Any]:
        """
        Get probability distribution data for a specific variable at a specific timestep.
        Uses the stored results from the last simulation run.
        
        Args:
            variable: The state variable to analyze (e.g., 'token_price', 'market_cap')
            timestep: The timestep to analyze
            
        Returns:
            Dictionary with distribution data
        """
        if self.results is None:
            raise ValueError("No simulation results available. Run a simulation first.")
            
        return self._get_distribution_data(self.results, variable, timestep)
    
    @staticmethod
    def _get_distribution_data(
        mc_results: Dict[str, Any], 
        variable: str, 
        timestep: int
    ) -> Dict[str, Any]:
        """
        Extract distribution data from Monte Carlo results.
        
        Args:
            mc_results: Results from a Monte Carlo simulation run
            variable: The state variable to analyze
            timestep: The timestep to analyze
            
        Returns:
            Dictionary with distribution data
        """
        # Get all values for this variable at this timestep across all runs
        raw_data = mc_results['raw_data']
        values = raw_data[(raw_data['timestep'] == timestep)][variable].values
        
        if len(values) == 0:
            raise ValueError(f"No data available for variable '{variable}' at timestep {timestep}")
        
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
    
    def get_time_series_data(self, variable: str) -> Dict[str, Any]:
        """
        Get time series data for a specific variable across all timesteps.
        Uses the stored results from the last simulation run.
        
        Args:
            variable: The state variable to extract (e.g., 'token_price', 'market_cap')
            
        Returns:
            Dictionary with time series data
        """
        if self.results is None:
            raise ValueError("No simulation results available. Run a simulation first.")
            
        if variable not in self.results['mean']:
            raise ValueError(f"Variable '{variable}' not found in simulation results")
            
        # Extract time series data
        time_series_data = {
            'x_values': self.results['mean'][variable].index,
            'mean': self.results['mean'][variable].values.flatten(),
            'lower_ci': [ci[0] for ci in self.results['conf_intervals'][variable].values],
            'upper_ci': [ci[1] for ci in self.results['conf_intervals'][variable].values],
            'percentiles': self.results['percentiles'][variable].values
        }
        
        return time_series_data