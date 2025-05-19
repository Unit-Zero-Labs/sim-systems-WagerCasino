"""
State management for the Unit Zero Labs Tokenomics Engine.
Provides utilities for managing Streamlit session state.
"""

import streamlit as st
from typing import Any, Dict, Optional, List, Callable
import pandas as pd

from tokenomics_data import TokenomicsData
from utils.config import get_simulation_defaults, get_monte_carlo_defaults


class StateManager:
    """
    State manager class that provides utilities for managing Streamlit session state.
    """
    
    @staticmethod
    def initialize_state() -> None:
        """
        Initialize the session state with default values if they don't exist.
        Should be called at the start of the application.
        """
        # Data state
        if 'tokenomics_data_obj' not in st.session_state:
            st.session_state.tokenomics_data_obj = None
        
        if 'radcad_uploaded_file_name' not in st.session_state:
            st.session_state.radcad_uploaded_file_name = None
        
        # Simulation parameters state
        sim_defaults = get_simulation_defaults()
        if 'simulation_params' not in st.session_state:
            st.session_state.simulation_params = sim_defaults
        
        # Monte Carlo parameters state
        mc_defaults = get_monte_carlo_defaults()
        if 'enable_monte_carlo' not in st.session_state:
            st.session_state.enable_monte_carlo = False
        
        if 'monte_carlo_params' not in st.session_state:
            st.session_state.monte_carlo_params = {
                'num_runs': mc_defaults['num_runs'],
                'show_confidence_intervals': mc_defaults['show_confidence_intervals'],
                'show_percentiles': mc_defaults['show_percentiles']
            }
        
        # Simulation results state
        if 'sim_result' not in st.session_state:
            st.session_state.sim_result = None
        
        if 'sim_result_displayed' not in st.session_state:
            st.session_state.sim_result_displayed = False
    
    @staticmethod
    def get_data() -> Optional[TokenomicsData]:
        """
        Get the tokenomics data object from session state.
        
        Returns:
            TokenomicsData object or None if not initialized
        """
        return st.session_state.get('tokenomics_data_obj')
    
    @staticmethod
    def set_data(data: TokenomicsData, file_name: str = None) -> None:
        """
        Set the tokenomics data object in session state.
        
        Args:
            data: TokenomicsData object
            file_name: Name of the uploaded file
        """
        st.session_state.tokenomics_data_obj = data
        if file_name:
            st.session_state.radcad_uploaded_file_name = file_name
    
    @staticmethod
    def get_uploaded_file_name() -> Optional[str]:
        """
        Get the name of the last uploaded file.
        
        Returns:
            String with file name or None
        """
        return st.session_state.get('radcad_uploaded_file_name')
    
    @staticmethod
    def clear_data() -> None:
        """
        Clear the tokenomics data from session state.
        """
        st.session_state.tokenomics_data_obj = None
        st.session_state.radcad_uploaded_file_name = None
    
    @staticmethod
    def get_simulation_params() -> Dict[str, Any]:
        """
        Get the simulation parameters from session state.
        
        Returns:
            Dictionary of simulation parameters
        """
        return st.session_state.get('simulation_params', get_simulation_defaults())
    
    @staticmethod
    def set_simulation_params(params: Dict[str, Any]) -> None:
        """
        Set the simulation parameters in session state.
        
        Args:
            params: Dictionary of simulation parameters
        """
        st.session_state.simulation_params = params
    
    @staticmethod
    def get_monte_carlo_enabled() -> bool:
        """
        Check if Monte Carlo simulation is enabled.
        
        Returns:
            Boolean indicating if Monte Carlo is enabled
        """
        return st.session_state.get('enable_monte_carlo', False)
    
    @staticmethod
    def set_monte_carlo_enabled(enabled: bool) -> None:
        """
        Set the Monte Carlo simulation enabled flag.
        
        Args:
            enabled: Boolean indicating if Monte Carlo should be enabled
        """
        st.session_state.enable_monte_carlo = enabled
    
    @staticmethod
    def get_monte_carlo_params() -> Dict[str, Any]:
        """
        Get the Monte Carlo parameters from session state.
        
        Returns:
            Dictionary of Monte Carlo parameters
        """
        return st.session_state.get('monte_carlo_params', {
            'num_runs': get_monte_carlo_defaults()['num_runs'],
            'show_confidence_intervals': get_monte_carlo_defaults()['show_confidence_intervals'],
            'show_percentiles': get_monte_carlo_defaults()['show_percentiles']
        })
    
    @staticmethod
    def set_monte_carlo_params(params: Dict[str, Any]) -> None:
        """
        Set the Monte Carlo parameters in session state.
        
        Args:
            params: Dictionary of Monte Carlo parameters
        """
        st.session_state.monte_carlo_params = params
    
    @staticmethod
    def get_simulation_result() -> Any:
        """
        Get the simulation result from session state.
        
        Returns:
            Simulation result (DataFrame or Dict)
        """
        return st.session_state.get('sim_result')
    
    @staticmethod
    def set_simulation_result(result: Any) -> None:
        """
        Set the simulation result in session state.
        
        Args:
            result: Simulation result (DataFrame or Dict)
        """
        st.session_state.sim_result = result
        st.session_state.sim_result_displayed = True
    
    @staticmethod
    def is_simulation_result_displayed() -> bool:
        """
        Check if simulation result has been displayed.
        
        Returns:
            Boolean indicating if simulation result has been displayed
        """
        return st.session_state.get('sim_result_displayed', False)
    
    @staticmethod
    def update_state_from_ui(key: str, value: Any) -> None:
        """
        Update a specific session state value from UI.
        
        Args:
            key: Session state key
            value: New value
        """
        st.session_state[key] = value
    
    @staticmethod
    def callback_wrapper(callback_fn: Callable, *args, **kwargs) -> Callable:
        """
        Create a wrapper for a callback function to ensure state is updated.
        
        Args:
            callback_fn: The callback function to wrap
            *args, **kwargs: Arguments to pass to the callback
            
        Returns:
            Wrapped callback function
        """
        def wrapped_callback():
            result = callback_fn(*args, **kwargs)
            # Trigger a rerun to ensure UI is updated
            st.experimental_rerun()
            return result
        return wrapped_callback