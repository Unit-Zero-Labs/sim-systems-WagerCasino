"""
State management for the Streamlit application.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Callable, List
from tokenomics_data import TokenomicsData
from utils.config import get_simulation_defaults, get_agent_defaults


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
        
        # Agent parameters state
        agent_defaults = get_agent_defaults()
        if 'agent_params' not in st.session_state:
            st.session_state.agent_params = agent_defaults
        
        # Simulation results state
        if 'sim_result' not in st.session_state:
            st.session_state.sim_result = None
        
        if 'sim_result_displayed' not in st.session_state:
            st.session_state.sim_result_displayed = False
        
        # Simulation type state (stochastic or agent)
        if 'simulation_type' not in st.session_state:
            st.session_state.simulation_type = "stochastic"
        
        # Agent-based simulation state
        if 'agent_data' not in st.session_state:
            st.session_state.agent_data = None
        
        # Monte Carlo simulator instance
        if 'simulator' not in st.session_state:
            st.session_state.simulator = None
    
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
    def get_agent_params() -> Dict[str, Any]:
        """
        Get the agent-based simulation parameters from session state.
        
        Returns:
            Dictionary of agent parameters
        """
        return st.session_state.get('agent_params', get_agent_defaults())
    
    @staticmethod
    def set_agent_params(params: Dict[str, Any]) -> None:
        """
        Set the agent parameters in session state.
        
        Args:
            params: Dictionary of agent parameters
        """
        st.session_state.agent_params = params
    
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
    def get_simulation_type() -> str:
        """
        Get the simulation type from session state.
        
        Returns:
            String indicating the simulation type ("stochastic" or "agent")
        """
        return st.session_state.get('simulation_type', "stochastic")
    
    @staticmethod
    def set_simulation_type(sim_type: str) -> None:
        """
        Set the simulation type in session state.
        
        Args:
            sim_type: String indicating the simulation type ("stochastic" or "agent")
        """
        st.session_state.simulation_type = sim_type
    
    @staticmethod
    def get_agent_data() -> pd.DataFrame:
        """
        Get the agent data from session state.
        
        Returns:
            DataFrame with agent data or None
        """
        return st.session_state.get('agent_data')
    
    @staticmethod
    def set_agent_data(agent_data: pd.DataFrame) -> None:
        """
        Set the agent data in session state.
        
        Args:
            agent_data: DataFrame with agent data
        """
        st.session_state.agent_data = agent_data
    
    @staticmethod
    def get_simulator() -> Any:
        """
        Get the simulator instance from session state.
        
        Returns:
            Simulator instance or None
        """
        return st.session_state.get('simulator')
    
    @staticmethod
    def set_simulator(simulator: Any) -> None:
        """
        Set the Monte Carlo simulator instance in session state.
        
        Args:
            simulator: MonteCarloSimulator instance
        """
        st.session_state.simulator = simulator
    
    @staticmethod
    def clear_simulation_results() -> None:
        """
        Clear all simulation-related state, making it safe to run a new simulation.
        Should be called before starting a new simulation or when resetting the app.
        """
        # Clear simulation results
        st.session_state.sim_result = None
        st.session_state.sim_result_displayed = False
        
        # Clear simulator instance
        st.session_state.simulator = None
        
        # Clear agent data
        st.session_state.agent_data = None
        
        # Reset simulation type to default
        st.session_state.simulation_type = "stochastic"
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
    
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