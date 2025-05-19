"""
UI components module for the Unit Zero Labs Tokenomics Engine.
Contains reusable UI components for the Streamlit application.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import pandas as pd
import plotly.graph_objects as go

from .charts import (
    plot_token_supply_simulation,
    plot_token_price_simulation,
    plot_market_cap_simulation,
    plot_staking_simulation,
    plot_monte_carlo_results,
    plot_distribution_at_timestep
)


def create_header(logo_path: Optional[str] = None, title: str = "Tokenomics Engine"):
    """
    Create a header with logo and title.
    
    Args:
        logo_path: Path to the logo image
        title: Title text to display
    """
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Add some top padding
    
    # Create a header container with columns for better control
    col1, col2, col3 = st.columns([4, 3, 4])
    
    with col1:
        # Left side with UZ logo and title
        cols_left = st.columns([1, 2])
        with cols_left[0]:
            try:
                if logo_path:
                    st.image(logo_path, width=120)
                else:
                    st.image("public/uz-logo.png", width=120)
            except:
                st.error("UZ logo not found. Please ensure 'uz-logo.png' is in the public directory.")
        
        with cols_left[1]:
            st.markdown(f"<h2 class='header-title'>{title}</h2>", unsafe_allow_html=True)
    
    with col3:
        # Right side with client logo
        try:
            st.image("public/client-logo.png", width=120)
        except:
            # Silently fail if client logo isn't found
            pass
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Add some bottom padding


def create_plot_container(plot_function: Callable, data: Any, *args, **kwargs) -> None:
    """
    Create a container for a plot with consistent styling.
    
    Args:
        plot_function: Function that returns a plotly figure
        data: Data to pass to the plot function
        *args, **kwargs: Additional arguments to pass to the plot function
    """
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig = plot_function(data, *args, **kwargs)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def display_simulation_parameter_controls() -> Dict[str, Any]:
    """
    Display controls for simulation parameters.
    
    Returns:
        Dictionary of parameter values
    """
    st.subheader("Simulation Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        staking_share = st.slider(
            "Staking Share", 
            0.0, 1.0, 
            0.75,  # Default value
            step=0.01
        )
        
        staking_apr_multiplier = st.slider(
            "Staking APR Multiplier", 
            0.5, 2.0, 
            1.0,  # Default value
            step=0.1
        )
    
    with col2:
        token_price = st.slider(
            "Token Launch Price", 
            0.01, 0.05, 
            0.03,  # Default value
            step=0.001,
            format="$%.5f"
        )
        
        market_volatility = st.slider(
            "Market Volatility", 
            0.0, 1.0, 
            0.2,  # Default value
            step=0.05
        )
    
    # Return parameters as a dictionary
    return {
        "staking_share": staking_share,
        "token_price": token_price,
        "staking_apr_multiplier": staking_apr_multiplier,
        "market_volatility": market_volatility
    }


def display_monte_carlo_controls() -> Tuple[bool, int, bool, bool]:
    """
    Display controls for Monte Carlo simulation.
    
    Returns:
        Tuple containing:
        - enable_monte_carlo: Whether to enable Monte Carlo simulation
        - num_runs: Number of simulation runs
        - show_confidence_intervals: Whether to show confidence intervals
        - show_percentiles: Whether to show percentile bands
    """
    st.subheader("Monte Carlo Simulation")
    enable_monte_carlo = st.checkbox("Enable Monte Carlo Simulation", value=False)
    
    if enable_monte_carlo:
        col1, col2 = st.columns(2)
        with col1:
            num_runs = st.slider("Number of Runs", 10, 100, 50, step=10)
            show_confidence_intervals = st.checkbox("Show Confidence Intervals", value=True)
        
        with col2:
            show_percentiles = st.checkbox("Show Percentile Bands", value=True)
    else:
        num_runs = 1
        show_confidence_intervals = True
        show_percentiles = True
    
    return enable_monte_carlo, num_runs, show_confidence_intervals, show_percentiles


def display_single_run_results(sim_result: pd.DataFrame) -> None:
    """
    Display the results of a single simulation run.
    
    Args:
        sim_result: DataFrame with simulation results
    """
    # Debug info
    st.info(f"Simulation result columns: {sim_result.columns.tolist()}")
    st.info(f"Initial values from simulation: token_supply={sim_result['token_supply'].iloc[0]:,.0f}, circulating_supply={sim_result['circulating_supply'].iloc[0]:,.0f}")
    
    # Create tabs for different metrics
    metric_tabs = st.tabs(["Token Supply", "Token Price", "Market Cap", "Staking"])
    
    with metric_tabs[0]:
        # Plot the token supply results
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig = plot_token_supply_simulation(sim_result)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display the simulation results
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Supply Results")
        
        # Make sure we display the token_supply from the simulation result
        st.write(f"Initial Total Supply: {sim_result['token_supply'].iloc[0]:,.0f} tokens")
        st.write(f"Initial Circulating Supply: {sim_result['circulating_supply'].iloc[0]:,.0f} tokens")
        st.write(f"Final Circulating Supply: {sim_result['circulating_supply'].iloc[-1]:,.0f} tokens")
        
        # Calculate change only if initial is not zero
        if sim_result['circulating_supply'].iloc[0] > 0:
            change_pct = (sim_result['circulating_supply'].iloc[-1] / sim_result['circulating_supply'].iloc[0] - 1) * 100
            st.write(f"Change: {(sim_result['circulating_supply'].iloc[-1] - sim_result['circulating_supply'].iloc[0]):,.0f} tokens ({change_pct:.2f}%)")
        else:
            st.write(f"Change: {(sim_result['circulating_supply'].iloc[-1] - sim_result['circulating_supply'].iloc[0]):,.0f} tokens (N/A%)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_tabs[1]:
        # Plot the token price results
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig = plot_token_price_simulation(sim_result)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display the simulation results
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Price Results")
        st.write(f"Initial Price: ${sim_result['token_price'].iloc[0]:.5f}")
        st.write(f"Final Price: ${sim_result['token_price'].iloc[-1]:.5f}")
        st.write(f"Change: ${(sim_result['token_price'].iloc[-1] - sim_result['token_price'].iloc[0]):.5f} ({(sim_result['token_price'].iloc[-1] / sim_result['token_price'].iloc[0] - 1) * 100:.2f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_tabs[2]:
        # Plot the market cap results
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig = plot_market_cap_simulation(sim_result)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display the simulation results
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Market Cap Results")
        st.write(f"Initial Market Cap: ${sim_result['market_cap'].iloc[0]:,.2f}")
        st.write(f"Final Market Cap: ${sim_result['market_cap'].iloc[-1]:,.2f}")
        st.write(f"Change: ${(sim_result['market_cap'].iloc[-1] - sim_result['market_cap'].iloc[0]):,.2f} ({(sim_result['market_cap'].iloc[-1] / sim_result['market_cap'].iloc[0] - 1) * 100:.2f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_tabs[3]:
        # Plot the staking results
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig = plot_staking_simulation(sim_result)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display the simulation results
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Staking Results")
        if len(sim_result) > 1:
            st.write(f"Initial Staked Tokens: {sim_result['staked_tokens'].iloc[0]:,.0f} tokens")
            st.write(f"Final Staked Tokens: {sim_result['staked_tokens'].iloc[-1]:,.0f} tokens")
            if sim_result['staked_tokens'].iloc[0] > 0:
                st.write(f"Change: {(sim_result['staked_tokens'].iloc[-1] - sim_result['staked_tokens'].iloc[0]):,.0f} tokens ({(sim_result['staked_tokens'].iloc[-1] / sim_result['staked_tokens'].iloc[0] - 1) * 100:.2f}%)")
            else:
                st.write(f"Change: {(sim_result['staked_tokens'].iloc[-1] - sim_result['staked_tokens'].iloc[0]):,.0f} tokens (N/A%)")
        st.markdown('</div>', unsafe_allow_html=True)


def display_monte_carlo_results(
    mc_results: Dict[str, Any], 
    show_confidence_intervals: bool = True, 
    show_percentiles: bool = True
) -> None:
    """
    Display the results of a Monte Carlo simulation.
    
    Args:
        mc_results: Dictionary with Monte Carlo simulation results
        show_confidence_intervals: Whether to show confidence intervals
        show_percentiles: Whether to show percentile bands
    """
    # Check if mc_results is valid
    if mc_results is None or not isinstance(mc_results, dict) or not mc_results:
        st.error("Invalid or empty Monte Carlo results. Please try running the simulation again.")
        return
    
    # Create tabs for Monte Carlo visualization
    mc_tabs = st.tabs(["Time Series", "Distribution"])
    
    with mc_tabs[0]:
        # Display Monte Carlo time series for different variables
        variable_tabs = st.tabs(["Token Supply", "Token Price", "Market Cap", "Staking"])
        
        with variable_tabs[0]:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            try:
                fig = plot_monte_carlo_results(
                    mc_results, 
                    "circulating_supply", 
                    show_confidence_intervals=show_confidence_intervals,
                    show_percentiles=show_percentiles
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting circulating supply: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with variable_tabs[1]:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            try:
                fig = plot_monte_carlo_results(
                    mc_results, 
                    "token_price", 
                    show_confidence_intervals=show_confidence_intervals,
                    show_percentiles=show_percentiles
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting token price: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with variable_tabs[2]:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            try:
                fig = plot_monte_carlo_results(
                    mc_results, 
                    "market_cap", 
                    show_confidence_intervals=show_confidence_intervals,
                    show_percentiles=show_percentiles
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting market cap: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with variable_tabs[3]:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            try:
                fig = plot_monte_carlo_results(
                    mc_results, 
                    "staked_tokens", 
                    show_confidence_intervals=show_confidence_intervals,
                    show_percentiles=show_percentiles
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting staked tokens: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with mc_tabs[1]:
        # Check if we have raw data for distribution analysis
        if 'raw_data' not in mc_results or mc_results['raw_data'] is None or len(mc_results['raw_data']) == 0:
            st.error("No raw data available for distribution analysis.")
            return
        
        # Distribution analysis
        variable_options = ["token_price", "market_cap", "circulating_supply", "staked_tokens"]
        available_vars = [var for var in variable_options if var in mc_results['raw_data'].columns]
        
        if not available_vars:
            st.error("No variables available for distribution analysis.")
            return
            
        selected_variable = st.selectbox(
            "Select Variable for Distribution Analysis",
            options=available_vars,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Get timesteps
        try:
            timesteps = sorted(mc_results['raw_data']['timestep'].unique())
        except Exception as e:
            st.error(f"Error getting timesteps: {str(e)}")
            return
            
        if not timesteps:
            st.error("No timesteps available for analysis.")
            return
        
        # Timestep slider
        selected_timestep = st.slider(
            "Select Timestep", 
            min_value=min(timesteps), 
            max_value=max(timesteps), 
            value=int(len(timesteps) / 2)
        )
        
        # Get the date for the selected timestep
        try:
            timestep_dates = mc_results['raw_data'][mc_results['raw_data']['timestep'] == selected_timestep]['date']
            if len(timestep_dates) > 0:
                timestep_date = timestep_dates.iloc[0]
                st.write(f"Selected Date: {timestep_date}")
            else:
                st.warning("No date available for selected timestep.")
        except Exception as e:
            st.warning(f"Error getting date for timestep: {str(e)}")
        
        # Plot distribution
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        try:
            hist_fig, stats = plot_distribution_at_timestep(mc_results, selected_variable, selected_timestep)
            st.plotly_chart(hist_fig, use_container_width=True)
            
            # Display statistics
            st.subheader("Distribution Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{stats['mean']:.5f}")
                st.metric("Standard Deviation", f"{stats['std_dev']:.5f}")
            with col2:
                st.metric("Median", f"{stats['median']:.5f}")
                st.metric("Coefficient of Variation", f"{stats['cv']:.5f}")
            with col3:
                if len(stats['percentiles']) >= 5:
                    st.metric("5th Percentile", f"{stats['percentiles'][0]:.5f}")
                    st.metric("95th Percentile", f"{stats['percentiles'][4]:.5f}")
        except Exception as e:
            st.error(f"Error creating distribution plot: {str(e)}")
            
        st.markdown('</div>', unsafe_allow_html=True)