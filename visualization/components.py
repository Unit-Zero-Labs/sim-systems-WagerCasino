"""
UI components module for the Unit Zero Labs Tokenomics Engine.
Contains reusable UI components for the Streamlit application.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from .charts import (
    plot_token_supply_simulation,
    plot_token_price_simulation,
    plot_market_cap_simulation,
    plot_staking_simulation,
    plot_monte_carlo_results,
    plot_distribution_at_timestep,
    plot_time_series_from_simulator
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


def display_stochastic_parameter_controls() -> Dict[str, Any]:
    """
    Display controls for stochastic simulation parameters.
    
    Returns:
        Dictionary of parameter values
    """
    st.subheader("Stochastic Model Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        staking_share = st.slider(
            "Staking Share", 
            0.0, 1.0, 
            0.75,  # Default value
            step=0.01,
            help="Fraction of circulating supply that gets staked"
        )
        
        staking_apr_multiplier = st.slider(
            "Staking APR Multiplier", 
            0.5, 2.0, 
            1.0,  # Default value
            step=0.1,
            help="Multiplier for the staking APR (base APR × multiplier)"
        )
    
    with col2:
        token_price = st.slider(
            "Token Launch Price", 
            0.01, 0.05, 
            0.03,  # Default value
            step=0.001,
            format="$%.5f",
            help="Initial token price at launch"
        )
        
        market_volatility = st.slider(
            "Market Volatility", 
            0.0, 1.0, 
            0.2,  # Default value
            step=0.05,
            help="Level of price volatility in the market (0=none, 1=high)"
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
    enable_monte_carlo = st.checkbox(
        "Enable Monte Carlo Simulation", 
        value=False,
        help="Run multiple simulations with different random seeds to analyze uncertainty"
    )
    
    if enable_monte_carlo:
        col1, col2 = st.columns(2)
        with col1:
            num_runs = st.slider(
                "Number of Runs", 
                10, 100, 50, 
                step=10,
                help="More runs = more accurate statistics but slower execution"
            )
            show_confidence_intervals = st.checkbox(
                "Show Confidence Intervals", 
                value=True,
                help="Display 95% confidence intervals around the mean"
            )
        
        with col2:
            show_percentiles = st.checkbox(
                "Show Percentile Bands", 
                value=True,
                help="Display 5th, 25th, 75th, and 95th percentile bands"
            )
    else:
        num_runs = 1
        show_confidence_intervals = True
        show_percentiles = True
    
    return enable_monte_carlo, num_runs, show_confidence_intervals, show_percentiles


def display_agent_based_parameter_controls() -> Dict[str, Any]:
    """
    Display controls for agent-based simulation parameters.
    
    Returns:
        Dictionary of parameter values
    """
    st.subheader("Agent-Based Model Parameters")
    
    # Agent counts
    st.write("Agent Populations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        random_trader_count = st.slider(
            "Random Traders", 
            0, 50, 
            20,  # Default value
            step=5,
            help="Agents that make random trading decisions"
        )
    
    with col2:
        trend_follower_count = st.slider(
            "Trend Followers", 
            0, 50, 
            15,  # Default value
            step=5,
            help="Agents that buy in uptrends and sell in downtrends"
        )
    
    with col3:
        staking_agent_count = st.slider(
            "Staking Agents", 
            0, 50, 
            10,  # Default value
            step=5,
            help="Agents that focus on staking based on APR"
        )
    
    # Market parameters
    st.write("Market Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        token_price = st.slider(
            "Initial Token Price", 
            0.01, 0.05, 
            0.03,  # Default value
            step=0.001,
            format="$%.5f",
            help="Initial token price at the start of simulation"
        )
        
        liquidity_factor = st.slider(
            "Liquidity Factor", 
            0.00001, 0.001, 
            0.0001,  # Default value
            step=0.00001,
            format="%.5f",
            help="How much price impact from buying/selling (lower=deeper liquidity)"
        )
    
    with col2:
        max_price_change_pct = st.slider(
            "Max Price Change %", 
            0.01, 0.5, 
            0.1,  # Default value
            step=0.01,
            format="%.2f",
            help="Maximum price change per time step"
        )
        
        min_token_price = st.slider(
            "Min Token Price", 
            0.0001, 0.01, 
            0.001,  # Default value
            step=0.0001,
            format="$%.4f",
            help="Minimum token price floor"
        )
    
    # Staking parameters
    st.write("Staking Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_staking_apr = st.slider(
            "Base Staking APR", 
            0.01, 0.2, 
            0.05,  # Default value
            step=0.01,
            format="%.2f",
            help="Initial staking annual percentage rate"
        )
    
    with col2:
        apr_scaling_factor = st.slider(
            "APR Scaling Factor", 
            0.1, 1.0, 
            0.5,  # Default value
            step=0.1,
            format="%.1f",
            help="How quickly APR decreases as more tokens are staked"
        )
    
    with col3:
        min_staking_apr = st.slider(
            "Min Staking APR", 
            0.001, 0.05, 
            0.01,  # Default value
            step=0.001,
            format="%.3f",
            help="Minimum staking APR floor"
        )
    
    # Return parameters as a dictionary
    return {
        "token_price": token_price,
        "random_trader_count": random_trader_count,
        "trend_follower_count": trend_follower_count,
        "staking_agent_count": staking_agent_count,
        "liquidity_factor": liquidity_factor,
        "max_price_change_pct": max_price_change_pct,
        "min_token_price": min_token_price,
        "base_staking_apr": base_staking_apr,
        "apr_scaling_factor": apr_scaling_factor,
        "min_staking_apr": min_staking_apr
    }


def display_single_run_results(sim_result: pd.DataFrame) -> None:
    """
    Display the results of a single simulation run.
    
    Args:
        sim_result: DataFrame with simulation results
    """
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


def display_agent_based_results(sim_result: pd.DataFrame, agent_data: pd.DataFrame) -> None:
    """
    Display the results of an agent-based simulation.
    
    Args:
        sim_result: DataFrame with simulation results
        agent_data: DataFrame with agent data (portfolio values, holdings, etc.)
    """
    # Create tabs for different aspects of the results
    result_tabs = st.tabs(["Market Metrics", "Agent Behavior", "Agent Portfolios"])
    
    with result_tabs[0]:
        # Display market metrics (similar to single_run_results but with different context)
        st.subheader("Market Metrics Over Time")
        
        # Create subtabs for different metrics
        metric_tabs = st.tabs(["Token Price", "Staking", "Supply", "Market Cap"])
        
        with metric_tabs[0]:
            # Plot token price
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig = plot_token_price_simulation(sim_result)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display price statistics
            st.subheader("Price Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Price", f"${sim_result['token_price'].iloc[0]:.5f}")
            with col2:
                st.metric("Final Price", f"${sim_result['token_price'].iloc[-1]:.5f}")
            with col3:
                price_change_pct = (sim_result['token_price'].iloc[-1] / sim_result['token_price'].iloc[0] - 1) * 100
                st.metric("Price Change", f"{price_change_pct:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_tabs[1]:
            # Plot staking
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig = plot_staking_simulation(sim_result)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display staking statistics
            st.subheader("Staking Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Staked Tokens", f"{sim_result['staked_tokens'].iloc[-1]:,.0f}")
            with col2:
                staking_ratio = sim_result['staked_tokens'].iloc[-1] / sim_result['circulating_supply'].iloc[-1] if sim_result['circulating_supply'].iloc[-1] > 0 else 0
                st.metric("Staking Ratio", f"{staking_ratio:.2%}")
            with col3:
                st.metric("Final Staking APR", f"{sim_result['staking_apr'].iloc[-1]:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_tabs[2]:
            # Plot token supply
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig = plot_token_supply_simulation(sim_result)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display supply statistics
            st.subheader("Supply Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Circulating Supply", f"{sim_result['circulating_supply'].iloc[-1]:,.0f}")
            with col2:
                effective_supply = sim_result['circulating_supply'].iloc[-1] - sim_result['staked_tokens'].iloc[-1]
                st.metric("Effective Circulating Supply", f"{effective_supply:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_tabs[3]:
            # Plot market cap
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig = plot_market_cap_simulation(sim_result)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display market cap statistics
            st.subheader("Valuation Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Market Cap", f"${sim_result['market_cap'].iloc[-1]:,.2f}")
            with col2:
                mc_change_pct = (sim_result['market_cap'].iloc[-1] / sim_result['market_cap'].iloc[0] - 1) * 100 if sim_result['market_cap'].iloc[0] > 0 else 0
                st.metric("Market Cap Change", f"{mc_change_pct:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with result_tabs[1]:
        # Display agent behavior analysis
        st.subheader("Agent Behavior Analysis")
        
        if agent_data is None or len(agent_data) == 0:
            st.error("No agent data available for analysis.")
            return
        
        # Aggregate statistics by agent type
        agent_types = agent_data["type"].unique()
        
        # Create bar chart for token holdings by agent type
        token_holdings = agent_data.groupby("type")["tokens"].sum()
        staked_holdings = agent_data.groupby("type")["staked_tokens"].sum()
        
        # Calculate total holdings including staked tokens
        total_holdings = pd.DataFrame({
            "tokens": token_holdings,
            "staked_tokens": staked_holdings
        })
        total_holdings["total"] = total_holdings["tokens"] + total_holdings["staked_tokens"]
        
        # Create holdings figure
        fig_holdings = go.Figure()
        fig_holdings.add_trace(go.Bar(
            x=total_holdings.index,
            y=total_holdings["tokens"],
            name="Liquid Tokens",
            marker_color="blue"
        ))
        fig_holdings.add_trace(go.Bar(
            x=total_holdings.index,
            y=total_holdings["staked_tokens"],
            name="Staked Tokens",
            marker_color="green"
        ))
        fig_holdings.update_layout(
            title="Token Holdings by Agent Type",
            xaxis_title="Agent Type",
            yaxis_title="Tokens",
            barmode="stack"
        )
        
        st.plotly_chart(fig_holdings, use_container_width=True)
        
        # Create bar chart for portfolio value by agent type
        portfolio_value = agent_data.groupby("type")["portfolio_value"].sum()
        
        fig_portfolio = go.Figure(go.Bar(
            x=portfolio_value.index,
            y=portfolio_value.values,
            marker_color="purple"
        ))
        fig_portfolio.update_layout(
            title="Total Portfolio Value by Agent Type",
            xaxis_title="Agent Type",
            yaxis_title="Portfolio Value (USD)"
        )
        
        st.plotly_chart(fig_portfolio, use_container_width=True)
    
    with result_tabs[2]:
        # Display individual agent portfolios
        st.subheader("Individual Agent Portfolios")
        
        if agent_data is None or len(agent_data) == 0:
            st.error("No agent data available for analysis.")
            return
        
        # Filter by agent type
        agent_type_filter = st.selectbox(
            "Filter by Agent Type",
            options=["All"] + list(agent_data["type"].unique()),
            index=0
        )
        
        filtered_data = agent_data
        if agent_type_filter != "All":
            filtered_data = agent_data[agent_data["type"] == agent_type_filter]
        
        # Display agent data as a table
        st.dataframe(
            filtered_data.style.format({
                "tokens": "{:,.2f}",
                "staked_tokens": "{:,.2f}",
                "cash": "${:,.2f}",
                "portfolio_value": "${:,.2f}"
            }),
            use_container_width=True
        )
        
        # Display distribution of portfolio values
        fig_dist = go.Figure()
        
        for agent_type in agent_data["type"].unique():
            type_data = agent_data[agent_data["type"] == agent_type]
            fig_dist.add_trace(go.Box(
                y=type_data["portfolio_value"],
                name=agent_type,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig_dist.update_layout(
            title="Distribution of Portfolio Values by Agent Type",
            yaxis_title="Portfolio Value (USD)"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)