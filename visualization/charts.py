"""
Visualization module for the Unit Zero Labs Tokenomics Engine.
Contains all plotting functions used throughout the application.
"""

import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union


def apply_chart_styling(fig: go.Figure) -> go.Figure:
    """
    Apply consistent styling to a Plotly figure.
    
    Args:
        fig: Plotly figure object
        
    Returns:
        Styled Plotly figure object
    """
    # Update paper and plot backgrounds
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor="rgba(255,255,255,0.1)",
        showline=True,
        linewidth=1,
        linecolor="rgba(255,255,255,0.5)"
    )
    
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor="rgba(255,255,255,0.1)",
        showline=True,
        linewidth=1,
        linecolor="rgba(255,255,255,0.5)"
    )
    
    return fig


def plot_token_buckets(data: Any) -> go.Figure:
    """
    Plot the token buckets and circulating supply over time.
    
    Args:
        data: TokenomicsData object containing vesting data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add traces for each token bucket
    for bucket in data.vesting_series.index:
        if bucket and bucket != "SUM":  # Skip empty or summary rows
            fig.add_trace(go.Scatter(
                x=data.dates, 
                y=data.vesting_cumulative.loc[bucket], 
                mode='lines', 
                name=bucket
            ))
    
    # Add circulating supply (sum of all buckets except Liquidity Pool)
    if "Liquidity Pool" in data.vesting_cumulative.index:
        circulating_supply = data.vesting_cumulative.drop("Liquidity Pool", errors='ignore').sum()
        fig.add_trace(go.Scatter(
            x=data.dates, 
            y=circulating_supply, 
            mode='lines', 
            name='Circulating Supply',
            line=dict(width=3, dash='dash')
        ))
    
    fig.update_layout(
        title="Project Token Buckets & Token Supply",
        xaxis_title="Date",
        yaxis_title="Tokens",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return apply_chart_styling(fig)


def plot_price(data: Any) -> go.Figure:
    """
    Plot the token price over time.
    
    Args:
        data: TokenomicsData object containing price data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Use the time-series data
    fig.add_trace(go.Scatter(
        x=data.dates, 
        y=data.token_price_series.loc["Token Price"], 
        mode='lines', 
        name='Price'
    ))
    
    fig.update_layout(
        title="Token Price",
        xaxis_title="Date",
        yaxis_title="USD",
    )
    
    return apply_chart_styling(fig)


def plot_valuations(data: Any) -> go.Figure:
    """
    Plot the token valuations (Market Cap and FDV) over time.
    
    Args:
        data: TokenomicsData object containing valuation data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Use the time-series data
    fig.add_trace(go.Scatter(
        x=data.dates, 
        y=data.market_cap_series.loc["Market Cap"], 
        mode='lines', 
        name='Market Cap'
    ))
    
    fig.add_trace(go.Scatter(
        x=data.dates, 
        y=data.fdv_mc_series.loc["FDV MC"], 
        mode='lines', 
        name='FDV MC'
    ))
    
    fig.update_layout(
        title="Token Valuations",
        xaxis_title="Date",
        yaxis_title="USD",
    )
    
    return apply_chart_styling(fig)


def plot_dex_liquidity(data: Any) -> go.Figure:
    """
    Plot the DEX liquidity valuation over time.
    
    Args:
        data: TokenomicsData object containing liquidity data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Use the time-series data
    fig.add_trace(go.Scatter(
        x=data.dates, 
        y=data.liquidity_pool_series.loc["LP Valuation"], 
        mode='lines', 
        name='LP Valuation'
    ))
    
    fig.update_layout(
        title="DEX Liquidity Valuation",
        xaxis_title="Date",
        yaxis_title="USD",
    )
    
    return apply_chart_styling(fig)


def plot_utility_allocations(data: Any) -> go.Figure:
    """
    Plot the utility allocations over time.
    
    Args:
        data: TokenomicsData object containing utility allocation data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add traces for each utility allocation
    for utility in data.utility_allocations.index:
        if utility and "CUM" in utility:  # Only include cumulative rows
            fig.add_trace(go.Scatter(
                x=data.dates, 
                y=data.utility_allocations.loc[utility], 
                mode='lines', 
                name=utility.replace(" CUM.", "")
            ))
    
    fig.update_layout(
        title="Utility Allocations",
        xaxis_title="Date",
        yaxis_title="Tokens",
    )
    
    return apply_chart_styling(fig)


def plot_monthly_utility(data: Any) -> go.Figure:
    """
    Plot the monthly utility incentives/burnings/transfers.
    
    Args:
        data: TokenomicsData object containing monthly utility data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add traces for each utility
    for utility in data.monthly_utility.index:
        if utility and utility not in ["SUM", "0", 0]:
            fig.add_trace(go.Scatter(
                x=data.dates, 
                y=data.monthly_utility.loc[utility], 
                mode='lines', 
                name=utility
            ))
    
    fig.update_layout(
        title="Monthly Utility Incentives / Burnings / Transfers",
        xaxis_title="Date",
        yaxis_title="Tokens",
    )
    
    return apply_chart_styling(fig)


def plot_staking_apr(data: Any) -> go.Figure:
    """
    Plot the staking APR over time.
    
    Args:
        data: TokenomicsData object containing staking APR data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.dates, 
        y=data.staking_apr.loc["Staking APR"], 
        mode='lines', 
        name='APR'
    ))
    
    fig.update_layout(
        title="Staking APR",
        xaxis_title="Date",
        yaxis_title="APR (%)",
    )
    
    return apply_chart_styling(fig)


def plot_monte_carlo_results(
    mc_results: Dict[str, Any], 
    variable: str, 
    show_confidence_intervals: bool = True, 
    show_percentiles: bool = True
) -> go.Figure:
    """
    Create a plot for Monte Carlo simulation results with confidence intervals and percentile bands.
    
    Args:
        mc_results: Results from a Monte Carlo simulation
        variable: The state variable to visualize (e.g., 'token_price', 'market_cap')
        show_confidence_intervals: Whether to show 95% confidence intervals
        show_percentiles: Whether to show percentile bands (5th, 25th, 75th, 95th)
        
    Returns:
        Plotly figure object
    """
    # Get visualization data
    x_values = mc_results['mean'][variable].index
    mean_values = mc_results['mean'][variable].iloc[:, 0].values
    
    # Create the plot
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=x_values,
        y=mean_values,
        mode='lines',
        name='Mean',
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence intervals
    if show_confidence_intervals:
        conf_intervals = mc_results['conf_intervals'][variable].values
        lower_ci = [ci[0] for ci in conf_intervals]
        upper_ci = [ci[1] for ci in conf_intervals]
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=upper_ci,
            mode='lines',
            name='95% CI Upper',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=lower_ci,
            mode='lines',
            name='95% CI Lower',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            showlegend=False
        ))
    
    # Add percentile bands
    if show_percentiles:
        percentiles = mc_results['percentiles'][variable].values
        
        # 5th and 95th percentiles
        fig.add_trace(go.Scatter(
            x=x_values,
            y=[p[4] for p in percentiles],  # 95th percentile
            mode='lines',
            name='95th Percentile',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=[p[0] for p in percentiles],  # 5th percentile
            mode='lines',
            name='5th Percentile',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dot')
        ))
        
        # 25th and 75th percentiles
        fig.add_trace(go.Scatter(
            x=x_values,
            y=[p[3] for p in percentiles],  # 75th percentile
            mode='lines',
            name='75th Percentile',
            line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=[p[1] for p in percentiles],  # 25th percentile
            mode='lines',
            name='25th Percentile',
            line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dot')
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Monte Carlo Simulation: {variable.replace('_', ' ').title()}",
        xaxis_title="Date",
        yaxis_title=variable.replace('_', ' ').title(),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    return apply_chart_styling(fig)


def plot_distribution_at_timestep(
    mc_results: Dict[str, Any], 
    variable: str, 
    timestep: int
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Create a probability distribution plot for a specific variable at a specific timestep.
    
    Args:
        mc_results: Results from a Monte Carlo simulation
        variable: The state variable to analyze (e.g., 'token_price', 'market_cap')
        timestep: The timestep to analyze
        
    Returns:
        Tuple containing:
            - Plotly figure object for the distribution
            - Dictionary of statistics
    """
    # Get distribution data
    raw_data = mc_results['raw_data']
    values = raw_data[(raw_data['timestep'] == timestep)][variable].values
    
    # Calculate statistics
    mean = np.mean(values)
    median = np.median(values)
    std_dev = np.std(values)
    percentiles = np.percentile(values, [5, 25, 50, 75, 95])
    
    # Create histogram with density curve
    hist_fig = ff.create_distplot(
        [values], 
        [variable.replace('_', ' ').title()],
        bin_size=(max(values) - min(values)) / 20,
        show_rug=False,
        colors=['rgba(0, 0, 255, 0.6)']
    )
    
    # Add vertical lines for statistics
    hist_fig.add_vline(x=mean, line_dash="solid", line_color="blue", annotation_text="Mean")
    hist_fig.add_vline(x=median, line_dash="dash", line_color="green", annotation_text="Median")
    hist_fig.add_vline(x=percentiles[0], line_dash="dot", line_color="red", annotation_text="5th")
    hist_fig.add_vline(x=percentiles[4], line_dash="dot", line_color="red", annotation_text="95th")
    
    # Update layout
    hist_fig.update_layout(
        title=f"Probability Distribution at Timestep {timestep}",
        xaxis_title=variable.replace('_', ' ').title(),
        yaxis_title="Probability Density",
    )
    
    # Apply consistent styling
    hist_fig = apply_chart_styling(hist_fig)
    
    # Create statistics dictionary
    stats = {
        'mean': mean,
        'median': median,
        'std_dev': std_dev,
        'percentiles': percentiles,
        'cv': std_dev / mean if mean != 0 else float('inf')  # Coefficient of variation
    }
    
    return hist_fig, stats


def plot_token_supply_simulation(sim_result: pd.DataFrame) -> go.Figure:
    """
    Plot the token supply simulation results.
    
    Args:
        sim_result: DataFrame with simulation results
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sim_result['date'], 
        y=sim_result['circulating_supply'], 
        mode='lines', 
        name='Circulating Supply'
    ))
    
    fig.add_trace(go.Scatter(
        x=sim_result['date'], 
        y=sim_result['token_supply'], 
        mode='lines', 
        name='Total Supply',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title="Simulated Token Supply",
        xaxis_title="Date",
        yaxis_title="Tokens",
    )
    
    return apply_chart_styling(fig)


def plot_token_price_simulation(sim_result: pd.DataFrame) -> go.Figure:
    """
    Plot the token price simulation results.
    
    Args:
        sim_result: DataFrame with simulation results
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sim_result['date'], 
        y=sim_result['token_price'], 
        mode='lines', 
        name='Token Price'
    ))
    
    fig.update_layout(
        title="Simulated Token Price",
        xaxis_title="Date",
        yaxis_title="USD",
    )
    
    return apply_chart_styling(fig)


def plot_market_cap_simulation(sim_result: pd.DataFrame) -> go.Figure:
    """
    Plot the market cap simulation results.
    
    Args:
        sim_result: DataFrame with simulation results
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sim_result['date'], 
        y=sim_result['market_cap'], 
        mode='lines', 
        name='Market Cap'
    ))
    
    fig.update_layout(
        title="Simulated Market Cap",
        xaxis_title="Date",
        yaxis_title="USD",
    )
    
    return apply_chart_styling(fig)


def plot_staking_simulation(sim_result: pd.DataFrame) -> go.Figure:
    """
    Plot the staking simulation results.
    
    Args:
        sim_result: DataFrame with simulation results
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sim_result['date'], 
        y=sim_result['staked_tokens'], 
        mode='lines', 
        name='Staked Tokens'
    ))
    
    fig.update_layout(
        title="Simulated Staked Tokens",
        xaxis_title="Date",
        yaxis_title="Tokens",
    )
    
    return apply_chart_styling(fig)