import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tokenomics_data import TokenomicsData, parse_csv
from simulate import TokenomicsSimulation

def plot_token_buckets(data):
    """Plot the token buckets and circulating supply over time."""
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

def plot_price(data):
    """Plot the token price over time."""
    fig = go.Figure()
    
    # Use the time-series data instead of a constant value
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

def plot_valuations(data):
    """Plot the token valuations (Market Cap and FDV) over time."""
    fig = go.Figure()
    
    # Use the time-series data instead of constant values
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

def plot_dex_liquidity(data):
    """Plot the DEX liquidity valuation over time."""
    fig = go.Figure()
    
    # Use the time-series data instead of a constant value
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

def plot_utility_allocations(data):
    """Plot the utility allocations over time."""
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

def plot_monthly_utility(data):
    """Plot the monthly utility incentives/burnings/transfers."""
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

def plot_staking_apr(data):
    """Plot the staking APR over time."""
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

def main():
    """Main function to run the Streamlit app."""
    # Set page config
    st.set_page_config(page_title="UZL Tokenomics Engine", layout="wide")
    
    # Apply custom CSS for dark purple gradient background and Inconsolata font
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inconsolata:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #2A0845 0%, #6441A5 100%);
        color: white;
        font-family: 'Inconsolata', monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        font-family: 'Inconsolata', monospace;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 4px 4px 0px 0px;
        color: white;
        padding: 10px 20px;
        font-family: 'Inconsolata', monospace;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
    }
    .plot-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        padding: 10px;
    }
    h1, h2, h3, .stMarkdown, p, div {
        color: white;
        font-family: 'Inconsolata', monospace;
    }
    .header-title {
        margin-top: 0;
        margin-bottom: 0;
        font-size: 1.8rem;
        display: inline-block;
        vertical-align: middle;
        margin-left: 10px;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 20px;
    }
    .logo-title-container {
        display: flex;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logos
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Add some top padding
    
    # Create a header container with columns for better control
    col1, col2, col3 = st.columns([4, 3, 4])
    
    with col1:
        # Left side with UZ logo and title
        cols_left = st.columns([1, 2])
        with cols_left[0]:
            try:
                st.image("public/uz-logo.png", width=120)
            except:
                st.error("UZ logo not found. Please ensure 'uz-logo.png' is in the public directory.")
        
        with cols_left[1]:
            st.markdown("<h2 class='header-title'>Tokenomics Engine</h2>", unsafe_allow_html=True)
    
    with col3:
        # Right side with client logo
        try:
            st.image("public/client-logo.png", width=120)
        except:
            st.error("Client logo not found. Please ensure 'client-logo.png' is in the public directory.")
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Add some bottom padding
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Analysis", "Simulation", "Scenario Analysis"])
    
    with tab1:
        st.header("Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload Data Tables CSV", type="csv")
        
        if uploaded_file:
            # Parse the CSV file
            data = parse_csv(uploaded_file)
            
            if data:
                # Display the charts with custom container styling
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(plot_token_buckets(data), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(plot_price(data), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(plot_valuations(data), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(plot_dex_liquidity(data), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(plot_utility_allocations(data), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(plot_monthly_utility(data), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(plot_staking_apr(data), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Simulation")
        
        if uploaded_file:
            # Parse the CSV file if not already parsed
            if 'data' not in locals():
                data = parse_csv(uploaded_file)
            
            if data:
                # Create a TokenomicsSimulation instance
                simulation = TokenomicsSimulation(data)
                
                # Parameter sliders
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
                    # Fix: Ensure token_price is a float before using it in the slider
                    default_price = data.token_price
                    if isinstance(default_price, list):
                        default_price = default_price[0] if default_price else 0.03
                    elif not isinstance(default_price, (int, float)) or pd.isna(default_price):
                        default_price = 0.03
                        
                    token_price = st.slider(
                        "Token Launch Price", 
                        0.01, 0.05, 
                        float(default_price),  # Ensure it's a float
                        step=0.001,
                        format="$%.5f"
                    )
                    
                    market_volatility = st.slider(
                        "Market Volatility", 
                        0.0, 1.0, 
                        0.2,  # Default value
                        step=0.05
                    )
                
                # Run the simulation with cadCAD
                params = {
                    "staking_share": staking_share, 
                    "token_price": token_price,
                    "staking_apr_multiplier": staking_apr_multiplier,
                    "market_volatility": market_volatility
                }
                sim_result = simulation.run_simulation(params)
                
                # Create tabs for different metrics
                metric_tabs = st.tabs(["Token Supply", "Token Price", "Market Cap", "Staking"])
                
                with metric_tabs[0]:
                    # Plot the token supply results
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display the simulation results
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    st.subheader("Supply Results")
                    st.write(f"Initial Supply: {sim_result['circulating_supply'].iloc[0]:,.0f} tokens")
                    st.write(f"Final Supply: {sim_result['circulating_supply'].iloc[-1]:,.0f} tokens")
                    st.write(f"Change: {(sim_result['circulating_supply'].iloc[-1] - sim_result['circulating_supply'].iloc[0]):,.0f} tokens ({(sim_result['circulating_supply'].iloc[-1] / sim_result['circulating_supply'].iloc[0] - 1) * 100:.2f}%)")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_tabs[1]:
                    # Plot the token price results
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
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
    
    with tab3:
        st.header("Scenario Analysis")
        
        if uploaded_file:
            # Parse the CSV file if not already parsed
            if 'data' not in locals():
                data = parse_csv(uploaded_file)
            
            if data:
                # Create a TokenomicsSimulation instance
                simulation = TokenomicsSimulation(data)
                
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                
                # Define scenarios
                st.subheader("Define Scenarios")
                
                # Allow user to create multiple scenarios
                num_scenarios = st.number_input("Number of Scenarios", min_value=1, max_value=5, value=2)
                
                scenarios = []
                for i in range(num_scenarios):
                    st.write(f"### Scenario {i+1}")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        name = st.text_input(f"Scenario Name", value=f"Scenario {i+1}", key=f"name_{i}")
                        staking_share = st.slider(
                            "Staking Share", 
                            0.0, 1.0, 
                            0.5 + i*0.25,  # Different defaults for different scenarios
                            step=0.01,
                            key=f"staking_{i}"
                        )
                    
                    with col2:
                        token_price = st.slider(
                            "Token Launch Price", 
                            0.01, 0.05, 
                            0.03 + i*0.005,  # Different defaults
                            step=0.001,
                            format="$%.5f",
                            key=f"price_{i}"
                        )
                        market_volatility = st.slider(
                            "Market Volatility", 
                            0.0, 1.0, 
                            0.2,
                            step=0.05,
                            key=f"volatility_{i}"
                        )
                    
                    scenarios.append({
                        "name": name,
                        "params": {
                            "staking_share": staking_share,
                            "token_price": token_price,
                            "market_volatility": market_volatility
                        }
                    })
                
                # Run button
                if st.button("Run Scenario Analysis"):
                    # Run simulations for all scenarios
                    results = simulation.run_scenario_comparison(scenarios)
                    
                    # Display comparative charts
                    metrics = ["circulating_supply", "token_price", "market_cap", "staked_tokens"]
                    metric_names = ["Circulating Supply", "Token Price", "Market Cap", "Staked Tokens"]
                    
                    for metric, name in zip(metrics, metric_names):
                        fig = go.Figure()
                        
                        for scenario_name, result in results.items():
                            fig.add_trace(go.Scatter(
                                x=result['date'], 
                                y=result[metric], 
                                mode='lines', 
                                name=scenario_name
                            ))
                        
                        fig.update_layout(
                            title=f"Scenario Comparison: {name}",
                            xaxis_title="Date",
                            yaxis_title=name,
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.write("Please upload a CSV file to use the Scenario Analysis tab.")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.write("Please upload a CSV file to use the Scenario Analysis tab.")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 