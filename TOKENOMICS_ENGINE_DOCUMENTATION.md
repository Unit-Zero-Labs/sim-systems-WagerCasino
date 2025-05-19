# Unit Zero Labs Tokenomics Engine Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Models](#data-models)
5. [Simulation Engine](#simulation-engine)
6. [Visualization Features](#visualization-features)
7. [Monte Carlo Simulation](#monte-carlo-simulation)
8. [Installation and Usage](#installation-and-usage)
9. [Technical Requirements](#technical-requirements)

## Overview

The Unit Zero Labs Tokenomics Engine is a comprehensive simulation and visualization tool designed to model token economics for blockchain projects. It enables users to:

- Visualize token distribution, vesting schedules, and utility allocations over time
- Simulate token price dynamics, market cap evolution, and staking behavior
- Run Monte Carlo simulations to assess risk and potential outcomes across multiple scenarios
- Analyze the impact of different parameter settings on token economics

The tool is built as a Streamlit web application with a Python backend using the cadCAD simulation framework, providing an interactive experience for both token designers and stakeholders.

## Architecture

The application is structured around three main components:

1. **Data Input and Parsing** (`tokenomics_data.py`): Handles the loading and parsing of tokenomics configuration data, typically from CSV files.

2. **Simulation Engine** (`simulate.py`): Implements the system dynamics model using cadCAD to perform simulations of token behavior over time.

3. **Web Interface** (`app.py`): Provides an interactive Streamlit frontend with visualization capabilities.

The application follows a modular design where data parsing, simulation logic, and visualization are separated, making it easier to maintain and extend the codebase.

## Core Components

### `TokenomicsData` Class

Serves as the central data structure for storing tokenomics parameters and time-series data:

- Token allocation and vesting schedules
- Utility metrics and allocations
- Financial time-series data (price, market cap, etc.)
- Static parameters for simulation

### `TokenomicsSimulation` Class

Implements the simulation logic:

- Sets up initial state based on parsed data
- Defines policy functions for token dynamics
- Defines state update functions
- Provides methods for running simulations and analyzing results

### Streamlit App (`app.py`)

Handles the user interface:

- Data upload and processing
- Interactive parameter adjustment
- Visualization of input data and simulation results
- Tab-based navigation for different features

## Data Models

### Static Parameters

The system uses a variety of static parameters for simulation:

- Initial token supply and distribution
- Launch date and simulation horizon
- Public sale metrics (valuations, percentages)
- Vesting schedules (initial vesting, cliff periods, durations)
- Utility allocations (staking, burning, etc.)

### Time-Series Data

Several time-series datasets are generated and tracked:

- Vesting schedules for different token buckets
- Token price evolution
- Market cap and fully diluted valuation (FDV)
- Staking APR
- Utility allocations over time

### Token Buckets

The system models tokens in different allocation buckets:

- Public Sale allocations
- Team/Founders
- Advisors
- Strategic Partners
- Community
- CEX Liquidity
- Liquidity Pool
- Airdrops
- Staking Incentives

## Simulation Engine

### State Variables

Key state variables tracked in the simulation:

- `token_supply`: Total token supply
- `circulating_supply`: Tokens in circulation (excludes locked tokens)
- `staked_tokens`: Tokens staked by users
- `liquidity_pool_tokens`: Tokens in liquidity pools
- `token_price`: Current token price
- `market_cap`: Market capitalization (circulating supply × price)
- `staking_apr`: Annual percentage rate for staking

### Policy Functions

Policy functions that model system behaviors:

- `p_vesting_schedule`: Determines newly vested tokens based on vesting schedules
- `p_staking`: Models user staking behavior based on staking share parameter
- `p_token_price`: Models token price dynamics with market volatility

### State Update Functions

Functions that update system state:

- `s_update_supply`: Updates circulating supply based on vesting
- `s_update_staking`: Updates staked tokens based on staking behavior
- `s_update_price`: Updates token price based on price dynamics
- `s_update_market_cap`: Recalculates market cap based on price and supply

## Visualization Features

### Token Supply Charts

- Distribution of tokens across different buckets over time
- Circulating supply vs. total supply
- Vesting schedules for different stakeholders

### Financial Metrics Charts

- Token price over time
- Market cap and fully diluted valuation (FDV)
- DEX liquidity valuation

### Utility Charts

- Utility allocations over time
- Monthly utility incentives, burnings, and transfers
- Staking APR

### Simulation Results

- Comparison of simulated metrics with initial values
- Time-series charts for key metrics from simulation runs
- Statistical analysis of simulation results

## Monte Carlo Simulation

### Implementation

The Monte Carlo simulation feature:

- Runs the model multiple times with different random seeds
- Captures the range of possible outcomes for key metrics
- Calculates statistical measures (mean, standard deviation, percentiles)
- Visualizes the distribution of outcomes

### Visualization

Monte Carlo results are visualized through:

- Time-series charts with confidence intervals and percentile bands
- Probability distribution histograms at specific timesteps
- Statistical summary tables

### Analysis Tools

The system provides tools for analyzing Monte Carlo results:

- Confidence intervals for assessing prediction reliability
- Percentile bands for understanding the range of outcomes
- Probability distributions for detailed analysis at specific points in time

## Installation and Usage

### Installation

1. Clone the repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit application:
```
streamlit run app.py
```

### Usage Workflow

1. **Data Loading**: Upload a radCAD inputs CSV file with tokenomics parameters
2. **Data Exploration**: Explore the loaded data through various visualizations
3. **Simulation**: Adjust parameters and run simulations to see impacts
4. **Analysis**: Analyze simulation results with various metrics and visualizations

## Technical Requirements

### Dependencies

- Python 3.9+
- pandas 2.0.0
- streamlit 1.22.0
- plotly 5.14.1
- numpy 1.24.3
- cadCAD 0.4.28

### Input Data Format

The application expects a CSV file with the following structure:
- Parameter Name: Name of the tokenomics parameter
- Initial Value: Value of the parameter
- Other optional columns for min/max values, units, comments

## File Structure Explanation

- `app.py`: Main Streamlit application with UI and visualization logic
- `tokenomics_data.py`: Data parsing and processing
- `simulate.py`: Simulation engine using cadCAD
- `docs/`: Contains reference data files
- `public/`: Contains static assets like logos

## Data Processing Flow

1. **Data Input**: User uploads radCAD inputs CSV file
2. **Parsing**: System parses parameters and generates time-series data
3. **Visualization**: Initial data is visualized in the Data Tables tab
4. **Simulation Setup**: User adjusts simulation parameters
5. **Simulation Run**: System runs simulation with specified parameters
6. **Results Analysis**: Results are visualized and analyzed

## Simulation Parameters

Key parameters that can be adjusted for simulation:

- **Staking Share**: Percentage of circulating supply that is staked
- **Token Launch Price**: Initial price of the token at launch
- **Staking APR Multiplier**: Adjustment factor for staking APR
- **Market Volatility**: Level of randomness in price movements

## Monte Carlo Specific Parameters

Additional parameters for Monte Carlo simulation:

- **Number of Runs**: Number of simulation runs to perform
- **Confidence Intervals**: Whether to show 95% confidence intervals
- **Percentile Bands**: Whether to show percentile bands

---

This documentation provides a comprehensive overview of the Unit Zero Labs Tokenomics Engine, explaining its architecture, features, and usage. For detailed technical information about specific components, refer to the code documentation in the respective files.