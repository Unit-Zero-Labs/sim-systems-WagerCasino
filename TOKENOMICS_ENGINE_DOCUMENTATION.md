# Unit Zero Labs Tokenomics Engine Documentation

## Running the Application

To run the Tokenomics Engine, execute the following commands:

```bash
# Install required packages
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

The application will open in your default web browser, typically at http://localhost:8501

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Models](#data-models)
5. [Simulation Engine](#simulation-engine)
6. [Visualization Features](#visualization-features)
7. [Monte Carlo Simulation](#monte-carlo-simulation)
8. [Caching System](#caching-system)
9. [Error Handling](#error-handling)
10. [Installation and Usage](#installation-and-usage)
11. [Technical Requirements](#technical-requirements)
12. [Testing](#testing)

## Overview

The Unit Zero Labs Tokenomics Engine is a comprehensive simulation and visualization tool designed to model token economics for blockchain projects. It enables users to:

- Visualize token distribution, vesting schedules, and utility allocations over time
- Simulate token price dynamics, market cap evolution, and staking behavior
- Run Monte Carlo simulations to assess risk and potential outcomes across multiple scenarios
- Analyze the impact of different parameter settings on token economics

The tool is built as a Streamlit web application with a Python backend using the cadCAD simulation framework, providing an interactive experience for both token designers and stakeholders.

## Architecture

The application is structured around the following components:

```
sim-systems/
├── app.py                      # Main entry point
├── simulate.py                 # Original simulation code
├── tokenomics_data.py          # Original data processing code
├── visualization/              # Visualization components
│   ├── __init__.py
│   ├── charts.py               # All plotting functions
│   ├── components.py           # Reusable UI components
│   └── styles.py               # CSS styles and theming
├── logic/                      # Business logic
│   ├── __init__.py
│   ├── data_manager.py         # Data loading and transformations
│   ├── state_manager.py        # Session state management
│   └── monte_carlo.py          # Monte Carlo specific logic
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── cache.py                # Caching utilities
│   ├── validators.py           # Input validation
│   └── error_handler.py        # Error handling
└── tests/                      # Testing framework
    ├── __init__.py
    └── test_charts.py          # Example test file
```

The application follows a modular design where data parsing, simulation logic, and visualization are separated, making it easier to maintain and extend the codebase.

## Core Components

### Data Layer

#### `TokenomicsData` Class

Serves as the central data structure for storing tokenomics parameters and time-series data:

- Token allocation and vesting schedules
- Utility metrics and allocations
- Financial time-series data (price, market cap, etc.)
- Static parameters for simulation

#### `DataManager` Class

Handles the loading and processing of tokenomics data:

- Validates input data
- Processes radCAD inputs
- Provides methods for exporting data
- Calculates summary statistics

### Business Logic

#### `TokenomicsSimulation` Class

Implements the simulation logic:

- Sets up initial state based on parsed data
- Defines policy functions for token dynamics
- Defines state update functions
- Provides methods for running simulations

#### `MonteCarloSimulator` Class

Optimizes Monte Carlo simulations:

- Runs simulations in parallel for better performance
- Provides progress tracking
- Calculates statistical measures from multiple runs

#### `StateManager` Class

Manages application state:

- Initializes and maintains session state
- Provides a consistent interface for state access
- Handles state updates from UI interactions

### Presentation Layer

#### Visualization Module

Contains all the plotting functions:

- Chart generation for token data
- Visualization of simulation results
- Monte Carlo result visualization

#### Components Module

Reusable UI components:

- Headers and layout elements
- Form controls for simulation parameters
- Results display components

#### Styles Module

CSS styling for the application:

- Consistent theming across components
- Responsive design elements
- Custom styling for Streamlit components

### Utilities

#### Configuration Management

Centralized configuration system:

- Default values for simulation parameters
- UI configuration
- Cache settings

#### Caching System

Optimizes performance through caching:

- Memory and persistent caching
- Time-to-live management
- Cache statistics

#### Error Handling

Consistent error management:

- Error logging
- User-friendly error messages
- Exception wrapping

#### Validators

Input validation utilities:

- Parameter validation
- Data format validation
- Error message formatting

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
- Uses parallel processing for improved performance
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

## Caching System

### Memory Cache

- Fast in-memory cache for frequently accessed data
- Automatic cache size management
- Access and update time tracking

### Persistent Cache

- File-based caching for data that persists between sessions
- Automatic cache invalidation
- Support for time-to-live settings

### Decorator Interface

- `@cached` decorator for easy function caching
- `@st_cache_data` decorator compatible with Streamlit's caching system
- Cache statistics and monitoring

## Error Handling

### Error Logging

- Centralized error logging
- Context-aware error messages
- Support for detailed error information

### User Feedback

- Consistent error presentation in the UI
- Appropriate error levels (info, warning, error)
- Detailed error information for debugging

### Validation System

- Input parameter validation
- Data format and consistency validation
- Helpful error messages for invalid inputs

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

## Testing

### Unit Tests

The application includes unit tests for key components:

- Chart generation
- Data processing
- Simulation logic

### Running Tests

Execute the tests using Python's unittest framework:

```bash
python -m unittest discover tests
```

### Test Coverage

The test framework focuses on:

- Ensuring visualizations work correctly
- Validating data processing logic
- Testing simulation calculations

---

This documentation provides a comprehensive overview of the Unit Zero Labs Tokenomics Engine, explaining its architecture, features, and usage. For detailed technical information about specific components, refer to the code documentation in the respective files.