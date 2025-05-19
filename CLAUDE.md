# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The Unit Zero Labs Tokenomics Engine is a comprehensive simulation and visualization tool for tokenomics data. It's built as a Streamlit web application with a Python backend using the cadCAD simulation framework, providing interactive visualization and Monte Carlo simulations for tokenomics modeling.

## Commands

### Setup and Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### Running the Application

```bash
# Run the Streamlit application
streamlit run app.py
```

The application will open in your default web browser, typically at http://localhost:8501

### Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run a specific test file
python -m unittest tests/test_charts.py
```

## Architecture

The application follows a modular architecture:

1. **Data Management:**
   - `TokenomicsData` class (tokenomics_data.py): Central data structure
   - `DataManager` class (logic/data_manager.py): Handles loading and processing

2. **Simulation Engine:**
   - `TokenomicsSimulation` class (simulate.py): Core simulation logic using radCAD
   - `MonteCarloSimulator` class (logic/monte_carlo.py): Optimizes Monte Carlo simulations

3. **State Management:**
   - `StateManager` class (logic/state_manager.py): Manages session state

4. **Visualization:**
   - `charts.py`: Plotting functions
   - `components.py`: Reusable UI components
   - `styles.py`: CSS styling

5. **Utilities:**
   - Caching (utils/cache.py)
   - Configuration (utils/config.py)
   - Error handling (utils/error_handler.py)
   - Validation (utils/validators.py)

## Key Components

### Simulation Engine

The core simulation engine uses radCAD to model token economics:
- State variables: token supply, circulating supply, staked tokens, etc.
- Policy functions: vesting schedule, staking behavior, token price dynamics
- State update functions: update supply, staking, price, market cap

### Monte Carlo Simulation

The Monte Carlo functionality extends the basic simulation:
- Runs multiple simulations with different random seeds
- Uses parallel processing for performance
- Calculates statistical measures (mean, confidence intervals, percentiles)
- Visualizes distributions and confidence bands

### Data Input

The system expects CSV files with:
- Parameter names and initial values
- Time-series data with dates in MM/YY format
- Sections for vesting schedules, adoption metrics, and utility allocations

## Key Files

- **app.py**: Main entry point and UI organization
- **simulate.py**: TokenomicsSimulation class for running simulations
- **tokenomics_data.py**: Data parsing and structure
- **logic/monte_carlo.py**: Monte Carlo simulation implementation
- **visualization/charts.py**: All plotting functions

## Technical Dependencies

- Python 3.9+
- pandas 2.0.0
- streamlit 1.22.0
- plotly 5.14.1
- numpy 1.24.3
- cadCAD 0.4.28