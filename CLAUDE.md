# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The Unit Zero Labs Tokenomics Engine is a comprehensive simulation and visualization tool for tokenomics data. It's built as a Streamlit web application with a Python backend using the radCAD simulation framework, providing interactive visualization and Monte Carlo simulations for tokenomics modeling.

**Key Features:**
- **Parameter-First Design**: Automatically discovers and adapts to new parameters from CSV inputs
- **Dynamic Policy Creation**: Creates simulation policies based on available parameters
- **Dynamic UI Generation**: Automatically generates appropriate UI controls for any parameter type
- **Extensible Architecture**: Scales seamlessly from one client to hundreds without code changes

## Recent Major Updates

### Unified Simulation Interface (Latest)
- **Removed Monte Carlo Toggle**: Eliminated the artificial distinction between "single" and "Monte Carlo" simulations
- **Simplified Interface**: Now uses a single "Number of Runs" slider (1 = single run, >1 = Monte Carlo analysis)
- **Cleaner UX**: Removed redundant controls and streamlined the simulation workflow
- **Conceptual Clarity**: Every simulation with randomness is Monte Carlo - single run is just n=1

### Parameter-First Architecture Implementation
- **Dynamic Parameter Discovery**: Automatically categorizes and types parameters from CSV inputs
- **Policy Factory System**: Creates radCAD policies dynamically based on available parameters
- **Dynamic UI Generator**: Generates Streamlit controls automatically based on parameter definitions
- **Parameter Registry**: Central system for managing parameter metadata and validation

### Error Resolution & Stability Improvements
- **Fixed Streamlit Type Errors**: Resolved slider parameter type mismatches (min_value, max_value, step consistency)
- **Fixed Pandas Deprecation Warnings**: Added `.infer_objects(copy=False)` after pandas fill operations
- **Fixed Pickle/Serialization Issues**: Resolved enum serialization problems in radCAD simulations
- **Safe DataFrame Operations**: Added empty checks and bounds validation for pandas indexing
- **Updated Deprecated Functions**: Replaced `st.experimental_rerun()` with `st.rerun()`

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

The application follows a modular, parameter-driven architecture:

1. **Parameter Management:**
   - `ParameterRegistry` class (logic/parameter_registry.py): Automatic parameter discovery and categorization
   - `ParameterDefinition` dataclass: Metadata for UI generation and validation
   - `PolicyFactory` class (logic/policy_factory.py): Dynamic policy creation based on parameters

2. **Data Management:**
   - `TokenomicsData` class (tokenomics_data.py): Central data structure
   - `DataManager` class (logic/data_manager.py): Enhanced with parameter registry integration

3. **Dynamic UI System:**
   - `DynamicUIGenerator` class (visualization/dynamic_components.py): Automatic UI control generation
   - Type-aware control creation (sliders, inputs, checkboxes, etc.)
   - Parameter validation and coverage reporting

4. **Simulation Engine:**
   - `TokenomicsSimulation` class (simulate.py): Core simulation logic using radCAD
   - `MonteCarloSimulator` class (logic/monte_carlo.py): Optimizes Monte Carlo simulations
   - Dynamic policy discovery and activation

5. **State Management:**
   - `StateManager` class (logic/state_manager.py): Manages session state

6. **Visualization:**
   - `charts.py`: Plotting functions
   - `components.py`: Reusable UI components
   - `styles.py`: CSS styling

7. **Utilities:**
   - Caching (utils/cache.py)
   - Configuration (utils/config.py)
   - Error handling (utils/error_handler.py)
   - Validation (utils/validators.py)

## Key Components

### Parameter-First Design

The system automatically adapts to new parameters without code changes:

```python
# Parameters are automatically discovered and categorized
parameter_registry.register_parameters_from_csv(csv_params)

# UI controls are generated dynamically
dynamic_ui = create_dynamic_ui_generator(parameter_registry)
params = dynamic_ui.display_parameter_controls()

# Policies are created based on available parameters
policies = policy_factory.discover_and_create_policies(data)
```

**Parameter Categories:**
- Tokenomics (supply, allocations)
- Vesting (schedules, cliffs)
- Staking (APR, rewards)
- Pricing (volatility, valuations)
- Points Campaigns (off-chain systems)
- Agent Behavior (trading patterns)
- Utility Mechanisms (burning, transfers)

### Dynamic Policy System

The policy factory automatically creates simulation policies:
- **Base Policies**: Vesting, staking, pricing (always available)
- **Points Campaign Policy**: Created when points parameters detected
- **Custom Utility Policy**: Created when utility parameters detected
- **Enhanced Staking Policy**: Created when advanced staking parameters detected

### Simulation Engine

The core simulation engine uses radCAD to model token economics:
- State variables: token supply, circulating supply, staked tokens, etc.
- Policy functions: vesting schedule, staking behavior, token price dynamics
- State update functions: update supply, staking, price, market cap
- **Enhanced with dynamic policies for client-specific mechanisms**

### Unified Simulation Approach

The simulation system now uses a unified approach:
- **Single Run** (num_runs=1): Fast execution for parameter testing and quick analysis
- **Monte Carlo Analysis** (num_runs>1): Statistical analysis with confidence intervals and percentiles
- **Automatic Detection**: UI automatically shows appropriate visualizations based on number of runs
- **No Artificial Distinction**: Removes confusion between "simulation" and "Monte Carlo simulation"

The Monte Carlo functionality extends the basic simulation:
- Runs multiple simulations with different random seeds
- Uses parallel processing for performance  
- Calculates statistical measures (mean, confidence intervals, percentiles)
- Visualizes distributions and confidence bands

### Data Input

The system expects CSV files with:
- Parameter names and initial values
- **Automatic parameter type detection** (numeric, percentage, boolean, etc.)
- **Automatic categorization** based on parameter names
- **Flexible parameter naming** (supports various CSV formats)

## Key Files

### Core Application
- **app.py**: Main entry point and UI organization with dynamic parameter controls
- **simulate.py**: TokenomicsSimulation class with dynamic policy system
- **tokenomics_data.py**: Data parsing and structure

### Parameter System
- **logic/parameter_registry.py**: Parameter discovery, categorization, and validation
- **logic/policy_factory.py**: Dynamic policy creation based on parameters
- **visualization/dynamic_components.py**: Automatic UI generation

### Enhanced Components
- **logic/data_manager.py**: Enhanced with parameter registry integration
- **logic/monte_carlo.py**: Monte Carlo simulation implementation
- **visualization/charts.py**: All plotting functions

## Error Resolution Guide

### Common Issues and Fixes

1. **Streamlit Type Errors**: Ensure all slider parameters are float type
2. **Pandas Warnings**: Use `.infer_objects(copy=False)` after fill operations
3. **Pickle Errors**: Avoid storing enum objects in data passed to radCAD
4. **Empty DataFrame Errors**: Add bounds checking before indexing operations

### Parameter Coverage Validation

The system validates parameter usage:
```python
# Check which parameters are being used
unused, missing = parameter_registry.validate_parameter_coverage(consumed_params)

# Display warnings for unused parameters
dynamic_ui.validate_and_display_coverage(consumed_params)
```

## Technical Dependencies

- Python 3.9+
- pandas 2.0.0
- streamlit 1.22.0
- plotly 5.14.1
- numpy 1.24.3
- radCAD 0.4.28

## Client Onboarding Process

1. **Upload CSV**: Client uploads their parameter CSV file
2. **Automatic Discovery**: System discovers parameter types and categories
3. **Policy Creation**: Relevant simulation policies are created automatically
4. **UI Generation**: Appropriate controls are generated for all configurable parameters
5. **Validation**: System reports parameter coverage and validates usage
6. **Simulation**: Run simulations with client-specific parameters and policies

This parameter-first approach enables scaling from one client to hundreds without any code modifications.