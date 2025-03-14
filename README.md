# Unit Zero Labs Tokenomics Engine

An interactive dashboard for visualizing and simulating tokenomics data by Unit Zero Labs.

## Overview

This app provides a comprehensive visualization and simulation tool for tokenomics data. After working with UZL to define basic token parameters and utility modules, it allows users to:

1. **Analyze** token distribution, price, valuations, and utility metrics over time
2. **Simulate** different tokenomics scenarios by adjusting parameters
3. **Compare** different scenarios (future feature)

## Features

### Analysis Tab
- Project Token Buckets & Token Supply
- Token Price
- Token Valuations (Market Cap and FDV)
- DEX Liquidity Valuation
- Utility Allocations
- Monthly Utility Incentives / Burnings / Transfers
- Staking APR

### Simulation Tab
- Adjust parameters like Staking Share and Token Price
- Visualize the impact on token supply
- Compare simulation results with initial values

### Scenario Analysis Tab
- Reserved for future features

## Install Steps for Reproducing

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Upload the "Data Tables" CSV file from the UZL tokenomics model (download as CSV).

3. Explore the different tabs:
   - **Analysis**: View various charts of your tokenomics data
   - **Simulation**: Adjust parameters and run simulations
   - **Scenario Analysis**: Future feature

## Data Format

The application expects a CSV file with the following structure:
- Unnamed columns with section headers
- Time-series data with dates in the format MM/YY
- Sections for vesting schedules, adoption metrics, and utility allocations

## Requirements

- Python 3.9 or higher
- pandas
- streamlit
- plotly
- numpy
- cadCAD

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

