# Monte Carlo Simulation for Tokenomics Modeling

## Overview

This feature extends the tokenomics simulation engine to support Monte Carlo simulations with multiple runs. Instead of producing a single deterministic outcome, Monte Carlo simulations run the model multiple times with different random seeds, allowing us to understand the range of possible outcomes and their probabilities.

## Key Benefits

- **Uncertainty Quantification**: Understand the range of possible outcomes for key metrics like token price and market cap.
- **Confidence Intervals**: Visualize 95% confidence intervals around mean values to assess prediction reliability.
- **Probability Distributions**: See the full distribution of possible outcomes at specific points in time.
- **Risk Assessment**: Identify potential downside risks and upside opportunities in your tokenomics design.

## How It Works

1. The simulation engine runs the model multiple times (e.g., 10, 50, or 100 runs).
2. Each run uses a different random seed, affecting stochastic elements like price volatility.
3. The results are aggregated to calculate statistical measures:
   - Mean values
   - Standard deviations
   - Confidence intervals
   - Percentiles (5th, 25th, 50th, 75th, 95th)
4. These statistics are visualized to provide insights into the range of possible outcomes.

## Using the Monte Carlo Feature

### In the Tokenomics Dashboard

1. Upload your tokenomics data CSV.
2. Go to the "Simulation" tab.
3. Set your desired parameters (Staking Share, Token Launch Price, etc.).
4. Set the "Number of Monte Carlo Runs" to a value greater than 1 (e.g., 10 for quick analysis, 50+ for more accurate results).
5. Click "Run Simulation".
6. View the results:
   - Time series chart with confidence intervals and percentile bands
   - Probability distribution at a specific timestep
   - Statistical summary

### In Code

```python
from simulate import TokenomicsSimulation
from data_parser import parse_csv

# Parse data
data = parse_csv("data_tables.csv")

# Create simulation
simulation = TokenomicsSimulation(data)

# Set parameters
params = {
    "staking_share": 0.5,
    "token_price": 0.03,
    "market_volatility": 0.2
}

# Run Monte Carlo simulation with 50 runs
results = simulation.run_simulation(params, num_runs=50)

# Access statistical measures
mean_price = results['mean']['token_price']
conf_intervals = results['conf_intervals']['token_price']
percentiles = results['percentiles']['token_price']

# Get distribution at a specific timestep
dist_at_timestep_10 = simulation.get_distribution_at_timestep(results, 'token_price', 10)
```

## Interpreting the Results

### Confidence Intervals

The 95% confidence interval represents the range within which we expect the true value to fall with 95% probability. Wider confidence intervals indicate greater uncertainty.

### Percentile Bands

- **5th Percentile**: Only 5% of simulations resulted in values below this line (downside risk).
- **25th Percentile**: 25% of simulations fell below this line.
- **75th Percentile**: 75% of simulations fell below this line.
- **95th Percentile**: 95% of simulations fell below this line (upside potential).

### Probability Distributions

The histogram and density plot show the full distribution of possible outcomes at a specific point in time. This helps assess the likelihood of different scenarios.

## Technical Implementation

The Monte Carlo simulation feature is implemented in the `TokenomicsSimulation` class with the following key components:

1. **run_simulation**: Modified to accept a `num_runs` parameter and process multiple runs.
2. **create_monte_carlo_visualization**: Helper method to prepare data for visualization.
3. **get_distribution_at_timestep**: Method to analyze the distribution at a specific timestep.

The implementation uses NumPy for statistical calculations and can be easily integrated with visualization libraries like Plotly.

## Future Enhancements

Potential future enhancements to the Monte Carlo simulation feature include:

1. **Sensitivity Analysis**: Automatically vary parameters to identify which ones have the most impact on outcomes.
2. **Scenario Optimization**: Use Monte Carlo results to find optimal parameter combinations.
3. **Value at Risk (VaR)**: Calculate financial risk metrics based on simulation results.
4. **Custom Distributions**: Allow users to specify custom probability distributions for random variables.

## Conclusion

The Monte Carlo simulation feature significantly enhances the tokenomics modeling capabilities by providing a more complete picture of possible outcomes and their likelihoods. This helps token designers make more informed decisions about tokenomics parameters and understand the potential risks and opportunities in their token design. 