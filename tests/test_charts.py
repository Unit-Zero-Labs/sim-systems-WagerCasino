"""
Tests for the charts module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.charts import (
    apply_chart_styling,
    plot_token_buckets,
    plot_price
)


class MockTokenomicsData:
    """Mock TokenomicsData class for testing."""
    
    def __init__(self):
        # Create sample dates
        self.dates = pd.date_range(start='2025-01-01', periods=12, freq='MS')
        
        # Create sample vesting data
        self.vesting_series = pd.DataFrame(
            np.random.rand(3, 12) * 1000000,
            index=['Public Sale', 'Team', 'Liquidity Pool'],
            columns=self.dates
        )
        
        # Create cumulative vesting data
        self.vesting_cumulative = self.vesting_series.cumsum(axis=1)
        
        # Create token price data
        self.token_price = 0.05
        self.token_price_series = pd.DataFrame(
            np.linspace(0.05, 0.10, 12),
            index=self.dates,
            columns=['Token Price']
        ).T
        
        # Create market cap data
        self.market_cap_series = pd.DataFrame(
            np.linspace(5000000, 10000000, 12),
            index=self.dates,
            columns=['Market Cap']
        ).T
        
        # Create FDV MC data
        self.fdv_mc_series = pd.DataFrame(
            np.linspace(10000000, 20000000, 12),
            index=self.dates,
            columns=['FDV MC']
        ).T
        
        # Create liquidity pool data
        self.liquidity_pool_series = pd.DataFrame(
            np.linspace(1000000, 2000000, 12),
            index=self.dates,
            columns=['LP Valuation']
        ).T
        
        # Create utility allocations data
        self.utility_allocations = pd.DataFrame(
            np.random.rand(5, 12) * 100000,
            index=['Staking CUM.', 'Liquidity Mining CUM.', 'Burning CUM.', 
                   'Holding CUM.', 'Transfer for Benefit CUM.'],
            columns=self.dates
        )
        
        # Create monthly utility data
        self.monthly_utility = pd.DataFrame(
            np.random.rand(5, 12) * 10000,
            index=['Staking', 'Liquidity Mining', 'Burning', 
                   'Holding', 'Transfer for Benefit'],
            columns=self.dates
        )
        
        # Create staking APR data
        self.staking_apr = pd.DataFrame(
            np.linspace(10, 5, 12),
            index=self.dates,
            columns=['Staking APR']
        ).T


class TestCharts(unittest.TestCase):
    """Tests for the charts module."""
    
    def setUp(self):
        """Set up test data."""
        self.mock_data = MockTokenomicsData()
    
    def test_apply_chart_styling(self):
        """Test that apply_chart_styling applies the expected styling."""
        import plotly.graph_objects as go
        
        # Create a simple figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        # Apply styling
        styled_fig = apply_chart_styling(fig)
        
        # Check that styling was applied
        self.assertEqual(styled_fig.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(styled_fig.layout.plot_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(styled_fig.layout.font.color, "white")
    
    def test_plot_token_buckets(self):
        """Test that plot_token_buckets creates a figure with the expected traces."""
        fig = plot_token_buckets(self.mock_data)
        
        # Check that the figure has the expected number of traces
        # (one for each bucket + one for circulating supply)
        expected_traces = len(self.mock_data.vesting_series.index) + 1
        self.assertEqual(len(fig.data), expected_traces)
        
        # Check that the figure has the expected title
        self.assertEqual(fig.layout.title.text, "Project Token Buckets & Token Supply")
    
    def test_plot_price(self):
        """Test that plot_price creates a figure with the expected trace."""
        fig = plot_price(self.mock_data)
        
        # Check that the figure has one trace for token price
        self.assertEqual(len(fig.data), 1)
        
        # Check that the trace has the expected name
        self.assertEqual(fig.data[0].name, "Price")
        
        # Check that the figure has the expected title
        self.assertEqual(fig.layout.title.text, "Token Price")


if __name__ == '__main__':
    unittest.main()