"""
Data management for the Unit Zero Labs Tokenomics Engine.
Provides utilities for loading and processing tokenomics data.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
import io

from tokenomics_data import TokenomicsData, generate_data_from_radcad_inputs


class DataManager:
    """
    Data manager class for loading and processing tokenomics data.
    """
    
    @staticmethod
    def load_radcad_inputs(uploaded_file) -> Optional[TokenomicsData]:
        """
        Load and process radCAD inputs from uploaded file.
        
        Args:
            uploaded_file: Uploaded file from Streamlit file_uploader
            
        Returns:
            TokenomicsData object or None if processing failed
        """
        try:
            with st.spinner("Processing radCAD Inputs CSV..."):
                tokenomics_data = generate_data_from_radcad_inputs(uploaded_file)
                
                if tokenomics_data:
                    return tokenomics_data
                else:
                    st.error("Failed to process radCAD Inputs CSV. Check file format and content.")
                    return None
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    
    @staticmethod
    def validate_data(data: TokenomicsData) -> Tuple[bool, List[str]]:
        """
        Validate the tokenomics data for consistency and completeness.
        
        Args:
            data: TokenomicsData object to validate
            
        Returns:
            Tuple containing:
            - Boolean indicating if data is valid
            - List of error messages if any
        """
        errors = []
        
        # Check for required data components
        if data.dates is None or len(data.dates) == 0:
            errors.append("Date range is missing or empty")
        
        if data.vesting_series.empty:
            errors.append("Vesting series data is missing")
        
        if data.token_price is None:
            errors.append("Token price is missing")
        
        # Validate token supply and allocation consistency
        total_allocation = 0
        for bucket_name in data.vesting_series.index:
            if bucket_name != "SUM":
                bucket_total = data.vesting_series.loc[bucket_name].sum()
                total_allocation += bucket_total
        
        initial_total_supply = data.static_params.get("Initial Total Supply of Tokens")
        if initial_total_supply is not None and abs(total_allocation - initial_total_supply) > 1:
            # Allow small rounding errors (within 1 token)
            errors.append(f"Total allocation ({total_allocation}) does not match initial supply ({initial_total_supply})")
        
        # Return validation result
        return len(errors) == 0, errors
    
    @staticmethod
    def export_to_csv(data: TokenomicsData) -> Optional[str]:
        """
        Export tokenomics data to CSV format.
        
        Args:
            data: TokenomicsData object to export
            
        Returns:
            CSV string or None if export failed
        """
        try:
            # Create a dictionary to store dataframes
            dataframes = {
                "Vesting Schedule": data.vesting_series,
                "Vesting Cumulative": data.vesting_cumulative,
                "Adoption": data.adoption,
                "Staking APR": data.staking_apr,
                "Utility Allocations": data.utility_allocations,
                "Monthly Utility": data.monthly_utility,
                "Token Price": data.token_price_series,
                "Market Cap": data.market_cap_series,
                "FDV MC": data.fdv_mc_series,
                "Liquidity Pool": data.liquidity_pool_series
            }
            
            # Create a buffer to write CSV data
            buffer = io.StringIO()
            
            # Write each dataframe to the buffer with a header
            for name, df in dataframes.items():
                if not df.empty:
                    buffer.write(f"# {name}\n")
                    df.to_csv(buffer)
                    buffer.write("\n\n")
            
            # Add static parameters
            buffer.write("# Static Parameters\n")
            for key, value in data.static_params.items():
                buffer.write(f"{key},{value}\n")
            
            # Get the CSV data as a string
            return buffer.getvalue()
        except Exception as e:
            st.error(f"Error exporting data to CSV: {e}")
            return None
    
    @staticmethod
    def get_summary_statistics(data: TokenomicsData) -> Dict[str, Any]:
        """
        Calculate summary statistics for the tokenomics data.
        
        Args:
            data: TokenomicsData object
            
        Returns:
            Dictionary of summary statistics
        """
        stats = {}
        
        # Basic token supply statistics
        stats["total_supply"] = data.static_params.get("Initial Total Supply of Tokens", 0)
        
        # Initial circulating supply
        if not data.vesting_cumulative.empty and "Liquidity Pool" in data.vesting_cumulative.index:
            initial_circulating = data.vesting_cumulative.drop("Liquidity Pool", errors='ignore').sum().iloc[0]
        else:
            initial_circulating = 0
        stats["initial_circulating_supply"] = initial_circulating
        
        # Final circulating supply (at the end of the simulation period)
        if not data.vesting_cumulative.empty and "Liquidity Pool" in data.vesting_cumulative.index:
            final_circulating = data.vesting_cumulative.drop("Liquidity Pool", errors='ignore').sum().iloc[-1]
        else:
            final_circulating = 0
        stats["final_circulating_supply"] = final_circulating
        
        # Token price statistics
        stats["initial_token_price"] = data.token_price
        
        if not data.token_price_series.empty:
            stats["final_token_price"] = data.token_price_series.iloc[0, -1]
        else:
            stats["final_token_price"] = data.token_price
        
        # Market cap statistics
        if not data.market_cap_series.empty:
            stats["initial_market_cap"] = data.market_cap_series.iloc[0, 0]
            stats["final_market_cap"] = data.market_cap_series.iloc[0, -1]
        else:
            stats["initial_market_cap"] = initial_circulating * data.token_price
            stats["final_market_cap"] = final_circulating * stats["final_token_price"]
        
        # FDV statistics
        if not data.fdv_mc_series.empty:
            stats["initial_fdv"] = data.fdv_mc_series.iloc[0, 0]
            stats["final_fdv"] = data.fdv_mc_series.iloc[0, -1]
        else:
            stats["initial_fdv"] = stats["total_supply"] * data.token_price
            stats["final_fdv"] = stats["total_supply"] * stats["final_token_price"]
        
        # Calculate allocation percentages
        allocations = {}
        if not data.vesting_series.empty:
            for bucket in data.vesting_series.index:
                if bucket != "SUM":
                    bucket_total = data.vesting_series.loc[bucket].sum()
                    allocation_percent = (bucket_total / stats["total_supply"]) * 100 if stats["total_supply"] > 0 else 0
                    allocations[bucket] = {
                        "tokens": bucket_total,
                        "percent": allocation_percent
                    }
        stats["allocations"] = allocations
        
        return stats