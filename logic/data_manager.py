"""
Data management for the Unit Zero Labs Tokenomics Engine.
Provides utilities for loading and processing tokenomics data.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
import io

from tokenomics_data import TokenomicsData, generate_data_from_radcad_inputs
from logic.parameter_registry import parameter_registry


class DataManager:
    """
    Data manager class for loading and processing tokenomics data.
    Enhanced with parameter registry integration for dynamic parameter handling.
    """
    
    @staticmethod
    def load_radcad_inputs(uploaded_file) -> Optional[TokenomicsData]:
        """
        Load and process radCAD inputs from uploaded file with parameter registry integration.
        
        Args:
            uploaded_file: Uploaded file from Streamlit file_uploader
            
        Returns:
            TokenomicsData object or None if processing failed
        """
        try:
            with st.spinner("Processing radCAD Inputs CSV..."):
                # First, load the CSV to extract parameters
                df_radcad = pd.read_csv(uploaded_file)
                
                # Validate CSV structure
                if not DataManager._validate_csv_structure(df_radcad):
                    return None
                
                # Extract parameters and register them
                raw_params = DataManager._extract_parameters(df_radcad)
                parameter_registry.register_parameters_from_csv(raw_params)
                
                # Log parameter discovery
                st.info(f"📊 **Parameter Discovery**: Found {len(raw_params)} parameters across {len(set(param.category for param in parameter_registry.parameters.values()))} categories")
                
                # Display discovered capabilities
                DataManager._display_parameter_insights()
                
                # Reset file pointer and generate data
                uploaded_file.seek(0)
                tokenomics_data = generate_data_from_radcad_inputs(uploaded_file)
                
                if tokenomics_data:
                    # Store parameter registry separately to avoid pickle issues in simulations
                    # The registry is available via the global parameter_registry instance
                    return tokenomics_data
                else:
                    st.error("Failed to process radCAD Inputs CSV. Check file format and content.")
                    return None
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    
    @staticmethod
    def _validate_csv_structure(df: pd.DataFrame) -> bool:
        """
        Validate the basic structure of the CSV file.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Boolean indicating if structure is valid
        """
        required_columns = ['Parameter Name', 'Initial Value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ **Invalid CSV Structure**: Missing required columns: {', '.join(missing_columns)}")
            st.info("Expected columns: Parameter Name, Initial Value, Min, Max, Interval Steps, Unit, Comment")
            return False
            
        # Check for empty parameter names
        empty_params = df['Parameter Name'].isna().sum()
        if empty_params > 0:
            st.warning(f"⚠️ Found {empty_params} rows with empty parameter names. These will be skipped.")
        
        return True
    
    @staticmethod
    def _extract_parameters(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract parameters from CSV DataFrame.
        
        Args:
            df: CSV DataFrame
            
        Returns:
            Dictionary of parameter name to value mappings
        """
        from tokenomics_data import _parse_radcad_param
        
        params = {}
        for idx, row in df.iterrows():
            param_key = row['Parameter Name']
            raw_value = row.get('Initial Value')
            
            if param_key and pd.notna(param_key):
                param_value = _parse_radcad_param(raw_value)
                params[str(param_key).strip()] = param_value
        
        return params
    
    @staticmethod
    def _display_parameter_insights():
        """Display insights about discovered parameters."""
        from logic.parameter_registry import ParameterCategory
        
        # Count parameters by category
        category_counts = {}
        for param in parameter_registry.parameters.values():
            category = param.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Display in columns
        if category_counts:
            st.subheader("📋 Parameter Categories Detected")
            
            # Create columns for display
            categories = list(category_counts.keys())
            if len(categories) <= 3:
                cols = st.columns(len(categories))
            else:
                cols = st.columns(3)
            
            for i, (category, count) in enumerate(category_counts.items()):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    category_name = category.value.replace('_', ' ').title()
                    st.metric(category_name, count)
            
            # Highlight special categories
            special_categories = [ParameterCategory.POINTS_CAMPAIGN, ParameterCategory.AGENT_BEHAVIOR, ParameterCategory.UTILITY]
            found_special = [cat for cat in special_categories if cat in category_counts]
            
            if found_special:
                st.success(f"🎯 **Advanced Features Detected**: {', '.join([cat.value.replace('_', ' ').title() for cat in found_special])}")
    
    @staticmethod
    def validate_data(data: TokenomicsData) -> Tuple[bool, List[str]]:
        """
        Validate the tokenomics data for consistency and completeness.
        Enhanced with parameter registry validation.
        
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
        
        # Parameter registry validation - use global registry
        if parameter_registry.parameters:
            # Check for critical missing parameters
            critical_params = ['initial_total_supply', 'launch_date']
            missing_critical = [param for param in critical_params if param not in data.static_params]
            if missing_critical:
                errors.extend([f"Missing critical parameter: {param}" for param in missing_critical])
        
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
            
            # Add parameter registry information if available
            if parameter_registry.parameters:
                buffer.write("\n# Parameter Categories\n")
                for param_name, param_def in parameter_registry.parameters.items():
                    buffer.write(f"{param_name},{param_def.category.value},{param_def.param_type.value},{param_def.value}\n")
            
            # Get the CSV data as a string
            return buffer.getvalue()
        except Exception as e:
            st.error(f"Error exporting data to CSV: {e}")
            return None
    
    @staticmethod
    def get_summary_statistics(data: TokenomicsData) -> Dict[str, Any]:
        """
        Calculate summary statistics for the tokenomics data.
        Enhanced with parameter registry insights.
        
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
        
        # Parameter registry statistics - use global registry
        if parameter_registry.parameters:
            registry_stats = {
                "total_parameters": len(parameter_registry.parameters),
                "configurable_parameters": len(parameter_registry.get_configurable_parameters()),
                "discovered_policies": parameter_registry.discover_new_policies(),
                "parameter_categories": list(set(param.category.value for param in parameter_registry.parameters.values()))
            }
            stats["parameter_registry"] = registry_stats
        
        return stats
    
    @staticmethod
    def get_parameter_validation_report(data: TokenomicsData) -> Dict[str, Any]:
        """
        Generate a comprehensive parameter validation report.
        
        Args:
            data: TokenomicsData object
            
        Returns:
            Dictionary containing validation report
        """
        if not parameter_registry.parameters:
            return {"error": "No parameter registry available"}
        
        # Get all parameter names that should be consumed
        expected_consumed = list(data.static_params.keys())
        
        # Validate parameter coverage
        unused, missing = parameter_registry.validate_parameter_coverage(expected_consumed)
        
        # Categorize parameters
        params_by_category = {}
        for param_name, param_def in parameter_registry.parameters.items():
            category = param_def.category.value
            if category not in params_by_category:
                params_by_category[category] = []
            params_by_category[category].append({
                "name": param_name,
                "value": param_def.value,
                "type": param_def.param_type.value,
                "configurable": param_def.is_configurable
            })
        
        return {
            "total_parameters": len(parameter_registry.parameters),
            "unused_parameters": unused,
            "missing_parameters": missing,
            "parameters_by_category": params_by_category,
            "discovered_policies": parameter_registry.discover_new_policies(),
            "configurable_count": len(parameter_registry.get_configurable_parameters())
        }