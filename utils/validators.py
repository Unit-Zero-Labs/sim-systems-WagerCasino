"""
Validation utilities for the Unit Zero Labs Tokenomics Engine.
Provides functions for validating user inputs and data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import re
from datetime import datetime


def validate_simulation_params(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate simulation parameters.
    
    Args:
        params: Dictionary of simulation parameters
        
    Returns:
        Tuple containing:
        - Boolean indicating if parameters are valid
        - List of error messages if any
    """
    errors = []
    
    # Validate staking_share
    if 'staking_share' in params:
        staking_share = params['staking_share']
        if not isinstance(staking_share, (int, float)):
            errors.append("Staking share must be a number")
        elif staking_share < 0 or staking_share > 1:
            errors.append("Staking share must be between 0 and 1")
    
    # Validate token_price
    if 'token_price' in params:
        token_price = params['token_price']
        if not isinstance(token_price, (int, float)):
            errors.append("Token price must be a number")
        elif token_price <= 0:
            errors.append("Token price must be greater than 0")
    
    # Validate staking_apr_multiplier
    if 'staking_apr_multiplier' in params:
        staking_apr_multiplier = params['staking_apr_multiplier']
        if not isinstance(staking_apr_multiplier, (int, float)):
            errors.append("Staking APR multiplier must be a number")
        elif staking_apr_multiplier <= 0:
            errors.append("Staking APR multiplier must be greater than 0")
    
    # Validate market_volatility
    if 'market_volatility' in params:
        market_volatility = params['market_volatility']
        if not isinstance(market_volatility, (int, float)):
            errors.append("Market volatility must be a number")
        elif market_volatility < 0 or market_volatility > 1:
            errors.append("Market volatility must be between 0 and 1")
    
    return len(errors) == 0, errors


def validate_monte_carlo_params(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate Monte Carlo simulation parameters.
    
    Args:
        params: Dictionary of Monte Carlo parameters
        
    Returns:
        Tuple containing:
        - Boolean indicating if parameters are valid
        - List of error messages if any
    """
    errors = []
    
    # Validate num_runs
    if 'num_runs' in params:
        num_runs = params['num_runs']
        if not isinstance(num_runs, int):
            errors.append("Number of runs must be an integer")
        elif num_runs <= 0:
            errors.append("Number of runs must be greater than 0")
        elif num_runs > 200:  # Arbitrary upper limit for safety
            errors.append("Number of runs cannot exceed 200 (for performance reasons)")
    
    # Boolean parameters don't need validation (show_confidence_intervals, show_percentiles)
    
    return len(errors) == 0, errors


def validate_radcad_csv_content(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the content of a radCAD inputs CSV file.
    
    Args:
        df: DataFrame containing the CSV content
        
    Returns:
        Tuple containing:
        - Boolean indicating if the content is valid
        - List of error messages if any
    """
    errors = []
    
    # Check for required columns
    required_columns = ['Parameter Name', 'Initial Value']
    for column in required_columns:
        if column not in df.columns:
            errors.append(f"Required column '{column}' is missing from the CSV")
    
    if errors:
        return False, errors
    
    # Check for required parameters
    required_params = [
        'initial_total_supply',
        'launch_date'
    ]
    
    param_names = df['Parameter Name'].tolist()
    for param in required_params:
        if not any(param in str(name) for name in param_names):
            errors.append(f"Required parameter '{param}' is missing from the CSV")
    
    # Check for data type consistency in critical parameters
    try:
        # Initial Total Supply must be a number
        total_supply_row = df[df['Parameter Name'].str.contains('initial_total_supply', case=False, na=False)]
        if not total_supply_row.empty:
            total_supply_value = total_supply_row['Initial Value'].iloc[0]
            # Clean up the value by removing commas and converting to float
            try:
                cleaned_value = str(total_supply_value).replace(',', '')
                float(cleaned_value)
            except ValueError:
                errors.append(f"Initial Total Supply value '{total_supply_value}' is not a valid number")
        
        # Launch Date must be a valid date
        launch_date_row = df[df['Parameter Name'].str.contains('launch_date', case=False, na=False)]
        if not launch_date_row.empty:
            launch_date_value = launch_date_row['Initial Value'].iloc[0]
            # Try to parse as date (assuming common formats)
            date_formats = ['%d.%m.%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
            valid_date = False
            for date_format in date_formats:
                try:
                    datetime.strptime(str(launch_date_value), date_format)
                    valid_date = True
                    break
                except ValueError:
                    continue
            
            if not valid_date:
                errors.append(f"Launch Date value '{launch_date_value}' is not a valid date format")
    
    except Exception as e:
        errors.append(f"Error validating CSV content: {e}")
    
    return len(errors) == 0, errors


def validate_decimal_number(value: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> Tuple[bool, str]:
    """
    Validate that a string represents a valid decimal number within an optional range.
    
    Args:
        value: String to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Returns:
        Tuple containing:
        - Boolean indicating if the value is valid
        - Error message if invalid, empty string if valid
    """
    # Check if empty
    if not value:
        return False, "Value cannot be empty"
    
    # Try to convert to float
    try:
        num_value = float(value)
    except ValueError:
        return False, "Not a valid number"
    
    # Check range if specified
    if min_value is not None and num_value < min_value:
        return False, f"Value must be at least {min_value}"
    
    if max_value is not None and num_value > max_value:
        return False, f"Value must not exceed {max_value}"
    
    return True, ""


def validate_integer(value: str, min_value: Optional[int] = None, max_value: Optional[int] = None) -> Tuple[bool, str]:
    """
    Validate that a string represents a valid integer within an optional range.
    
    Args:
        value: String to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Returns:
        Tuple containing:
        - Boolean indicating if the value is valid
        - Error message if invalid, empty string if valid
    """
    # Check if empty
    if not value:
        return False, "Value cannot be empty"
    
    # Try to convert to integer
    try:
        num_value = int(value)
    except ValueError:
        return False, "Not a valid integer"
    
    # Check range if specified
    if min_value is not None and num_value < min_value:
        return False, f"Value must be at least {min_value}"
    
    if max_value is not None and num_value > max_value:
        return False, f"Value must not exceed {max_value}"
    
    return True, ""


def validate_date_string(value: str, format_str: str = '%d.%m.%Y') -> Tuple[bool, str]:
    """
    Validate that a string represents a valid date in the specified format.
    
    Args:
        value: String to validate
        format_str: Date format string (default: %d.%m.%Y)
        
    Returns:
        Tuple containing:
        - Boolean indicating if the value is valid
        - Error message if invalid, empty string if valid
    """
    # Check if empty
    if not value:
        return False, "Date cannot be empty"
    
    # Try to parse as date
    try:
        datetime.strptime(value, format_str)
        return True, ""
    except ValueError:
        return False, f"Not a valid date in format {format_str}"


def validate_percentage(value: str, allow_zero: bool = True) -> Tuple[bool, str]:
    """
    Validate that a string represents a valid percentage (0-100).
    
    Args:
        value: String to validate
        allow_zero: Whether to allow zero as a valid percentage
        
    Returns:
        Tuple containing:
        - Boolean indicating if the value is valid
        - Error message if invalid, empty string if valid
    """
    # Check if empty
    if not value:
        return False, "Percentage cannot be empty"
    
    # Remove % sign if present
    cleaned_value = value.strip()
    if cleaned_value.endswith('%'):
        cleaned_value = cleaned_value[:-1].strip()
    
    # Try to convert to float
    try:
        num_value = float(cleaned_value)
    except ValueError:
        return False, "Not a valid percentage"
    
    # Check range
    if not allow_zero and num_value == 0:
        return False, "Percentage must be greater than 0"
    
    if num_value < 0:
        return False, "Percentage cannot be negative"
    
    if num_value > 100:
        return False, "Percentage cannot exceed 100"
    
    return True, ""


def format_error_message(errors: List[str]) -> str:
    """
    Format a list of error messages into a single string.
    
    Args:
        errors: List of error messages
        
    Returns:
        Formatted error message string
    """
    if not errors:
        return ""
    
    if len(errors) == 1:
        return errors[0]
    
    return "• " + "\n• ".join(errors)