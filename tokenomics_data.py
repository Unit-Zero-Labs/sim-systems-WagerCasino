import pandas as pd
import numpy as np
import streamlit as st

class TokenomicsData:
    def __init__(self):
        self.static_params = {}
        self.vesting_series = pd.DataFrame()
        self.vesting_cumulative = pd.DataFrame()
        self.adoption = pd.DataFrame()
        self.staking_apr = pd.DataFrame()
        self.utility_allocations = pd.DataFrame()
        self.monthly_utility = pd.DataFrame()
        self.token_price = None
        self.market_cap = None
        self.fdv_mc = None
        self.liquidity_pool_valuation = None
        self.dates = None
        # Add new time-series data for charts
        self.token_price_series = pd.DataFrame()
        self.market_cap_series = pd.DataFrame()
        self.fdv_mc_series = pd.DataFrame()
        self.liquidity_pool_series = pd.DataFrame()

def parse_csv(file_path):
    """
    Parse the tokenomics CSV file and extract relevant data.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        TokenomicsData object containing the parsed data
    """
    try:
        # Read the CSV file without headers due to unnamed columns
        df = pd.read_csv(file_path, header=None, encoding='utf-8')
        df = df.fillna(0)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None
    
    data = TokenomicsData()
    
    # Extract dates from the third row
    date_row = df.iloc[2]
    dates = []
    for i in range(3, len(date_row)):
        if isinstance(date_row[i], str) and len(date_row[i]) > 0:
            # Convert date format from MM/YY to YYYY-MM-01
            try:
                month, year = date_row[i].split('/')
                dates.append(f"20{year}-{month}-01")
            except:
                dates.append(date_row[i])
    
    data.dates = pd.to_datetime(dates, errors='coerce')
    
    # Extract vesting series
    vesting_idx = None
    for i, row in df.iterrows():
        if row[1] == "Vesting Series / Token":
            vesting_idx = i
            break
    
    if vesting_idx is not None:
        # Fix: Select only the columns we need (bucket column + date columns)
        vesting_data = df.iloc[vesting_idx+2:vesting_idx+15, 1:len(dates)+2].copy()
        
        # Ensure we have the right number of columns
        if len(vesting_data.columns) == len(dates) + 1:
            vesting_data.columns = ['bucket'] + dates
            
            # Convert string values with commas to float
            for col in vesting_data.columns[1:]:  # Skip the bucket column
                vesting_data[col] = vesting_data[col].apply(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x)
            
            # Filter out empty rows
            vesting_data = vesting_data[vesting_data.index.notnull()]
            vesting_data = vesting_data[vesting_data['bucket'] != '0']
            vesting_data = vesting_data[vesting_data['bucket'] != 0]
            
            # Set the index to bucket
            vesting_data = vesting_data.set_index('bucket')
            
            data.vesting_series = vesting_data
        else:
            st.warning(f"Column mismatch in vesting data: {len(vesting_data.columns)} columns vs {len(dates) + 1} expected columns")
    
    # Extract vesting cumulative
    vesting_cum_idx = None
    for i, row in df.iterrows():
        if row[1] == "Vesting Series / Cumulative Tokens":
            vesting_cum_idx = i
            break
    
    if vesting_cum_idx is not None:
        # Fix: Select only the columns we need (bucket column + date columns)
        vesting_cum_data = df.iloc[vesting_cum_idx+2:vesting_cum_idx+15, 1:len(dates)+2].copy()
        
        # Ensure we have the right number of columns
        if len(vesting_cum_data.columns) == len(dates) + 1:
            vesting_cum_data.columns = ['bucket'] + dates
            
            # Convert string values with commas to float
            for col in vesting_cum_data.columns[1:]:  # Skip the bucket column
                vesting_cum_data[col] = vesting_cum_data[col].apply(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x)
            
            # Filter out empty rows
            vesting_cum_data = vesting_cum_data[vesting_cum_data.index.notnull()]
            vesting_cum_data = vesting_cum_data[vesting_cum_data['bucket'] != '0']
            vesting_cum_data = vesting_cum_data[vesting_cum_data['bucket'] != 0]
            
            # Set the index to bucket
            vesting_cum_data = vesting_cum_data.set_index('bucket')
            
            data.vesting_cumulative = vesting_cum_data
        else:
            st.warning(f"Column mismatch in vesting cumulative data: {len(vesting_cum_data.columns)} columns vs {len(dates) + 1} expected columns")
    
    # Extract adoption metrics
    adoption_idx = None
    for i, row in df.iterrows():
        if row[1] == "Adoption":
            adoption_idx = i
            break
    
    if adoption_idx is not None:
        # Fix: Select only the columns we need
        adoption_data = df.iloc[adoption_idx+1:adoption_idx+3, 1:len(dates)+2].copy()
        
        # Ensure we have the right number of columns
        if len(adoption_data.columns) == len(dates) + 1:
            adoption_data.columns = ['metric'] + dates
            adoption_data = adoption_data.set_index('metric')
            data.adoption = adoption_data
        else:
            st.warning(f"Column mismatch in adoption data: {len(adoption_data.columns)} columns vs {len(dates) + 1} expected columns")
    
    # Extract staking APR
    apr_idx = None
    for i, row in df.iterrows():
        if row[1] == "(A) Effective Token APR (based on current stake)":
            apr_idx = i
            break
    
    if apr_idx is not None:
        # Fix: Select only the columns we need
        apr_data = df.iloc[apr_idx:apr_idx+1, 2:len(dates)+2].copy()
        
        # Ensure we have the right number of columns
        if len(apr_data.columns) == len(dates):
            apr_data.columns = dates
            apr_data.index = ["Staking APR"]
            data.staking_apr = apr_data
        else:
            st.warning(f"Column mismatch in APR data: {len(apr_data.columns)} columns vs {len(dates)} expected columns")
    
    # Extract utility allocations
    utility_idx = None
    for i, row in df.iterrows():
        if row[1] == "Staking CUM.":
            utility_idx = i
            break
    
    if utility_idx is not None:
        # Fix: Select only the columns we need
        utility_data = df.iloc[utility_idx:utility_idx+5, 1:len(dates)+2].copy()
        
        # Ensure we have the right number of columns
        if len(utility_data.columns) == len(dates) + 1:
            utility_data.columns = ['utility'] + dates
            
            # Convert string values with commas to float
            for col in utility_data.columns[1:]:  # Skip the utility column
                utility_data[col] = utility_data[col].apply(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x)
            
            utility_data = utility_data.set_index('utility')
            data.utility_allocations = utility_data
        else:
            st.warning(f"Column mismatch in utility allocations data: {len(utility_data.columns)} columns vs {len(dates) + 1} expected columns")
    
    # Extract monthly utility incentives
    monthly_idx = None
    for i, row in df.iterrows():
        if row[1] == "Staking":
            monthly_idx = i
            break
    
    if monthly_idx is not None:
        # Fix: Select only the columns we need
        monthly_data = df.iloc[monthly_idx:monthly_idx+5, 1:len(dates)+2].copy()
        
        # Ensure we have the right number of columns
        if len(monthly_data.columns) == len(dates) + 1:
            monthly_data.columns = ['utility'] + dates
            
            # Convert string values with commas to float
            for col in monthly_data.columns[1:]:  # Skip the utility column
                monthly_data[col] = monthly_data[col].apply(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x)
            
            monthly_data = monthly_data.set_index('utility')
            data.monthly_utility = monthly_data
        else:
            st.warning(f"Column mismatch in monthly utility data: {len(monthly_data.columns)} columns vs {len(dates) + 1} expected columns")
    
    # Extract token price data
    # Look for token price in the CSV
    token_price_idx = None
    for i, row in df.iterrows():
        if isinstance(row[1], str) and "Token Price" in row[1]:
            token_price_idx = i
            break
    
    # If we couldn't find token price directly, we'll calculate it
    if token_price_idx is None:
        # Calculate token price based on circulating supply and market cap
        # For now, we'll use a placeholder that changes over time
        initial_price = 0.03096846847  # Initial token price
        price_series = pd.Series(index=dates)
        
        # Create a simple price model that increases over time
        for i, date in enumerate(dates):
            # Simple model: price increases by 0.5% each month
            price_series[date] = initial_price * (1 + 0.005) ** i
        
        # Store as DataFrame
        price_df = pd.DataFrame(price_series).T
        price_df.index = ["Token Price"]
        data.token_price_series = price_df
        
        # Set the initial token price
        data.token_price = initial_price
    else:
        # Extract the token price data
        price_data = df.iloc[token_price_idx:token_price_idx+1, 2:len(dates)+2].copy()
        if len(price_data.columns) == len(dates):
            price_data.columns = dates
            price_data.index = ["Token Price"]
            data.token_price_series = price_data
            data.token_price = price_data.iloc[0, 0]  # Initial token price
        else:
            st.warning(f"Column mismatch in token price data")
            # Use placeholder
            data.token_price = 0.03096846847
    
    # Extract or calculate market cap
    market_cap_idx = None
    for i, row in df.iterrows():
        if isinstance(row[1], str) and "Market Cap" in row[1]:
            market_cap_idx = i
            break
    
    if market_cap_idx is None:
        # Calculate market cap based on circulating supply and token price
        if not data.vesting_cumulative.empty and "Liquidity Pool" in data.vesting_cumulative.index:
            # Calculate circulating supply (all tokens except Liquidity Pool)
            circulating_supply = data.vesting_cumulative.drop("Liquidity Pool", errors='ignore').sum()
            
            # Calculate market cap for each date
            market_cap_series = pd.Series(index=dates)
            for i, date in enumerate(dates):
                # Market cap = circulating supply * token price
                market_cap_series[date] = circulating_supply[date] * data.token_price_series.loc["Token Price", date]
            
            # Store as DataFrame
            market_cap_df = pd.DataFrame(market_cap_series).T
            market_cap_df.index = ["Market Cap"]
            data.market_cap_series = market_cap_df
            
            # Set the initial market cap
            data.market_cap = market_cap_series[dates[0]]
        else:
            # Use placeholder
            data.market_cap = 57196305.79
            
            # Create a simple market cap model
            market_cap_series = pd.Series(index=dates)
            for i, date in enumerate(dates):
                # Simple model: market cap increases by 1% each month
                market_cap_series[date] = data.market_cap * (1 + 0.01) ** i
            
            # Store as DataFrame
            market_cap_df = pd.DataFrame(market_cap_series).T
            market_cap_df.index = ["Market Cap"]
            data.market_cap_series = market_cap_df
    else:
        # Extract the market cap data
        market_cap_data = df.iloc[market_cap_idx:market_cap_idx+1, 2:len(dates)+2].copy()
        if len(market_cap_data.columns) == len(dates):
            market_cap_data.columns = dates
            market_cap_data.index = ["Market Cap"]
            data.market_cap_series = market_cap_data
            data.market_cap = market_cap_data.iloc[0, 0]  # Initial market cap
        else:
            st.warning(f"Column mismatch in market cap data")
            # Use placeholder
            data.market_cap = 57196305.79
    
    # Extract or calculate FDV (Fully Diluted Valuation)
    fdv_idx = None
    for i, row in df.iterrows():
        if isinstance(row[1], str) and "FDV" in row[1]:
            fdv_idx = i
            break
    
    if fdv_idx is None:
        # Calculate FDV based on total supply and token price
        if not data.vesting_series.empty and "SUM" in data.vesting_series.index:
            # Get total supply
            total_supply = data.vesting_series.loc["SUM", dates[0]]
            
            # Calculate FDV for each date
            fdv_series = pd.Series(index=dates)
            for i, date in enumerate(dates):
                # FDV = total supply * token price
                fdv_series[date] = total_supply * data.token_price_series.loc["Token Price", date]
            
            # Store as DataFrame
            fdv_df = pd.DataFrame(fdv_series).T
            fdv_df.index = ["FDV MC"]
            data.fdv_mc_series = fdv_df
            
            # Set the initial FDV
            data.fdv_mc = fdv_series[dates[0]]
        else:
            # Use placeholder
            data.fdv_mc = 27500000
            
            # Create a simple FDV model
            fdv_series = pd.Series(index=dates)
            for i, date in enumerate(dates):
                # Simple model: FDV increases by 0.8% each month
                fdv_series[date] = data.fdv_mc * (1 + 0.008) ** i
            
            # Store as DataFrame
            fdv_df = pd.DataFrame(fdv_series).T
            fdv_df.index = ["FDV MC"]
            data.fdv_mc_series = fdv_df
    else:
        # Extract the FDV data
        fdv_data = df.iloc[fdv_idx:fdv_idx+1, 2:len(dates)+2].copy()
        if len(fdv_data.columns) == len(dates):
            fdv_data.columns = dates
            fdv_data.index = ["FDV MC"]
            data.fdv_mc_series = fdv_data
            data.fdv_mc = fdv_data.iloc[0, 0]  # Initial FDV
        else:
            st.warning(f"Column mismatch in FDV data")
            # Use placeholder
            data.fdv_mc = 27500000
    
    # Extract or calculate Liquidity Pool valuation
    lp_idx = None
    for i, row in df.iterrows():
        if isinstance(row[1], str) and "Liquidity Pool" in row[1] and "Valuation" in row[1]:
            lp_idx = i
            break
    
    if lp_idx is None:
        # Calculate LP valuation based on Liquidity Pool tokens and token price
        if not data.vesting_cumulative.empty and "Liquidity Pool" in data.vesting_cumulative.index:
            # Get Liquidity Pool tokens
            lp_tokens = data.vesting_cumulative.loc["Liquidity Pool"]
            
            # Calculate LP valuation for each date
            lp_series = pd.Series(index=dates)
            for i, date in enumerate(dates):
                # LP valuation = LP tokens * token price
                lp_series[date] = lp_tokens[date] * data.token_price_series.loc["Token Price", date]
            
            # Store as DataFrame
            lp_df = pd.DataFrame(lp_series).T
            lp_df.index = ["LP Valuation"]
            data.liquidity_pool_series = lp_df
            
            # Set the initial LP valuation
            data.liquidity_pool_valuation = lp_series[dates[0]]
        else:
            # Use placeholder
            data.liquidity_pool_valuation = 1637049.55
            
            # Create a simple LP valuation model
            lp_series = pd.Series(index=dates)
            for i, date in enumerate(dates):
                # Simple model: LP valuation increases by 0.7% each month
                lp_series[date] = data.liquidity_pool_valuation * (1 + 0.007) ** i
            
            # Store as DataFrame
            lp_df = pd.DataFrame(lp_series).T
            lp_df.index = ["LP Valuation"]
            data.liquidity_pool_series = lp_df
    else:
        # Extract the LP valuation data
        lp_data = df.iloc[lp_idx:lp_idx+1, 2:len(dates)+2].copy()
        if len(lp_data.columns) == len(dates):
            lp_data.columns = dates
            lp_data.index = ["LP Valuation"]
            data.liquidity_pool_series = lp_data
            data.liquidity_pool_valuation = lp_data.iloc[0, 0]  # Initial LP valuation
        else:
            st.warning(f"Column mismatch in LP valuation data")
            # Use placeholder
            data.liquidity_pool_valuation = 1637049.55
    
    # Add static parameters to static_params
    data.static_params["Token Launch Price / $"] = data.token_price
    data.static_params["Initial Token MC / $"] = data.market_cap
    data.static_params["Initial Token FDV MC / $"] = data.fdv_mc
    data.static_params["Liquidity Pool Fund Allocation / $"] = data.liquidity_pool_valuation
    
    # Try to extract the total supply from the vesting series
    if not data.vesting_series.empty and "SUM" in data.vesting_series.index:
        data.static_params["Initial Total Supply of Tokens"] = data.vesting_series.loc["SUM", dates[0]]
    
    return data 