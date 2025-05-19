import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

########################################################
####### UNIT ZERO LABS TOKEN SIMULATION ENGINE #########
########################################################


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

def _parse_radcad_param(value):
    if pd.isna(value) or str(value).strip() == '-':
        return None
    if isinstance(value, str):
        value = value.strip()
        if '%' in value:
            try:
                return float(value.replace('%', '').replace(',', '')) / 100.0
            except ValueError:
                # If percentage parsing fails, it might be a string with '%' not meant as percentage
                pass # Fall through to other parsing attempts or return as string
        try:
            return float(value.replace(',', ''))
        except ValueError:
            # Attempt to parse as date
            try:
                # Try specific format first that might be common in sheets
                dt_obj = datetime.strptime(value, '%d.%m.%Y')
                return pd.Timestamp(dt_obj).tz_localize(None) # Ensure timezone naive
            except ValueError:
                # Try more general parsing if specific format fails
                # errors='coerce' will return NaT if parsing fails, which pd.isna() can catch later
                dt_obj = pd.to_datetime(value, errors='coerce')
                if pd.notna(dt_obj):
                    return pd.Timestamp(dt_obj).tz_localize(None) # Ensure timezone naive
                return value # Return as string if all parsing fails
    elif isinstance(value, datetime): # Handle if it's already a Python datetime
        return pd.Timestamp(value).tz_localize(None)
    elif isinstance(value, pd.Timestamp): # Handle if it's already a Pandas Timestamp
        return value.tz_localize(None) if value.tzinfo is not None else value
    return value


def _get_radcad_params(df_radcad):
    params = {}
    if 'Parameter Name' not in df_radcad.columns or 'Initial Value' not in df_radcad.columns:
        st.error("radCAD input CSV must contain 'Parameter Name' and 'Initial Value' columns.")
        return None # Or raise an error

    for _, row in df_radcad.iterrows():
        param_key = row['Parameter Name']
        # Use .get to avoid KeyError if 'Initial Value' is missing for some reason, though it should be there.
        raw_value = row.get('Initial Value')
        param_value = _parse_radcad_param(raw_value)
        if param_key and pd.notna(param_key):
            params[str(param_key).strip()] = param_value
    return params

ALLOCATION_PARAM_KEYWORDS = [
    'Public Sale 1 Token Allocation',
    'Public Sale 2 Token Allocation',
    'Public Sale 3 Token Allocation',
    'Founders / Team Token Allocation',
    'Advisors Token Allocation',
    'Strategic Partners Token Allocation',
    'Community Token Allocation',
    'CEX Liquidity Token Allocation',
    'Incentivisation Token Allocation',
    'Staking Vesting Token Allocation',
    'Liquidity Pool Token Allocation',
    'Airdrop Allocation'
]

# Mapping from radCAD param name prefix to a more readable bucket name
BUCKET_NAME_MAPPING = {
    'Public Sale 1': 'Public Sale 1',
    'Public Sale 2': 'Public Sale 2',
    'Public Sale 3': 'Public Sale 3',
    'Founders / Team': 'Founders / Team',
    'Advisors': 'Advisors',
    'Strategic Partners': 'Strategic Partners',
    'Community': 'Community',
    'CEX Liquidity': 'CEX Liquidity',
    'Incentivisation': 'Incentivisation',
    'Staking Vesting': 'Staking Vesting',
    'Liquidity Pool': 'Liquidity Pool',
    'Airdrop': 'Airdrops' # Assuming Airdrop Allocation becomes 'Airdrops' bucket
}

def generate_data_from_radcad_inputs(uploaded_file):
    data = TokenomicsData()
    try:
        df_radcad = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading radCAD_inputs.csv: {e}")
        return None

    radcad_params = _get_radcad_params(df_radcad)
    if radcad_params is None: # Error handled in _get_radcad_params
        return None
    data.static_params = radcad_params # Store raw params

    # 1. Generate Date Range
    raw_launch_date = radcad_params.get('launch_date')

    if not isinstance(raw_launch_date, pd.Timestamp):
        st.error(f"Launch Date ('launch_date') not found, invalid, or could not be parsed to a date in radCAD inputs. Value was: {raw_launch_date}. Expected format like 'dd.mm.yyyy' or other standard date formats.")
        return None
    
    launch_date = raw_launch_date # Already a tz-naive pd.Timestamp from _parse_radcad_param

    simulation_horizon_years = radcad_params.get('simulation_horizon_years', 10) # Default to 10 years
    if not isinstance(simulation_horizon_years, (int, float)) or simulation_horizon_years <= 0:
        st.warning(f"Invalid 'simulation_horizon_years': {simulation_horizon_years}. Defaulting to 10 years.")
        simulation_horizon_years = 10
    num_months = int(simulation_horizon_years * 12)
    
    # Create a timezone-naive DatetimeIndex with Month Start frequency
    data.dates = pd.date_range(start=launch_date, periods=num_months, freq='MS')
    # data.dates is already tz-naive if launch_date is tz-naive.

    # 2. Vesting Schedules
    initial_total_supply = radcad_params.get('initial_total_supply')
    if initial_total_supply is None:
        st.error("Initial Total Supply not found in radCAD inputs.")
        return None

    vesting_series_data = {}
    stakeholder_details = []

    for param_name_key, bucket_name_prefix in BUCKET_NAME_MAPPING.items():
        alloc_perc = radcad_params.get(f'{param_name_key} Token Allocation', radcad_params.get(f'{param_name_key}_allocation'))
        if alloc_perc is None and param_name_key == 'Airdrop': # Special handling for Airdrop
            alloc_perc = radcad_params.get('airdrop_allocation')
        elif alloc_perc is None:
            # Attempt to find a generic key if specific one isn't present (e.g. Public Sale 1 instead of Public Sale 1 Token Allocation)
            alloc_perc = radcad_params.get(param_name_key)
        
        if alloc_perc is not None and alloc_perc > 0:
            # Parameter names in radCAD_inputs.csv might vary slightly (e.g. spaces vs underscores)
            # We'll try to be flexible with suffixes like " Initial Vesting", " Cliff Months", " Vesting Duration Months"
            initial_vest_perc_key_options = [f'{param_name_key} Initial Vesting', f'{param_name_key}_initial_vesting']
            cliff_months_key_options = [f'{param_name_key} Cliff Months', f'{param_name_key}_cliff']
            duration_months_key_options = [f'{param_name_key} Vesting Duration Months', f'{param_name_key}_vesting_duration']

            initial_vest_perc = next((radcad_params.get(k, 0) for k in initial_vest_perc_key_options if radcad_params.get(k) is not None), 0) / 100.0
            cliff_months = int(next((radcad_params.get(k, 0) for k in cliff_months_key_options if radcad_params.get(k) is not None), 0))
            vesting_duration_months = int(next((radcad_params.get(k, 0) for k in duration_months_key_options if radcad_params.get(k) is not None), 0))
            
            # Special handling for Airdrop dates if airdrop_date1, airdrop_amount1 etc. are used
            if bucket_name_prefix == 'Airdrops':
                airdrop_dates_params = []
                for i in range(1, 4): # Check for up to 3 airdrops
                    raw_date_val = radcad_params.get(f'airdrop_date{i}')
                    amount_val = radcad_params.get(f'airdrop_amount{i}') # This should be a percentage like 0.5 for 50%

                    parsed_date_val = None
                    if isinstance(raw_date_val, pd.Timestamp): # Already parsed and tz-naive
                        parsed_date_val = raw_date_val
                    elif raw_date_val is not None: # If not None and not Timestamp, it means parsing failed or it's not a date
                        st.warning(f"Airdrop date {raw_date_val} (for 'airdrop_date{i}') could not be parsed into a valid date. Skipping this airdrop event.")

                    # Amount should be a float (percentage)
                    parsed_amount_val = None
                    if isinstance(amount_val, (float, int)) and amount_val > 0:
                         parsed_amount_val = amount_val # Assumes it's already a fraction (e.g. 0.5 for 50%)
                    elif amount_val is not None:
                        st.warning(f"Airdrop amount {amount_val} (for 'airdrop_amount{i}') is not a valid positive number. Skipping this airdrop event.")


                    if parsed_date_val and parsed_amount_val:
                        airdrop_dates_params.append({'date': parsed_date_val, 'amount_perc': parsed_amount_val})
                    elif (raw_date_val or amount_val) and not (parsed_date_val and parsed_amount_val) : # If either was provided but parsing/validation failed for one or both
                        if raw_date_val: st.warning(f"Details for airdrop event {i} (date: {raw_date_val}, amount: {amount_val}) were incomplete or invalid after parsing. Skipping.")

                if not airdrop_dates_params and any(radcad_params.get(f'airdrop_date{j}') or radcad_params.get(f'airdrop_amount{j}') for j in range(1,4)):
                    st.info(f"Some airdrop parameters were found for '{bucket_name_prefix}', but no complete airdrop events could be scheduled.")
                
                # Add to stakeholder_details only if there are valid airdrop events
                if airdrop_dates_params:
                    stakeholder_details.append({
                        'name': bucket_name_prefix,
                        'total_tokens': initial_total_supply * alloc_perc,
                        'is_airdrop': True,
                        'airdrop_events': airdrop_dates_params
                    })
                elif alloc_perc > 0 : # If allocation exists but no valid airdrop events configured
                     st.warning(f"Allocation for '{bucket_name_prefix}' exists ({alloc_perc*100}%), but no valid airdrop dates/amounts ('airdrop_dateX', 'airdrop_amountX') were configured or parsed correctly. These tokens will not be distributed as airdrops.")

            else: # Not an airdrop, regular vesting
                stakeholder_details.append({
                    'name': bucket_name_prefix,
                    'total_tokens': initial_total_supply * alloc_perc,
                    'initial_vest_perc': initial_vest_perc,
                    'cliff_months': cliff_months,
                    'vesting_duration_months': vesting_duration_months,
                    'is_airdrop': False
                })

    monthly_releases = np.zeros((len(stakeholder_details), len(data.dates)))

    for i, stakeholder in enumerate(stakeholder_details):
        total_stakeholder_tokens = stakeholder['total_tokens']
        
        if stakeholder['is_airdrop']:
            for event in stakeholder['airdrop_events']:
                event_date_ts = event['date'] # This is a pd.Timestamp, tz-naive
                event_amount_perc = event['amount_perc'] # This is a float e.g. 0.5

                if pd.isna(event_date_ts): # Explicitly check for NaT
                    st.warning(f"Airdrop date for {stakeholder['name']} is invalid (NaT). Skipping this airdrop event.")
                    continue

                # Align event_date_ts to the start of its month to match data.dates index
                try:
                    # data.dates is already tz-naive and 'MS' frequency.
                    aligned_event_date = event_date_ts.floor('MS')
                except Exception as e:
                    st.error(f"Error aligning airdrop date {event_date_ts} for {stakeholder['name']} using .floor('MS'): {e}. Skipping this event.")
                    continue

                try:
                    if not (data.dates[0] <= aligned_event_date <= data.dates[-1]):
                        st.warning(f"Airdrop date {event_date_ts.strftime('%Y-%m-%d')} for {stakeholder['name']} (aligns to {aligned_event_date.strftime('%Y-%m-%d')}) is outside the simulation period ({data.dates[0].strftime('%Y-%m-%d')} to {data.dates[-1].strftime('%Y-%m-%d')}). Skipping.")
                        continue
                    
                    # Get the integer location of the aligned date in our DatetimeIndex
                    event_month_index = data.dates.get_loc(aligned_event_date)
                    monthly_releases[i, event_month_index] += total_stakeholder_tokens * event_amount_perc
                except KeyError:
                    # This should be rare if the date is within range and data.dates is a complete MS-frequency index
                    st.warning(f"Airdrop date {event_date_ts.strftime('%Y-%m-%d')} for {stakeholder['name']} (aligns to {aligned_event_date.strftime('%Y-%m-%d')}) could not be precisely located in the simulation's date index. This may indicate an unexpected issue. Skipping this event.")
                except Exception as e:
                    st.error(f"An unexpected error occurred while processing airdrop for {stakeholder['name']} on {event_date_ts.strftime('%Y-%m-%d')}: {e}")
        else:
            initial_vest_tokens = total_stakeholder_tokens * stakeholder['initial_vest_perc']
            cliff_months = stakeholder['cliff_months']
            vesting_duration_months = stakeholder['vesting_duration_months']
            
            # TGE release (Month 0 if cliff is 0, or first month after launch)
            if cliff_months == 0:
                 monthly_releases[i, 0] += initial_vest_tokens
                 remaining_tokens = total_stakeholder_tokens - initial_vest_tokens
                 start_vesting_month_idx = 0 # Vesting starts from month 0 if no TGE or TGE is part of linear
            else: # Cliff > 0
                 if initial_vest_tokens > 0:
                     monthly_releases[i, 0] += initial_vest_tokens # TGE happens at launch_date (month 0)
                 remaining_tokens = total_stakeholder_tokens - initial_vest_tokens
                 start_vesting_month_idx = cliff_months # Linear vesting starts after the cliff

            if vesting_duration_months > 0 and remaining_tokens > 0:
                tokens_per_month_linear = remaining_tokens / vesting_duration_months
                for m_idx in range(vesting_duration_months):
                    current_month_abs_idx = start_vesting_month_idx + m_idx
                    if current_month_abs_idx < len(data.dates):
                        monthly_releases[i, current_month_abs_idx] += tokens_per_month_linear
            elif remaining_tokens > 0 and vesting_duration_months == 0 and cliff_months > 0 : # All remaining released after cliff if duration is 0
                 if cliff_months < len(data.dates):
                    monthly_releases[i, cliff_months] += remaining_tokens
            elif remaining_tokens > 0 and vesting_duration_months == 0 and cliff_months == 0 and initial_vest_tokens < total_stakeholder_tokens:
                 # If no cliff, no duration, but not fully vested at TGE -> release rest in month 0
                 monthly_releases[i, 0] += remaining_tokens
    
    bucket_names = [s['name'] for s in stakeholder_details]
    data.vesting_series = pd.DataFrame(monthly_releases, index=bucket_names, columns=data.dates)
    data.vesting_cumulative = data.vesting_series.cumsum(axis=1)

    # 3. Adoption Metrics (Placeholder)
    data.adoption = pd.DataFrame(index=['Product Users', 'Token Holders'], columns=data.dates).fillna(0)
    # Example: simple linear growth for Product Users
    initial_pu = radcad_params.get('initial_product_users', 0)
    final_pu = radcad_params.get('product_users_after_10y', initial_pu if initial_pu is not None else 0) # Ensure final_pu has a fallback
    if initial_pu is None: initial_pu = 0
    if final_pu is None: final_pu = initial_pu

    if len(data.dates) > 0:
        data.adoption.loc['Product Users'] = np.linspace(initial_pu, final_pu, len(data.dates))
    # Example: simple linear growth for Token Holders
    initial_th = radcad_params.get('initial_token_holders', 0)
    final_th = radcad_params.get('token_holders_after_10y', initial_th if initial_th is not None else 0) # Ensure final_th has a fallback
    if initial_th is None: initial_th = 0
    if final_th is None: final_th = initial_th

    if len(data.dates) > 0:
        data.adoption.loc['Token Holders'] = np.linspace(initial_th, final_th, len(data.dates))


    # 4. Staking APR (Placeholder)
    # Use 'Liquidity Mining APR/% from radCAD, or a default if not present
    staking_apr_val = radcad_params.get('liquidity_mining_APR', 0.10) # Default to 10% if not found
    if staking_apr_val is None or not isinstance(staking_apr_val, (float,int)): 
        st.warning(f"Invalid or missing 'liquidity_mining_APR' ({staking_apr_val}). Defaulting to 10%.")
        staking_apr_val = 0.10
    
    data.staking_apr = pd.DataFrame([staking_apr_val * 100] * len(data.dates), index=data.dates, columns=['Staking APR']).T
    # Ensure columns are DatetimeIndex if they were meant to be dates.
    # For staking_apr, index is ['Staking APR'] and columns are data.dates
    data.staking_apr.columns = data.dates


    # 5. Utility Allocations (Placeholders)
    # These need to be derived based on rules for how tokens flow into utility categories
    utility_categories = ['Staking', 'Liquidity Mining', 'Burning', 'Holding', 'Transfer for Benefit']
    data.monthly_utility = pd.DataFrame(0, index=utility_categories, columns=data.dates)
    data.utility_allocations = pd.DataFrame(0, index=[f'{cat} CUM.' for cat in utility_categories], columns=data.dates)
    # Example: allocate a fixed portion of "Community" tokens to "Staking" monthly_utility
    if 'Community' in data.vesting_series.index and 'Staking' in data.monthly_utility.index:
        community_monthly_release = data.vesting_series.loc['Community']
        # Assume 50% of monthly community release goes to staking incentives
        data.monthly_utility.loc['Staking'] = community_monthly_release * 0.5 
        data.utility_allocations.loc['Staking CUM.'] = data.monthly_utility.loc['Staking'].cumsum()


    # 6. Financial Time Series (Placeholders/Simple Models)
    # Token Price:
    public_sale_valuation = radcad_params.get('public_sale_valuation')
    public_sale_supply_perc = radcad_params.get('public_sale_supply_perc') # e.g., 0.05 for 5%
    initial_token_price = radcad_params.get('initial_token_price', 0.03) # Allow direct override, else default

    if initial_token_price is None: initial_token_price = 0.03 # Default if key exists but value is None

    # Attempt to calculate from public sale if not directly provided or if direct value is default placeholder
    if initial_token_price == 0.03 and public_sale_valuation is not None and public_sale_supply_perc is not None and initial_total_supply is not None and public_sale_supply_perc > 0 and initial_total_supply > 0:
        public_sale_tokens = initial_total_supply * public_sale_supply_perc
        if public_sale_tokens > 0:
            calculated_price = public_sale_valuation / public_sale_tokens
            if calculated_price > 0: # Use calculated price if valid
                initial_token_price = calculated_price
            else:
                st.warning(f"Calculated initial token price from public sale params was not positive ({calculated_price}). Using default/provided initial_token_price: {initial_token_price}")
        else:
            st.warning(f"Public sale supply percentage or initial total supply resulted in zero public sale tokens. Cannot calculate price. Using default/provided initial_token_price: {initial_token_price}")
    elif initial_token_price == 0.03 and (public_sale_valuation is None or public_sale_supply_perc is None):
        st.info(f"Public sale valuation/supply parameters not fully provided. Using default/provided initial_token_price: {initial_token_price}")


    data.token_price = initial_token_price # Store scalar initial price
    
    # Price Growth Model - configurable via radCAD params or simple default
    monthly_price_growth_rate = radcad_params.get('monthly_price_growth_rate', 0.005) # Default 0.5%
    if monthly_price_growth_rate is None or not isinstance(monthly_price_growth_rate, (float,int)):
        st.warning(f"Invalid 'monthly_price_growth_rate' ({monthly_price_growth_rate}). Defaulting to 0.5%.")
        monthly_price_growth_rate = 0.005

    price_series = [initial_token_price * (1 + monthly_price_growth_rate) ** i for i in range(len(data.dates))]
    data.token_price_series = pd.DataFrame(price_series, index=data.dates, columns=['Token Price']).T
    data.token_price_series.columns = data.dates # Ensure columns are dates

    # Market Cap:
    # Circulating supply = sum of all cumulative vested tokens, EXCLUDING 'Liquidity Pool' (common definition)
    if not data.vesting_cumulative.empty:
        circulating_supply_df = data.vesting_cumulative.drop('Liquidity Pool', axis=0, errors='ignore').sum(axis=0)
        # Ensure consistent indexing for multiplication
        token_prices_for_calc = data.token_price_series.loc['Token Price'].reindex(circulating_supply_df.index).fillna(method='ffill').fillna(method='bfill')
        market_cap_values = circulating_supply_df.values * token_prices_for_calc.values
        data.market_cap_series = pd.DataFrame(market_cap_values, index=data.dates, columns=['Market Cap']).T
        data.market_cap_series.columns = data.dates
    else:
        st.warning("Vesting cumulative data is empty. Market cap series cannot be calculated accurately.")
        data.market_cap_series = pd.DataFrame(0, index=['Market Cap'], columns=data.dates) # Fallback

    # FDV MC:
    if initial_total_supply is not None and initial_total_supply > 0:
        token_prices_for_fdv = data.token_price_series.loc['Token Price'].reindex(data.dates).fillna(method='ffill').fillna(method='bfill')
        fdv_mc_values = initial_total_supply * token_prices_for_fdv.values
        data.fdv_mc_series = pd.DataFrame(fdv_mc_values, index=data.dates, columns=['FDV MC']).T
        data.fdv_mc_series.columns = data.dates
    else:
        st.warning("Initial total supply is not valid. FDV MC series cannot be calculated accurately.")
        data.fdv_mc_series = pd.DataFrame(0, index=['FDV MC'], columns=data.dates) # Fallback


    # Liquidity Pool Valuation:
    lp_initial_tokens_perc = radcad_params.get('Liquidity Pool Token Allocation') # This is total allocation %
    lp_valuation = 0
    
    if lp_initial_tokens_perc is not None and initial_total_supply is not None and initial_total_supply > 0:
        # Assume all LP tokens are available at TGE from its allocation for initial valuation
        lp_tokens_at_tge = initial_total_supply * lp_initial_tokens_perc
        lp_valuation = lp_tokens_at_tge * initial_token_price # Initial valuation
    else:
        st.warning("Liquidity Pool token allocation or initial total supply missing. Initial LP valuation may be inaccurate.")

    data.liquidity_pool_valuation = lp_valuation # Store initial scalar value

    # For simplicity, LP valuation series grows with token price (token side of LP)
    # A proper model would consider both assets in the pool and impermanent loss.
    if initial_token_price > 0:
        lp_series_values = lp_valuation * (data.token_price_series.loc['Token Price'].values / initial_token_price)
    else: # Avoid division by zero if initial_token_price is 0
        lp_series_values = np.zeros(len(data.dates))
        if lp_valuation != 0 : # If LP valuation was non-zero but price is zero, warn.
             st.warning("Initial token price is zero, but LP valuation was non-zero. LP valuation series will be zero.")

    data.liquidity_pool_series = pd.DataFrame(lp_series_values, index=data.dates, columns=['LP Valuation']).T
    data.liquidity_pool_series.columns = data.dates
    
    # Add key initial scalar values to static_params for display or easy access
    data.static_params['Calculated Initial Token Price / $'] = data.token_price
    if not data.market_cap_series.empty:
        data.static_params['Calculated Initial Token MC / $'] = data.market_cap_series.iloc[0,0] if len(data.market_cap_series.columns) > 0 else 0
    if not data.fdv_mc_series.empty:
        data.static_params['Calculated Initial Token FDV MC / $'] = data.fdv_mc_series.iloc[0,0] if len(data.fdv_mc_series.columns) > 0 else 0
    data.static_params['Calculated Initial Liquidity Pool Fund Allocation / $'] = data.liquidity_pool_valuation
    data.static_params['Initial Total Supply of Tokens'] = initial_total_supply
    
    return data 