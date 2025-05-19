"""
Unit Zero Labs Tokenomics Engine.

This application provides a comprehensive simulation and visualization tool for tokenomics data.
"""

import streamlit as st
from tokenomics_data import TokenomicsData, generate_data_from_radcad_inputs
from simulate import TokenomicsSimulation
from logic.state_manager import StateManager
from logic.data_manager import DataManager
from logic.monte_carlo import MonteCarloSimulator
from utils.config import get_ui_config
from utils.error_handler import ErrorHandler
from utils.cache import st_cache_data
from visualization.styles import apply_custom_css, set_page_config
from visualization.components import (
    create_header,
    create_plot_container,
    display_simulation_parameter_controls,
    display_monte_carlo_controls,
    display_single_run_results,
    display_monte_carlo_results
)
from visualization.charts import (
    plot_token_buckets,
    plot_price,
    plot_valuations,
    plot_dex_liquidity,
    plot_utility_allocations,
    plot_monthly_utility,
    plot_staking_apr
)


def main():
    """Main function to run the Streamlit app."""
    
    # Set page config and apply custom CSS
    set_page_config()
    apply_custom_css()
    
    # Initialize session state
    StateManager.initialize_state()
    
    # Get UI configuration
    ui_config = get_ui_config()
    
    # Create header with logos
    create_header(
        logo_path=ui_config["logo_path"],
        title=ui_config["product_name"]
    )
    
    # Create tabs
    tab_load_data, tab_data_tables, tab_simulation, tab_scenario_analysis = st.tabs([
        "Load Data", "Data Tables", "Simulation", "Scenario Analysis"
    ])
    
    with tab_load_data:
        st.header("Load radCAD Inputs")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload radCAD Inputs CSV", type="csv", key="radcad_uploader_main")
        
        if uploaded_file is not None:
            # Check if the uploaded file is different from the one already processed
            if StateManager.get_uploaded_file_name() != uploaded_file.name:
                # Process the file
                data = DataManager.load_radcad_inputs(uploaded_file)
                if data:
                    StateManager.set_data(data, uploaded_file.name)
                    st.success("Successfully processed radCAD Inputs CSV!")
                else:
                    st.error("Failed to process radCAD Inputs CSV. Check file format and content.")
                    StateManager.clear_data()
            else:
                st.info(f"Using previously loaded data from '{uploaded_file.name}'. To reload, upload a different file or clear session.")
        else:
            st.info("Please upload your radCAD Inputs CSV file to begin.")
            # Clear session state if no file is uploaded
            if StateManager.get_uploaded_file_name() is not None:
                StateManager.clear_data()
    
    with tab_data_tables:
        st.header("Data Tables")
        data = StateManager.get_data()
        
        if data is not None:
            # Display charts
            create_plot_container(plot_token_buckets, data)
            create_plot_container(plot_price, data)
            create_plot_container(plot_valuations, data)
            create_plot_container(plot_dex_liquidity, data)
            create_plot_container(plot_utility_allocations, data)
            create_plot_container(plot_monthly_utility, data)
            create_plot_container(plot_staking_apr, data)
        else:
            st.info("Please upload and process a radCAD Inputs CSV in the 'Load Data' tab to view data tables.")
    
    with tab_simulation:
        st.header("Simulation")
        data_for_simulation = StateManager.get_data()
        
        if data_for_simulation is not None:
            # Create a TokenomicsSimulation instance
            simulation = TokenomicsSimulation(data_for_simulation)
            
            # Display simulation parameter controls
            params = display_simulation_parameter_controls()
            
            # Display Monte Carlo controls
            enable_monte_carlo, num_runs, show_confidence_intervals, show_percentiles = display_monte_carlo_controls()
            
            # Store parameters in session state
            StateManager.set_simulation_params(params)
            StateManager.set_monte_carlo_enabled(enable_monte_carlo)
            StateManager.set_monte_carlo_params({
                "num_runs": num_runs,
                "show_confidence_intervals": show_confidence_intervals,
                "show_percentiles": show_percentiles
            })
            
            # Run button
            if st.button("Run Simulation"):
                # Get parameters from session state
                params = StateManager.get_simulation_params()
                enable_monte_carlo = StateManager.get_monte_carlo_enabled()
                mc_params = StateManager.get_monte_carlo_params()
                
                if enable_monte_carlo:
                    # Run Monte Carlo simulation with progress bar
                    with st.spinner(f"Running Monte Carlo simulation with {mc_params['num_runs']} runs..."):
                        try:
                            mc_simulator = MonteCarloSimulator(simulation)
                            sim_result = MonteCarloSimulator.run_monte_carlo_with_progress(
                                mc_simulator, 
                                params, 
                                mc_params['num_runs']
                            )
                            StateManager.set_simulation_result(sim_result)
                        except Exception as e:
                            ErrorHandler.show_error("Error in Monte Carlo simulation", str(e))
                else:
                    # Run single simulation
                    with st.spinner("Running simulation..."):
                        try:
                            sim_result = simulation.run_simulation(params, num_runs=1)
                            StateManager.set_simulation_result(sim_result)
                        except Exception as e:
                            ErrorHandler.show_error("Error in simulation", str(e))
            
            # Display simulation results if available
            sim_result = StateManager.get_simulation_result()
            if sim_result is not None:
                enable_monte_carlo = StateManager.get_monte_carlo_enabled()
                mc_params = StateManager.get_monte_carlo_params()
                
                if enable_monte_carlo:
                    # Display Monte Carlo results
                    display_monte_carlo_results(
                        sim_result, 
                        show_confidence_intervals=mc_params['show_confidence_intervals'],
                        show_percentiles=mc_params['show_percentiles']
                    )
                else:
                    # Display single run results
                    display_single_run_results(sim_result)
            elif not StateManager.is_simulation_result_displayed():
                # Initial run when app loads if no simulation has been run yet
                with st.spinner("Running initial simulation..."):
                    try:
                        sim_result = simulation.run_simulation(params, num_runs=1)
                        display_single_run_results(sim_result)
                        StateManager.set_simulation_result(sim_result)
                    except Exception as e:
                        ErrorHandler.show_error("Error in initial simulation", str(e))
        else:
            st.info("Please upload and process a radCAD Inputs CSV in the 'Load Data' tab to run simulations.")
    
    with tab_scenario_analysis:
        st.header("Scenario Analysis")
        data_for_scenario = StateManager.get_data()
        
        if data_for_scenario is not None:
            st.write("Scenario analysis will be available in a future update.")
            st.info("This feature allows you to define and compare multiple scenarios based on the uploaded radCAD inputs.")
        else:
            st.info("Please upload and process a radCAD Inputs CSV in the 'Load Data' tab for scenario analysis.")


if __name__ == "__main__":
    # Wrap the main function with error handling
    try:
        main()
    except Exception as e:
        ErrorHandler.log_error(e, "main")
        st.error(f"An unexpected error occurred: {str(e)}")