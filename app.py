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
from logic.parameter_registry import parameter_registry
from visualization.dynamic_components import create_dynamic_ui_generator
from utils.config import get_ui_config
from utils.error_handler import ErrorHandler
from utils.cache import st_cache_data
from visualization.styles import apply_custom_css, set_page_config
from visualization.components import (
    create_header,
    create_plot_container,
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
    plot_staking_apr,
    plot_time_series_from_simulator
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
    tab_load_data, tab_data_tables, tab_simulation = st.tabs([
        "Load Data", "Data Tables", "Simulation"
    ])
    
    with tab_load_data:
        st.header("Load radCAD Inputs")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload radCAD Inputs CSV", type="csv", key="radcad_uploader_main")
        
        if uploaded_file is not None:
            # Check if the uploaded file is different from the one already processed
            if StateManager.get_uploaded_file_name() != uploaded_file.name:
                # Process the file with enhanced parameter discovery
                data = DataManager.load_radcad_inputs(uploaded_file)
                if data:
                    StateManager.set_data(data, uploaded_file.name)
                    st.success("✅ **Data Uploaded Successfully**")
                    
                    # Display parameter validation report
                    validation_report = DataManager.get_parameter_validation_report(data)
                    if "error" not in validation_report:
                        with st.expander("📊 Parameter Analysis Report", expanded=False):
                            st.write(f"**Total Parameters Loaded:** {validation_report['total_parameters']}")
                            st.write(f"**Configurable Parameters:** {validation_report['configurable_count']}")
                            st.write(f"**Discovered Policies:** {', '.join(validation_report['discovered_policies']) if validation_report['discovered_policies'] else 'Standard policies only'}")
                            
                            if validation_report['parameters_by_category']:
                                st.write("**Parameters by Category:**")
                                for category, params in validation_report['parameters_by_category'].items():
                                    if params:  # Only show categories with parameters
                                        st.write(f"  • {category.replace('_', ' ').title()}: {len(params)} parameters")
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
        st.header("Tokenomics Simulation")
        data_for_simulation = StateManager.get_data()
        
        if data_for_simulation is not None:
            # Create a TokenomicsSimulation instance
            simulation = TokenomicsSimulation(data_for_simulation)
            
            st.write("""
            This unified simulation engine automatically adapts to your CSV parameters and includes support for:
            • **Standard Tokenomics**: Vesting, staking, pricing, utility mechanisms
            • **Agent Behavior**: Individual trader behaviors and market dynamics  
            • **Points Campaigns**: Off-chain systems with conversion mechanics
            • **Monte Carlo Analysis**: Statistical analysis across multiple simulation runs
            """)
            
            # Display dynamic parameter controls based on loaded CSV
            dynamic_ui = create_dynamic_ui_generator(parameter_registry)
            
            # Show discovered policies
            discovered_policies = dynamic_ui.discover_and_display_new_policies()
            
            # Display parameter controls
            params = dynamic_ui.display_parameter_controls(key_prefix="unified")
            
            # Display simulation mode selection
            with st.expander("🎛️ Simulation Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Simulation approach selection
                    simulation_approach = st.selectbox(
                        "Simulation Approach",
                        ["Standard Stochastic", "Agent-Based Modeling"],
                        help="Choose how to model market behavior",
                        key="simulation_approach"
                    )
                    
                    if simulation_approach == "Agent-Based Modeling":
                        st.info("💡 Agent behavior parameters can be defined in your CSV file with names like `agent_random_traders`, `agent_trend_followers`, etc.")
                
                with col2:
                    # Simulation runs configuration
                    num_runs = st.slider(
                        "Number of Simulation Runs",
                        min_value=1,
                        max_value=100,
                        value=1,
                        help="Number of simulation runs. 1 = single run, >1 = Monte Carlo analysis with statistical measures",
                        key="num_runs"
                    )
                    
                    # Only show additional Monte Carlo options if multiple runs
                    if num_runs > 1:
                        show_confidence_intervals = st.checkbox(
                            "Show Confidence Intervals",
                            value=True,
                            help="Display 95% confidence intervals around the mean",
                            key="show_confidence_intervals"
                        )
                        show_percentiles = st.checkbox(
                            "Show Percentiles",
                            value=True,
                            help="Display 5th, 25th, 75th, and 95th percentiles",
                            key="show_percentiles"
                        )
                    else:
                        show_confidence_intervals = False
                        show_percentiles = False
            
            # Store parameters in session state
            StateManager.set_simulation_params(params)
            StateManager.set_simulation_type(simulation_approach.lower().replace(" ", "_").replace("-", "_"))
            
            # Display parameter summary
            if params:
                with st.expander("📋 Current Parameter Values", expanded=False):
                    dynamic_ui.display_parameter_summary(params)
            
            # Create a row with Run and Reset buttons
            col1, col2 = st.columns([1, 1])
            
            # Run button
            with col1:
                run_button = st.button("🚀 Run Simulation", type="primary")
            
            # Reset button
            with col2:
                if st.button("🔄 Reset Simulation"):
                    StateManager.clear_simulation_results()
                    st.success("Simulation state has been reset. You can run a new simulation now.")
                    st.rerun()
            
            if run_button:
                # Get parameters from session state
                params = StateManager.get_simulation_params()
                sim_approach = StateManager.get_simulation_type()
                
                # Determine simulation type for backend
                simulation_type = "agent" if "agent" in sim_approach else "stochastic"
                
                # Run simulation with progress bar
                with st.spinner(f"Running simulation with {num_runs} run{'s' if num_runs > 1 else ''}..."):
                    try:
                        if simulation_type == "agent":
                            # For agent-based, use the simulation's native method
                            sim_result = simulation.run_simulation(
                                params, 
                                num_runs=num_runs,
                                simulation_type="agent"
                            )
                        else:
                            # For stochastic, use Monte Carlo simulator
                            mc_simulator = MonteCarloSimulator(simulation)
                            sim_result = mc_simulator.run_simulations(
                                params, 
                                num_runs=num_runs
                            )
                            StateManager.set_simulator(mc_simulator)
                        
                        StateManager.set_simulation_result(sim_result)
                        
                        # Store agent data if available (for agent-based simulations)
                        if hasattr(simulation, 'agent_data') and simulation.agent_data is not None:
                            StateManager.set_agent_data(simulation.agent_data)
                            
                    except Exception as e:
                        ErrorHandler.show_error("Error in simulation", str(e))
            
            # Display simulation results if available
            sim_result = StateManager.get_simulation_result()
            
            if sim_result is not None:
                # Get the current number of runs to determine how to display results
                current_num_runs = st.session_state.get('num_runs', 1)
                
                # Display results based on number of runs performed
                if current_num_runs == 1:
                    # Display single run results
                    display_single_run_results(sim_result)
                    
                    # Show parameter coverage validation after simulation
                    if parameter_registry.parameters:
                        consumed_params = simulation.consumed_params if hasattr(simulation, 'consumed_params') else []
                        dynamic_ui.validate_and_display_coverage(consumed_params)
                else:
                    # Display Monte Carlo results
                    display_monte_carlo_results(
                        sim_result, 
                        show_confidence_intervals=show_confidence_intervals,
                        show_percentiles=show_percentiles
                    )
            elif not StateManager.is_simulation_result_displayed() and params:
                # Only run initial simulation if we have valid parameters from the UI
                try:
                    # Check if we have a reasonable set of parameters
                    has_valid_params = (
                        len(params) > 0 and 
                        any(param_value is not None for param_value in params.values()) and
                        parameter_registry.parameters  # Ensure registry is populated
                    )
                    
                    if has_valid_params:
                        with st.spinner("Running initial simulation with current parameters..."):
                            sim_result = simulation.run_simulation(params, num_runs=1, simulation_type="stochastic")
                            StateManager.set_simulation_result(sim_result)
                            StateManager.set_simulation_result_displayed(True)
                            
                            # Display single run results
                            display_single_run_results(sim_result)
                    else:
                        st.info("💡 **Ready to simulate!** Adjust parameters above and click 'Run Simulation' to begin.")
                except Exception as e:
                    st.info(f"💡 **Welcome!** Please adjust parameters above and click 'Run Simulation' to begin.")
                    st.text(f"Debug: Initial simulation skipped - {str(e)}")
            elif not params and not StateManager.is_simulation_result_displayed():
                st.info("💡 **Welcome!** Upload a CSV file with parameters to start simulating your tokenomics.")
        else:
            st.info("Please upload and process a radCAD Inputs CSV in the 'Load Data' tab to run simulations.")


if __name__ == "__main__":
    # Wrap the main function with error handling
    try:
        main()
    except Exception as e:
        ErrorHandler.log_error(e, "main")
        st.error(f"An unexpected error occurred: {str(e)}")