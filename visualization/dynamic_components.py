"""
Dynamic UI component generation for parameter-driven simulations.
Automatically creates Streamlit controls based on parameter definitions.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from logic.parameter_registry import ParameterRegistry, ParameterDefinition, ParameterType, ParameterCategory


class DynamicUIGenerator:
    """
    Generates Streamlit UI controls dynamically based on parameter definitions.
    Enables automatic UI adaptation for new client parameters.
    """
    
    def __init__(self, registry: ParameterRegistry):
        self.registry = registry
        
    def display_parameter_controls(self, key_prefix: str = "") -> Dict[str, Any]:
        """
        Display all configurable parameters organized by category.
        
        Args:
            key_prefix: Prefix for Streamlit widget keys
            
        Returns:
            Dictionary of parameter values from user inputs
        """
        configurable_params = self.registry.get_configurable_parameters()
        
        if not configurable_params:
            st.warning("No configurable parameters found. Please upload a CSV file with parameters.")
            return {}
        
        # Add expand/collapse controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader("Parameter Configuration")
        with col2:
            if st.button("Expand All", key=f"{key_prefix}_expand_all"):
                # Expand all sections
                for category in ParameterCategory:
                    st.session_state[f"{key_prefix}_show_{category.value}"] = True
        with col3:
            if st.button("Collapse All", key=f"{key_prefix}_collapse_all"):
                # Collapse all sections
                for category in ParameterCategory:
                    st.session_state[f"{key_prefix}_show_{category.value}"] = False
        
        # Group parameters by category
        params_by_category = {}
        for name, param_def in configurable_params.items():
            category = param_def.category
            if category not in params_by_category:
                params_by_category[category] = {}
            params_by_category[category][name] = param_def
        
        # Create expandable sections for each category
        user_inputs = {}
        
        for category, params in params_by_category.items():
            if params:  # Only show categories that have parameters
                category_name = category.value.replace('_', ' ').title()
                
                # Use a checkbox to control section visibility - this persists across reruns
                show_key = f"{key_prefix}_show_{category.value}"
                if show_key not in st.session_state:
                    st.session_state[show_key] = True  # Default to expanded so users can see parameters
                
                # Create a clickable header
                col_header, col_toggle = st.columns([4, 1])
                with col_header:
                    st.markdown(f"### {category_name} Parameters")
                with col_toggle:
                    # Use arrow icon to indicate expand/collapse state
                    arrow_icon = "▼" if st.session_state[show_key] else "▶️"
                    if st.button(
                        arrow_icon,
                        key=f"{show_key}_toggle",
                        help="Click to expand/collapse this section"
                    ):
                        # Toggle the state when button is clicked
                        st.session_state[show_key] = not st.session_state[show_key]
                        st.rerun()
                
                # Show parameters if section is expanded
                if st.session_state[show_key]:
                    with st.container():
                        category_inputs = self._display_category_parameters(
                            params, 
                            key_prefix=f"{key_prefix}_{category.value}"
                        )
                        user_inputs.update(category_inputs)
                    st.divider()  # Add visual separation between sections
        
        return user_inputs
    
    def _display_category_parameters(self, params: Dict[str, ParameterDefinition], key_prefix: str = "") -> Dict[str, Any]:
        """Display parameters for a specific category."""
        user_inputs = {}
        
        # Determine optimal column layout based on number of parameters
        num_params = len(params)
        if num_params <= 2:
            cols = st.columns(num_params)
        elif num_params <= 4:
            cols = st.columns(2)
        else:
            cols = st.columns(3)
        
        for i, (param_name, param_def) in enumerate(params.items()):
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                value = self._create_parameter_control(param_def, key_prefix)
                user_inputs[param_name] = value
                
        return user_inputs
    
    def _create_parameter_control(self, param_def: ParameterDefinition, key_prefix: str = "") -> Any:
        """Create an appropriate Streamlit control for a parameter."""
        key = f"{key_prefix}_{param_def.name}"
        
        if param_def.param_type == ParameterType.BOOLEAN:
            return st.checkbox(
                self._format_label(param_def.name),
                value=bool(param_def.value) if param_def.value is not None else False,
                key=key,
                help=param_def.help_text
            )
            
        elif param_def.param_type == ParameterType.PERCENTAGE:
            return st.slider(
                self._format_label(param_def.name),
                min_value=param_def.min_value or 0.0,
                max_value=param_def.max_value or 1.0,
                value=float(param_def.value) if param_def.value is not None else 0.5,
                step=float(param_def.step) if param_def.step is not None else 0.01,
                format=param_def.format_str or "%.2f",
                key=key,
                help=param_def.help_text
            )
            
        elif param_def.param_type == ParameterType.NUMERIC:
            # For numeric parameters, check if we have reasonable bounds
            if param_def.min_value is not None and param_def.max_value is not None:
                return st.slider(
                    self._format_label(param_def.name),
                    min_value=float(param_def.min_value),
                    max_value=float(param_def.max_value),
                    value=float(param_def.value) if param_def.value is not None else float(param_def.min_value),
                    step=float(param_def.step) if param_def.step is not None else 1.0,
                    key=key,
                    help=param_def.help_text
                )
            else:
                # Use number input for unbounded numeric values
                return st.number_input(
                    self._format_label(param_def.name),
                    value=float(param_def.value) if param_def.value is not None else 0.0,
                    step=float(param_def.step) if param_def.step is not None else 1.0,
                    format=param_def.format_str,
                    key=key,
                    help=param_def.help_text
                )
                
        elif param_def.param_type == ParameterType.SELECTION:
            return st.selectbox(
                self._format_label(param_def.name),
                options=param_def.options or [],
                index=0 if param_def.options else None,
                key=key,
                help=param_def.help_text
            )
            
        elif param_def.param_type == ParameterType.STRING:
            return st.text_input(
                self._format_label(param_def.name),
                value=str(param_def.value) if param_def.value is not None else "",
                key=key,
                help=param_def.help_text
            )
            
        elif param_def.param_type == ParameterType.DATE:
            # For dates, we'll use text input for now since date input requires datetime objects
            return st.text_input(
                self._format_label(param_def.name),
                value=str(param_def.value) if param_def.value is not None else "",
                key=key,
                help=f"{param_def.help_text} (Format: DD.MM.YYYY)"
            )
            
        else:
            # Fallback to text input
            return st.text_input(
                self._format_label(param_def.name),
                value=str(param_def.value) if param_def.value is not None else "",
                key=key,
                help=param_def.help_text
            )
    
    def _format_label(self, param_name: str) -> str:
        """Format parameter name for display as a label."""
        # Convert snake_case and camelCase to Title Case
        import re
        
        # Handle camelCase
        formatted = re.sub(r'([a-z])([A-Z])', r'\1 \2', param_name)
        # Handle snake_case
        formatted = formatted.replace('_', ' ')
        # Convert to title case
        formatted = formatted.title()
        
        return formatted
    
    def display_parameter_summary(self, params: Dict[str, Any]) -> None:
        """Display a summary of current parameter values."""
        if not params:
            return
            
        st.subheader("Parameter Summary")
        
        # Group by category for display
        categorized_params = {}
        for param_name, value in params.items():
            if param_name in self.registry.parameters:
                param_def = self.registry.parameters[param_name]
                category = param_def.category.value.replace('_', ' ').title()
                
                if category not in categorized_params:
                    categorized_params[category] = {}
                categorized_params[category][self._format_label(param_name)] = value
        
        # Display in columns
        if categorized_params:
            categories = list(categorized_params.keys())
            num_cols = min(len(categories), 3)
            cols = st.columns(num_cols)
            
            for i, (category, category_params) in enumerate(categorized_params.items()):
                col_idx = i % num_cols
                
                with cols[col_idx]:
                    st.write(f"**{category}**")
                    for label, value in category_params.items():
                        if isinstance(value, float):
                            st.write(f"• {label}: {value:.3f}")
                        else:
                            st.write(f"• {label}: {value}")
    
    def discover_and_display_new_policies(self) -> List[str]:
        """
        Discover new policies based on parameters and display recommendations.
        
        Returns:
            List of discovered policy names
        """
        discovered_policies = self.registry.discover_new_policies()
        
        if discovered_policies:
            st.info("🔍 **Discovered New Capabilities**: Based on your parameters, the following new simulation features are available:")
            
            for policy in discovered_policies:
                if policy == "points_campaign":
                    st.write("• **Off-Chain Points Campaign**: Parameters detected for modeling user engagement through points systems")
                elif policy == "custom_agent_behavior":
                    st.write("• **Custom Agent Behavior**: Parameters detected for specialized agent trading patterns")
                elif policy == "custom_utility":
                    st.write("• **Custom Utility Mechanisms**: Parameters detected for novel token utility functions")
                else:
                    st.write(f"• **{policy.replace('_', ' ').title()}**: Custom simulation module")
        
        return discovered_policies
    
    def validate_and_display_coverage(self, consumed_params: List[str]) -> None:
        """Validate parameter coverage and display warnings for unused parameters."""
        unused, missing = self.registry.validate_parameter_coverage(consumed_params)
        
        if unused:
            st.warning(f"⚠️ **Unused Parameters**: The following parameters from your CSV are not being used in the simulation: {', '.join(unused[:5])}{'...' if len(unused) > 5 else ''}")
            
            with st.expander("View All Unused Parameters"):
                for param in unused:
                    if param in self.registry.parameters:
                        param_def = self.registry.parameters[param]
                        st.write(f"• `{param}` ({param_def.category.value}) = {param_def.value}")
        
        if missing:
            st.error(f"❌ **Missing Parameters**: The simulation is trying to use parameters that weren't found in your CSV: {', '.join(missing)}")


def create_dynamic_ui_generator(registry: ParameterRegistry) -> DynamicUIGenerator:
    """Factory function to create a dynamic UI generator."""
    return DynamicUIGenerator(registry) 