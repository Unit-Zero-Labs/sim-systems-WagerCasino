"""
Error handling utilities for the Unit Zero Labs Tokenomics Engine.
Provides functions for handling and displaying errors consistently.
"""

import streamlit as st
import sys
import traceback
from typing import Optional, Dict, Any, List, Callable, Union
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("tokenomics_engine")


class ErrorHandler:
    """
    Error handler class for managing errors in the application.
    """
    
    @staticmethod
    def log_error(error: Exception, context: Optional[str] = None) -> None:
        """
        Log an error with optional context.
        
        Args:
            error: The exception to log
            context: Optional context string describing where the error occurred
        """
        error_traceback = traceback.format_exc()
        error_message = str(error)
        
        if context:
            logger.error(f"Error in {context}: {error_message}")
            logger.debug(error_traceback)
        else:
            logger.error(f"Error: {error_message}")
            logger.debug(error_traceback)
    
    @staticmethod
    def show_error(error_message: str, error_detail: Optional[str] = None) -> None:
        """
        Display an error message in the Streamlit UI.
        
        Args:
            error_message: The main error message to display
            error_detail: Optional detailed error information
        """
        st.error(error_message)
        if error_detail:
            with st.expander("Error Details"):
                st.code(error_detail)
    
    @staticmethod
    def handle_exception(func: Callable) -> Callable:
        """
        Decorator to handle exceptions in a function.
        
        Args:
            func: The function to wrap
            
        Returns:
            Wrapped function that handles exceptions
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                ErrorHandler.log_error(e, func.__name__)
                
                # Display user-friendly error message
                error_message = f"An error occurred during {func.__name__}"
                error_detail = str(e)
                ErrorHandler.show_error(error_message, error_detail)
                
                # Return None to indicate failure
                return None
        return wrapper
    
    @staticmethod
    def handle_validation_errors(errors: List[str]) -> None:
        """
        Display validation errors in the Streamlit UI.
        
        Args:
            errors: List of validation error messages
        """
        if not errors:
            return
        
        error_message = "Validation errors:"
        error_detail = "\n".join([f"- {error}" for error in errors])
        
        st.error(error_message)
        st.markdown(error_detail)
    
    @staticmethod
    def safe_execute(func: Callable, error_message: str, *args, **kwargs) -> Optional[Any]:
        """
        Execute a function safely, handling any exceptions.
        
        Args:
            func: The function to execute
            error_message: Message to display if an error occurs
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function or None if an error occurred
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error
            ErrorHandler.log_error(e, func.__name__)
            
            # Display user-friendly error message
            ErrorHandler.show_error(error_message, str(e))
            
            # Return None to indicate failure
            return None