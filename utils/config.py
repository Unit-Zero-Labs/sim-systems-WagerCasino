"""
Configuration management for the Unit Zero Labs Tokenomics Engine.
Provides a centralized way to manage application settings.
"""

import os
import json
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager that handles application settings.
    
    Attributes:
        _instance: Singleton instance
        _config: Dictionary of configuration values
        _config_path: Path to the configuration file
    """
    _instance = None
    _config = None
    _config_path = None
    
    def __new__(cls, config_path: Optional[str] = None):
        """
        Create a singleton instance of the Config class.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Config: Singleton instance of the Config class
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config = {}
            cls._instance._config_path = config_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config.json'
            )
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from the config file if it exists."""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, 'r') as f:
                    self._config = json.load(f)
            else:
                self._config = {
                    "simulation": {
                        "default_staking_share": 0.75,
                        "default_token_price": 0.03,
                        "default_staking_apr_multiplier": 1.0,
                        "default_market_volatility": 0.2
                    },
                    "monte_carlo": {
                        "default_num_runs": 50,
                        "default_show_confidence_intervals": True,
                        "default_show_percentiles": True,
                        "max_runs": 100
                    },
                    "agent_based": {
                        "default_random_trader_count": 20,
                        "default_trend_follower_count": 15,
                        "default_staking_agent_count": 10,
                        "default_liquidity_factor": 0.0001,
                        "default_max_price_change_pct": 0.1,
                        "default_min_token_price": 0.001,
                        "default_base_staking_apr": 0.05,
                        "default_apr_scaling_factor": 0.5,
                        "default_min_staking_apr": 0.01
                    },
                    "ui": {
                        "default_theme": "dark",
                        "company_name": "Unit Zero Labs",
                        "product_name": "Tokenomics Engine"
                    },
                    "paths": {
                        "logo_path": "public/uz-logo.png",
                        "client_logo_path": "public/client-logo.png"
                    }
                }
                self._save_config()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            # Fall back to default configuration
            self._config = {
                "simulation": {
                    "default_staking_share": 0.75,
                    "default_token_price": 0.03,
                    "default_staking_apr_multiplier": 1.0,
                    "default_market_volatility": 0.2
                },
                "monte_carlo": {
                    "default_num_runs": 50,
                    "default_show_confidence_intervals": True,
                    "default_show_percentiles": True,
                    "max_runs": 100
                },
                "agent_based": {
                    "default_random_trader_count": 20,
                    "default_trend_follower_count": 15,
                    "default_staking_agent_count": 10,
                    "default_liquidity_factor": 0.0001,
                    "default_max_price_change_pct": 0.1,
                    "default_min_token_price": 0.001,
                    "default_base_staking_apr": 0.05,
                    "default_apr_scaling_factor": 0.5,
                    "default_min_staking_apr": 0.01
                },
                "ui": {
                    "default_theme": "dark",
                    "company_name": "Unit Zero Labs",
                    "product_name": "Tokenomics Engine"
                },
                "paths": {
                    "logo_path": "public/uz-logo.png",
                    "client_logo_path": "public/client-logo.png"
                }
            }
    
    def _save_config(self):
        """Save the current configuration to the config file."""
        try:
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, 'w') as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested values)
            default: Default value to return if the key doesn't exist
            
        Returns:
            The configuration value or the default
        """
        try:
            parts = key.split('.')
            value = self._config
            for part in parts:
                value = value.get(part, {})
            
            # Check if we got a non-empty value
            if value == {} and default is not None:
                return default
            return value
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested values)
            value: Value to set
        """
        try:
            parts = key.split('.')
            config = self._config
            
            # Navigate to the correct position in the config dictionary
            for i, part in enumerate(parts[:-1]):
                if part not in config:
                    config[part] = {}
                config = config[part]
            
            # Set the value
            config[parts[-1]] = value
            
            # Save the updated configuration
            self._save_config()
        except Exception as e:
            print(f"Error setting configuration {key}: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            Dict: The entire configuration
        """
        return self._config.copy()


# Global configuration instance for easy access
config = Config()


def get_simulation_defaults() -> Dict[str, Any]:
    """
    Get default simulation parameters.
    
    Returns:
        Dict: Default simulation parameters
    """
    return {
        "staking_share": config.get("simulation.default_staking_share", 0.75),
        "token_price": config.get("simulation.default_token_price", 0.03),
        "staking_apr_multiplier": config.get("simulation.default_staking_apr_multiplier", 1.0),
        "market_volatility": config.get("simulation.default_market_volatility", 0.2)
    }


def get_monte_carlo_defaults() -> Dict[str, Any]:
    """
    Get default Monte Carlo simulation parameters.
    
    Returns:
        Dict: Default Monte Carlo parameters
    """
    return {
        "num_runs": config.get("monte_carlo.default_num_runs", 50),
        "show_confidence_intervals": config.get("monte_carlo.default_show_confidence_intervals", True),
        "show_percentiles": config.get("monte_carlo.default_show_percentiles", True),
        "max_runs": config.get("monte_carlo.max_runs", 100)
    }


def get_agent_defaults() -> Dict[str, Any]:
    """
    Get default agent-based simulation parameters.
    
    Returns:
        Dict: Default agent-based parameters
    """
    return {
        "token_price": config.get("simulation.default_token_price", 0.03),
        "random_trader_count": config.get("agent_based.default_random_trader_count", 20),
        "trend_follower_count": config.get("agent_based.default_trend_follower_count", 15),
        "staking_agent_count": config.get("agent_based.default_staking_agent_count", 10),
        "liquidity_factor": config.get("agent_based.default_liquidity_factor", 0.0001),
        "max_price_change_pct": config.get("agent_based.default_max_price_change_pct", 0.1),
        "min_token_price": config.get("agent_based.default_min_token_price", 0.001),
        "base_staking_apr": config.get("agent_based.default_base_staking_apr", 0.05),
        "apr_scaling_factor": config.get("agent_based.default_apr_scaling_factor", 0.5),
        "min_staking_apr": config.get("agent_based.default_min_staking_apr", 0.01)
    }


def get_ui_config() -> Dict[str, Any]:
    """
    Get UI configuration.
    
    Returns:
        Dict: UI configuration
    """
    return {
        "theme": config.get("ui.default_theme", "dark"),
        "company_name": config.get("ui.company_name", "Unit Zero Labs"),
        "product_name": config.get("ui.product_name", "Tokenomics Engine"),
        "logo_path": config.get("paths.logo_path", "public/uz-logo.png"),
        "client_logo_path": config.get("paths.client_logo_path", "public/client-logo.png")
    }