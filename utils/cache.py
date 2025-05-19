"""
Caching utilities for the Unit Zero Labs Tokenomics Engine.
Provides caching mechanisms for expensive computations.
"""

import functools
import hashlib
import json
import time
import streamlit as st
from typing import Dict, Any, Callable, Optional, Tuple, List, Union
import os
import pickle
import datetime


class ComputationCache:
    """
    Cache manager for expensive computations.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size: int = 100):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store persistent cache files
            max_cache_size: Maximum number of items to keep in memory cache
        """
        self.memory_cache = {}
        self.memory_cache_info = {}
        self.max_cache_size = max_cache_size
        
        # Setup persistent cache if a directory is provided
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """
        Generate a cache key based on function name and arguments.
        
        Args:
            func_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            String cache key
        """
        # Convert args and kwargs to a serializable form
        cache_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs
        }
        
        # Create a JSON string and hash it
        try:
            json_str = json.dumps(cache_data, sort_keys=True)
        except (TypeError, OverflowError):
            # If arguments are not JSON serializable, use their string representation
            json_str = str(cache_data)
        
        return hashlib.md5(json_str.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """
        Get the file path for a persistent cache item.
        
        Args:
            cache_key: Cache key
            
        Returns:
            File path for the cache item
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pickle")
    
    def get(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Get a cached result if available.
        
        Args:
            func_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Tuple containing:
            - Boolean indicating if a cache hit occurred
            - Cached result or None if cache miss
        """
        cache_key = self._get_cache_key(func_name, args, kwargs)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            # Update access time
            self.memory_cache_info[cache_key]["last_access"] = time.time()
            self.memory_cache_info[cache_key]["access_count"] += 1
            return True, self.memory_cache[cache_key]
        
        # If persistent cache is enabled, check there
        if self.cache_dir:
            cache_file = self._get_cache_file_path(cache_key)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    # Add to memory cache
                    self.memory_cache[cache_key] = cache_data
                    self.memory_cache_info[cache_key] = {
                        "last_access": time.time(),
                        "last_update": os.path.getmtime(cache_file),
                        "access_count": 1
                    }
                    
                    # Manage cache size
                    self._manage_cache_size()
                    
                    return True, cache_data
                except Exception:
                    # If there's an error reading the cache, ignore it
                    pass
        
        return False, None
    
    def set(self, func_name: str, args: Tuple, kwargs: Dict[str, Any], result: Any) -> None:
        """
        Store a result in the cache.
        
        Args:
            func_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments
            result: Result to cache
        """
        cache_key = self._get_cache_key(func_name, args, kwargs)
        
        # Store in memory cache
        self.memory_cache[cache_key] = result
        current_time = time.time()
        self.memory_cache_info[cache_key] = {
            "last_access": current_time,
            "last_update": current_time,
            "access_count": 1
        }
        
        # Store in persistent cache if enabled
        if self.cache_dir:
            cache_file = self._get_cache_file_path(cache_key)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception:
                # If there's an error writing to the cache, ignore it
                pass
        
        # Manage cache size
        self._manage_cache_size()
    
    def _manage_cache_size(self) -> None:
        """
        Manage the memory cache size by removing least recently used items.
        """
        if len(self.memory_cache) <= self.max_cache_size:
            return
        
        # Sort cache items by last access time
        sorted_items = sorted(
            self.memory_cache_info.items(), 
            key=lambda x: x[1]["last_access"]
        )
        
        # Remove oldest items until we're within the size limit
        items_to_remove = len(self.memory_cache) - self.max_cache_size
        for i in range(items_to_remove):
            key_to_remove = sorted_items[i][0]
            del self.memory_cache[key_to_remove]
            del self.memory_cache_info[key_to_remove]
    
    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        self.memory_cache.clear()
        self.memory_cache_info.clear()
        
        # Clear persistent cache if enabled
        if self.cache_dir:
            try:
                for file in os.listdir(self.cache_dir):
                    if file.endswith(".pickle"):
                        os.remove(os.path.join(self.cache_dir, file))
            except Exception:
                # If there's an error clearing the cache, ignore it
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "max_cache_size": self.max_cache_size
        }
        
        if self.cache_dir:
            try:
                persistent_cache_files = [
                    f for f in os.listdir(self.cache_dir) if f.endswith(".pickle")
                ]
                stats["persistent_cache_size"] = len(persistent_cache_files)
                stats["persistent_cache_dir"] = self.cache_dir
            except Exception:
                stats["persistent_cache_size"] = "Error reading directory"
        
        return stats


# Create a global instance of the cache
computation_cache = ComputationCache(
    cache_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"),
    max_cache_size=100
)


def cached(ttl: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds (optional)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function name
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Check if result is in cache
            cache_hit, cached_result = computation_cache.get(func_name, args, kwargs)
            
            if cache_hit:
                # Check TTL if provided
                if ttl is not None:
                    cache_key = computation_cache._get_cache_key(func_name, args, kwargs)
                    last_update = computation_cache.memory_cache_info.get(cache_key, {}).get("last_update", 0)
                    
                    if time.time() - last_update > ttl:
                        # TTL expired, compute new result
                        result = func(*args, **kwargs)
                        computation_cache.set(func_name, args, kwargs, result)
                        return result
                
                return cached_result
            
            # Cache miss, compute result
            result = func(*args, **kwargs)
            computation_cache.set(func_name, args, kwargs, result)
            return result
        
        return wrapper
    
    return decorator


def st_cache_data(ttl: Optional[int] = None, max_entries: int = 100):
    """
    Decorator that mimics st.cache_data but with improved control and persistence.
    For use in non-Streamlit contexts or when more control is needed.
    
    Args:
        ttl: Time-to-live in seconds (optional)
        max_entries: Maximum number of entries to cache
        
    Returns:
        Decorated function
    """
    return cached(ttl)


def clear_all_caches():
    """
    Clear all caches (both Streamlit's built-in cache and our custom cache).
    """
    # Clear Streamlit's cache
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except:
        # Ignore if we're in a non-Streamlit context
        pass
    
    # Clear our custom cache
    computation_cache.clear()


def get_cache_stats():
    """
    Get statistics about all caches.
    
    Returns:
        Dictionary with cache statistics
    """
    stats = {
        "custom_cache": computation_cache.get_stats(),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return stats