#!/usr/bin/env python3
"""
Configuration loader for RAG Tutorial v2.
Handles loading configuration from config.yaml file with proper logging setup.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging to suppress verbose HTTP logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

# Suppress ChromaDB telemetry
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)


class ConfigLoader:
    """Configuration loader for RAG Tutorial v2."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self._config_cache = None
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        # Check cache first
        if self._config_cache is not None:
            return self._config_cache
        
        if not os.path.exists(self.config_file):
            logger.warning(f"Config file {self.config_file} not found, using defaults")
            config = self._get_default_config()
        else:
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Merge with defaults to ensure all required keys exist
                default_config = self._get_default_config()
                config = self._merge_configs(default_config, config)
                
                logger.info(f"Loaded configuration from {self.config_file}")
                
            except Exception as e:
                logger.error(f"Failed to load config from {self.config_file}: {e}")
                logger.warning("Using default configuration")
                config = self._get_default_config()
        
        # Cache the result
        self._config_cache = config
        return config
    
    def clear_cache(self):
        """Clear the configuration cache."""
        self._config_cache = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "embedding": {
                "provider": "ollama",
                "model": None,
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "timeout": 120
                },
                "openai": {
                    "api_key": None
                },
                "huggingface": {
                    "api_key": None,
                    "cache_dir": "./cache"
                },
                "bedrock": {
                    "aws_profile": "default",
                    "aws_region": "us-east-1"
                }
            },
            "llm": {
                "provider": "ollama",
                "model": "llama3.1:latest",
                "ollama": {
                    "base_url": "http://localhost:11434"
                }
            },
            "database": {
                "chroma_path": "chroma",
                "data_path": "data"
            },
            "documents": {
                "chunk_size": 800,
                "chunk_overlap": 80,
                "supported_formats": ["pdf", "md", "txt", "rtf"]
            },
            "query": {
                "max_results": 5,
                "similarity_threshold": 0.0
            }
        }
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults."""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        config = self.load_config()
        return config.get("embedding", {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        config = self.load_config()
        return config.get("llm", {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        config = self.load_config()
        return config.get("database", {})
    
    def get_documents_config(self) -> Dict[str, Any]:
        """Get documents configuration."""
        config = self.load_config()
        return config.get("documents", {})
    
    def get_query_config(self) -> Dict[str, Any]:
        """Get query configuration."""
        config = self.load_config()
        return config.get("query", {})


# Global config loader instance
_config_loader = ConfigLoader()


def load_config() -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Returns:
        Configuration dictionary
    """
    return _config_loader.load_config()


def get_embedding_config() -> Dict[str, Any]:
    """
    Convenience function to get embedding configuration.
    
    Returns:
        Embedding configuration dictionary
    """
    return _config_loader.get_embedding_config()


def get_llm_config() -> Dict[str, Any]:
    """
    Convenience function to get LLM configuration.
    
    Returns:
        LLM configuration dictionary
    """
    return _config_loader.get_llm_config()


def get_database_config() -> Dict[str, Any]:
    """
    Convenience function to get database configuration.
    
    Returns:
        Database configuration dictionary
    """
    return _config_loader.get_database_config()


def get_documents_config() -> Dict[str, Any]:
    """
    Convenience function to get documents configuration.
    
    Returns:
        Documents configuration dictionary
    """
    return _config_loader.get_documents_config()


def get_query_config() -> Dict[str, Any]:
    """
    Convenience function to get query configuration.
    
    Returns:
        Query configuration dictionary
    """
    return _config_loader.get_query_config()


def clear_config_cache():
    """Clear the configuration cache."""
    _config_loader.clear_cache()


if __name__ == "__main__":
    # Test loading configuration
    config = load_config()
    print("Configuration loaded successfully:")
    print(f"Embedding provider: {config['embedding']['provider']}")
    print(f"Embedding model: {config['embedding']['model']}")
    print(f"LLM provider: {config['llm']['provider']}")
    print(f"LLM model: {config['llm']['model']}")
