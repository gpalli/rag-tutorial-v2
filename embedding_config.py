"""
Configuration system for different embedding models and providers.
Supports multiple embedding providers including Ollama, OpenAI, Hugging Face, and AWS Bedrock.
"""

import os
from typing import Dict, Any, Optional
from enum import Enum


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    BEDROCK = "bedrock"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class EmbeddingConfig:
    """Configuration class for embedding models."""
    
    # Default configurations for each provider
    DEFAULT_CONFIGS = {
        EmbeddingProvider.OLLAMA: {
            "models": {
                "nomic-embed-text": {
                    "model_name": "nomic-embed-text",
                    "dimensions": 768,
                    "max_tokens": 2048,
                    "description": "Nomic's text embedding model via Ollama"
                },
                "mxbai-embed-large": {
                    "model_name": "mxbai-embed-large", 
                    "dimensions": 1024,
                    "max_tokens": 512,
                    "description": "Mixedbread AI's large embedding model via Ollama"
                },
                "all-minilm": {
                    "model_name": "all-minilm",
                    "dimensions": 384,
                    "max_tokens": 256,
                    "description": "All-MiniLM embedding model via Ollama"
                }
            },
            "default_model": "nomic-embed-text"
        },
        EmbeddingProvider.OPENAI: {
            "models": {
                "text-embedding-3-small": {
                    "model_name": "text-embedding-3-small",
                    "dimensions": 1536,
                    "max_tokens": 8191,
                    "description": "OpenAI's latest small embedding model"
                },
                "text-embedding-3-large": {
                    "model_name": "text-embedding-3-large",
                    "dimensions": 3072,
                    "max_tokens": 8191,
                    "description": "OpenAI's latest large embedding model"
                },
                "text-embedding-ada-002": {
                    "model_name": "text-embedding-ada-002",
                    "dimensions": 1536,
                    "max_tokens": 8191,
                    "description": "OpenAI's previous generation embedding model"
                }
            },
            "default_model": "text-embedding-3-small"
        },
        EmbeddingProvider.HUGGINGFACE: {
            "models": {
                "sentence-transformers/all-MiniLM-L6-v2": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "max_tokens": 256,
                    "description": "Popular sentence transformer model"
                },
                "sentence-transformers/all-mpnet-base-v2": {
                    "model_name": "sentence-transformers/all-mpnet-base-v2",
                    "dimensions": 768,
                    "max_tokens": 512,
                    "description": "High-quality sentence transformer model"
                },
                "intfloat/e5-large-v2": {
                    "model_name": "intfloat/e5-large-v2",
                    "dimensions": 1024,
                    "max_tokens": 512,
                    "description": "Microsoft's E5 large embedding model"
                }
            },
            "default_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        EmbeddingProvider.BEDROCK: {
            "models": {
                "amazon.titan-embed-text-v1": {
                    "model_name": "amazon.titan-embed-text-v1",
                    "dimensions": 1536,
                    "max_tokens": 8000,
                    "description": "AWS Titan text embedding model"
                },
                "amazon.titan-embed-text-v2": {
                    "model_name": "amazon.titan-embed-text-v2",
                    "dimensions": 1024,
                    "max_tokens": 8000,
                    "description": "AWS Titan text embedding model v2"
                }
            },
            "default_model": "amazon.titan-embed-text-v1"
        },
        EmbeddingProvider.SENTENCE_TRANSFORMERS: {
            "models": {
                "all-MiniLM-L6-v2": {
                    "model_name": "all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "max_tokens": 256,
                    "description": "Lightweight sentence transformer"
                },
                "all-mpnet-base-v2": {
                    "model_name": "all-mpnet-base-v2",
                    "dimensions": 768,
                    "max_tokens": 512,
                    "description": "High-quality sentence transformer"
                }
            },
            "default_model": "all-MiniLM-L6-v2"
        }
    }
    
    def __init__(self, provider: EmbeddingProvider = EmbeddingProvider.OLLAMA, 
                 model_name: Optional[str] = None, **kwargs):
        """
        Initialize embedding configuration.
        
        Args:
            provider: The embedding provider to use
            model_name: Specific model name (uses default if None)
            **kwargs: Additional configuration parameters
        """
        self.provider = provider
        self.config = self.DEFAULT_CONFIGS[provider].copy()
        
        # Use provided model name or default
        if model_name is None:
            model_name = self.config["default_model"]
        
        if model_name not in self.config["models"]:
            available_models = list(self.config["models"].keys())
            raise ValueError(f"Model '{model_name}' not found for provider '{provider}'. "
                           f"Available models: {available_models}")
        
        self.model_name = model_name
        self.model_config = self.config["models"][model_name]
        
        # Store additional parameters
        self.extra_params = kwargs
        
        # Set up provider-specific configurations
        self._setup_provider_config()
    
    def _setup_provider_config(self):
        """Set up provider-specific configuration parameters."""
        if self.provider == EmbeddingProvider.OLLAMA:
            self.ollama_base_url = self.extra_params.get("ollama_base_url", "http://localhost:11434")
            self.ollama_timeout = self.extra_params.get("ollama_timeout", 120)
            
        elif self.provider == EmbeddingProvider.OPENAI:
            self.openai_api_key = self.extra_params.get("openai_api_key")
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required. Set api_key in config.yaml or pass openai_api_key parameter.")
                
        elif self.provider == EmbeddingProvider.HUGGINGFACE:
            self.hf_api_key = self.extra_params.get("hf_api_key")
            self.hf_cache_dir = self.extra_params.get("hf_cache_dir", "./cache")
            
        elif self.provider == EmbeddingProvider.BEDROCK:
            self.aws_profile = self.extra_params.get("aws_profile", "default")
            self.aws_region = self.extra_params.get("aws_region", "us-east-1")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "dimensions": self.model_config["dimensions"],
            "max_tokens": self.model_config["max_tokens"],
            "description": self.model_config["description"]
        }
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models for the current provider."""
        return self.config["models"]
    
    @classmethod
    def list_all_providers_and_models(cls) -> Dict[str, Any]:
        """List all available providers and their models."""
        result = {}
        for provider, config in cls.DEFAULT_CONFIGS.items():
            result[provider.value] = {
                "default_model": config["default_model"],
                "models": config["models"]
            }
        return result


def get_embedding_config(provider: str = "ollama", model: str = None, **kwargs) -> EmbeddingConfig:
    """
    Convenience function to get embedding configuration.
    
    Args:
        provider: Provider name (ollama, openai, huggingface, bedrock, sentence_transformers)
        model: Model name (uses default if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        EmbeddingConfig instance
    """
    try:
        provider_enum = EmbeddingProvider(provider.lower())
    except ValueError:
        available_providers = [p.value for p in EmbeddingProvider]
        raise ValueError(f"Unknown provider '{provider}'. Available providers: {available_providers}")
    
    return EmbeddingConfig(provider_enum, model, **kwargs)


if __name__ == "__main__":
    # Example usage and model listing
    print("Available Embedding Providers and Models:")
    print("=" * 50)
    
    all_configs = EmbeddingConfig.list_all_providers_and_models()
    for provider, info in all_configs.items():
        print(f"\n{provider.upper()}:")
        print(f"  Default: {info['default_model']}")
        for model_name, model_info in info['models'].items():
            print(f"  - {model_name}: {model_info['description']} ({model_info['dimensions']}D)")
