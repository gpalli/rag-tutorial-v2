import os
from typing import Optional
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from embedding_config import EmbeddingConfig, EmbeddingProvider
from config_loader import get_embedding_config


def get_embedding_function(provider: str = "ollama", model: str = None, **kwargs):
    """
    Get embedding function based on provider and model.
    
    Args:
        provider: Embedding provider (ollama, openai, huggingface, bedrock, sentence_transformers)
        model: Specific model name (uses default if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        Embedding function instance
        
    Raises:
        ValueError: If provider is unsupported
        RuntimeError: If model is not available (for Ollama)
    """
    try:
        config = EmbeddingConfig(EmbeddingProvider(provider.lower()), model, **kwargs)
        
        if config.provider == EmbeddingProvider.OLLAMA:
            # Test if Ollama model is available before creating the embedding function
            try:
                import ollama
                client = ollama.Client(host=config.ollama_base_url)
                # Try to pull the model to check if it exists
                client.pull(config.model_name)
            except Exception as e:
                if "not found" in str(e).lower() or "404" in str(e):
                    raise RuntimeError(f"Model '{config.model_name}' is not available. Please install it first with: ollama pull {config.model_name}")
                else:
                    raise RuntimeError(f"Failed to connect to Ollama at {config.ollama_base_url}. Make sure Ollama is running.")
            
            return OllamaEmbeddings(
                model=config.model_name,
                base_url=config.ollama_base_url
            )
        
        elif config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=config.openai_api_key
            )
        
        elif config.provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbeddings(
                model_name=config.model_name,
                cache_folder=config.hf_cache_dir,
                model_kwargs={'device': 'cpu'},  # Use GPU if available: 'cuda'
                encode_kwargs={'normalize_embeddings': True}
            )
        
        elif config.provider == EmbeddingProvider.BEDROCK:
            return BedrockEmbeddings(
                credentials_profile_name=config.aws_profile,
                region_name=config.aws_region,
                model_id=config.model_name
            )
        
        elif config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return SentenceTransformerEmbeddings(
                model_name=config.model_name,
                model_kwargs={'device': 'cpu'},  # Use GPU if available: 'cuda'
                encode_kwargs={'normalize_embeddings': True}
            )
        
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        else:
            raise RuntimeError(f"Failed to create embedding function: {str(e)}")


def get_embedding_function_from_config():
    """
    Get embedding function from config.yaml file.
    """
    config = get_embedding_config()
    provider = config.get("provider", "ollama")
    model = config.get("model")
    
    # Provider-specific configuration
    kwargs = {}
    
    if provider == "openai":
        openai_config = config.get("openai", {})
        kwargs["openai_api_key"] = openai_config.get("api_key")
        if not model and openai_config.get("model"):
            model = openai_config.get("model")
    elif provider == "huggingface":
        hf_config = config.get("huggingface", {})
        kwargs["hf_api_key"] = hf_config.get("api_key")
        kwargs["hf_cache_dir"] = hf_config.get("cache_dir", "./cache")
        if not model and hf_config.get("model"):
            model = hf_config.get("model")
    elif provider == "bedrock":
        bedrock_config = config.get("bedrock", {})
        kwargs["aws_profile"] = bedrock_config.get("aws_profile", "default")
        kwargs["aws_region"] = bedrock_config.get("aws_region", "us-east-1")
        if not model and bedrock_config.get("model"):
            model = bedrock_config.get("model")
    elif provider == "ollama":
        ollama_config = config.get("ollama", {})
        kwargs["ollama_base_url"] = ollama_config.get("base_url", "http://localhost:11434")
        kwargs["ollama_timeout"] = ollama_config.get("timeout", 120)
    
    return get_embedding_function(provider, model, **kwargs)


# Backward compatibility - default function
def get_embedding_function_default():
    """Default embedding function for backward compatibility."""
    return get_embedding_function("ollama", "nomic-embed-text")


# New default function using config
def get_embedding_function_default_config():
    """Default embedding function using config.yaml."""
    return get_embedding_function_from_config()
