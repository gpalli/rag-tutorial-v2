import os
from typing import Optional
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from embedding_config import EmbeddingConfig, EmbeddingProvider


def get_embedding_function(provider: str = "ollama", model: str = None, **kwargs):
    """
    Get embedding function based on provider and model.
    
    Args:
        provider: Embedding provider (ollama, openai, huggingface, bedrock, sentence_transformers)
        model: Specific model name (uses default if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        Embedding function instance
    """
    config = EmbeddingConfig(EmbeddingProvider(provider.lower()), model, **kwargs)
    
    if config.provider == EmbeddingProvider.OLLAMA:
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


def get_embedding_function_from_env():
    """
    Get embedding function from environment variables.
    Environment variables:
    - EMBEDDING_PROVIDER: Provider name (default: ollama)
    - EMBEDDING_MODEL: Model name (uses default for provider)
    - Additional provider-specific variables as needed
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
    model = os.getenv("EMBEDDING_MODEL")
    
    # Provider-specific environment variables
    kwargs = {}
    
    if provider == "openai":
        kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    elif provider == "huggingface":
        kwargs["hf_api_key"] = os.getenv("HUGGINGFACE_API_KEY")
        kwargs["hf_cache_dir"] = os.getenv("HUGGINGFACE_CACHE_DIR", "./cache")
    elif provider == "bedrock":
        kwargs["aws_profile"] = os.getenv("AWS_PROFILE", "default")
        kwargs["aws_region"] = os.getenv("AWS_REGION", "us-east-1")
    elif provider == "ollama":
        kwargs["ollama_base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        kwargs["ollama_timeout"] = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    
    return get_embedding_function(provider, model, **kwargs)


# Backward compatibility - default function
def get_embedding_function_default():
    """Default embedding function for backward compatibility."""
    return get_embedding_function("ollama", "nomic-embed-text")
