#!/usr/bin/env python3
"""
Example usage of the enhanced RAG system with different embedding models.
This script demonstrates how to use various embedding providers and models.
"""

import os
from get_embedding_function import get_embedding_function, get_embedding_function_from_config
from embedding_config import EmbeddingConfig


def example_ollama_usage():
    """Example using Ollama embedding models."""
    print("=== Ollama Embedding Example ===")
    
    # Using nomic-embed-text (default)
    embeddings = get_embedding_function("ollama", "nomic-embed-text")
    test_text = "This is a test document for embedding generation."
    result = embeddings.embed_query(test_text)
    print(f"Ollama nomic-embed-text: {len(result)} dimensions")
    
    # Using mxbai-embed-large
    try:
        embeddings = get_embedding_function("ollama", "mxbai-embed-large")
        result = embeddings.embed_query(test_text)
        print(f"Ollama mxbai-embed-large: {len(result)} dimensions")
    except Exception as e:
        print(f"mxbai-embed-large not available: {e}")


def example_openai_usage():
    """Example using OpenAI embedding models."""
    print("\n=== OpenAI Embedding Example ===")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return
    
    try:
        # Using text-embedding-3-small
        embeddings = get_embedding_function("openai", "text-embedding-3-small")
        test_text = "This is a test document for embedding generation."
        result = embeddings.embed_query(test_text)
        print(f"OpenAI text-embedding-3-small: {len(result)} dimensions")
    except Exception as e:
        print(f"OpenAI embedding failed: {e}")


def example_huggingface_usage():
    """Example using Hugging Face embedding models."""
    print("\n=== Hugging Face Embedding Example ===")
    
    try:
        # Using sentence-transformers model
        embeddings = get_embedding_function("huggingface", "sentence-transformers/all-MiniLM-L6-v2")
        test_text = "This is a test document for embedding generation."
        result = embeddings.embed_query(test_text)
        print(f"Hugging Face all-MiniLM-L6-v2: {len(result)} dimensions")
    except Exception as e:
        print(f"Hugging Face embedding failed: {e}")


def example_sentence_transformers_usage():
    """Example using Sentence Transformers directly."""
    print("\n=== Sentence Transformers Example ===")
    
    try:
        # Using sentence-transformers directly
        embeddings = get_embedding_function("sentence_transformers", "all-MiniLM-L6-v2")
        test_text = "This is a test document for embedding generation."
        result = embeddings.embed_query(test_text)
        print(f"Sentence Transformers all-MiniLM-L6-v2: {len(result)} dimensions")
    except Exception as e:
        print(f"Sentence Transformers embedding failed: {e}")


def example_config_based():
    """Example using config.yaml-based configuration."""
    print("\n=== Config-based Configuration Example ===")
    
    try:
        embeddings = get_embedding_function_from_config()
        test_text = "This is a test document for embedding generation."
        result = embeddings.embed_query(test_text)
        print(f"Config-based configuration: {len(result)} dimensions")
    except Exception as e:
        print(f"Config-based configuration failed: {e}")


def list_available_models():
    """List all available models and providers."""
    print("\n=== Available Models ===")
    
    all_configs = EmbeddingConfig.list_all_providers_and_models()
    for provider, info in all_configs.items():
        print(f"\n{provider.upper()}:")
        print(f"  Default: {info['default_model']}")
        for model_name, model_info in info['models'].items():
            is_default = " (DEFAULT)" if model_name == info['default_model'] else ""
            print(f"  - {model_name}{is_default}: {model_info['description']} ({model_info['dimensions']}D)")


def main():
    """Run all examples."""
    print("RAG System - Embedding Model Examples")
    print("=" * 50)
    
    # List available models
    list_available_models()
    
    # Run examples
    example_ollama_usage()
    example_openai_usage()
    example_huggingface_usage()
    example_sentence_transformers_usage()
    example_config_based()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use these models in your RAG system:")
    print("1. Populate database: python populate_database.py --provider ollama --model nomic-embed-text")
    print("2. Query system: python query_data.py 'Your question' --provider ollama --model nomic-embed-text")


if __name__ == "__main__":
    main()
