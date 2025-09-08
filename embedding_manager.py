#!/usr/bin/env python3
"""
Embedding Model Manager
Utility script to easily switch between different embedding models and providers.
"""

import argparse
import os
import sys
from typing import Dict, Any
from embedding_config import EmbeddingConfig, EmbeddingProvider


def list_models(provider: str = None):
    """List available models for a provider or all providers."""
    if provider:
        try:
            provider_enum = EmbeddingProvider(provider.lower())
            config = EmbeddingConfig(provider_enum)
            models = config.list_available_models()
            
            print(f"\n{provider.upper()} Models:")
            print("=" * 50)
            for model_name, model_info in models.items():
                is_default = " (DEFAULT)" if model_name == config.config["default_model"] else ""
                print(f"  {model_name}{is_default}")
                print(f"    Dimensions: {model_info['dimensions']}")
                print(f"    Max Tokens: {model_info['max_tokens']}")
                print(f"    Description: {model_info['description']}")
                print()
        except ValueError:
            print(f"Unknown provider: {provider}")
            print(f"Available providers: {[p.value for p in EmbeddingProvider]}")
    else:
        all_configs = EmbeddingConfig.list_all_providers_and_models()
        print("\nAll Available Embedding Models:")
        print("=" * 60)
        
        for provider_name, info in all_configs.items():
            print(f"\n{provider_name.upper()}:")
            print(f"  Default: {info['default_model']}")
            for model_name, model_info in info['models'].items():
                is_default = " (DEFAULT)" if model_name == info['default_model'] else ""
                print(f"  - {model_name}{is_default}: {model_info['description']} ({model_info['dimensions']}D)")


def test_embedding(provider: str, model: str = None, **kwargs):
    """Test an embedding model with a sample text."""
    try:
        from get_embedding_function import get_embedding_function
        
        print(f"Testing {provider} embedding model...")
        if model:
            print(f"Model: {model}")
        else:
            print("Using default model for provider")
        
        # Get embedding function
        embedding_func = get_embedding_function(provider, model, **kwargs)
        
        # Test with sample text
        test_text = "This is a test document for embedding generation."
        print(f"Sample text: '{test_text}'")
        
        # Generate embedding
        embedding = embedding_func.embed_query(test_text)
        
        print(f"✅ Success! Generated embedding with {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ Error testing embedding: {e}")
        return False
    
    return True


def setup_environment(provider: str, model: str = None, **kwargs):
    """Set up environment variables for the specified configuration."""
    env_vars = {
        "EMBEDDING_PROVIDER": provider,
    }
    
    if model:
        env_vars["EMBEDDING_MODEL"] = model
    
    # Provider-specific environment variables
    if provider == "openai" and "openai_api_key" in kwargs:
        env_vars["OPENAI_API_KEY"] = kwargs["openai_api_key"]
    elif provider == "huggingface" and "hf_api_key" in kwargs:
        env_vars["HUGGINGFACE_API_KEY"] = kwargs["hf_api_key"]
    elif provider == "bedrock":
        env_vars["AWS_PROFILE"] = kwargs.get("aws_profile", "default")
        env_vars["AWS_REGION"] = kwargs.get("aws_region", "us-east-1")
    elif provider == "ollama":
        env_vars["OLLAMA_BASE_URL"] = kwargs.get("ollama_base_url", "http://localhost:11434")
        env_vars["OLLAMA_TIMEOUT"] = str(kwargs.get("ollama_timeout", 120))
    
    print("Environment variables to set:")
    print("=" * 40)
    for key, value in env_vars.items():
        print(f"export {key}='{value}'")
    
    # Create .env file
    with open(".env", "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print(f"\n✅ Environment variables written to .env file")


def main():
    parser = argparse.ArgumentParser(description="Embedding Model Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--provider", type=str, help="Filter by provider")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test an embedding model")
    test_parser.add_argument("--provider", type=str, required=True, help="Provider to test")
    test_parser.add_argument("--model", type=str, help="Model to test (uses default if not specified)")
    test_parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    test_parser.add_argument("--hf-key", type=str, help="Hugging Face API key")
    test_parser.add_argument("--aws-profile", type=str, default="default", help="AWS profile")
    test_parser.add_argument("--aws-region", type=str, default="us-east-1", help="AWS region")
    test_parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up environment for a model")
    setup_parser.add_argument("--provider", type=str, required=True, help="Provider to set up")
    setup_parser.add_argument("--model", type=str, help="Model to set up (uses default if not specified)")
    setup_parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    setup_parser.add_argument("--hf-key", type=str, help="Hugging Face API key")
    setup_parser.add_argument("--aws-profile", type=str, default="default", help="AWS profile")
    setup_parser.add_argument("--aws-region", type=str, default="us-east-1", help="AWS region")
    setup_parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_models(args.provider)
    
    elif args.command == "test":
        kwargs = {}
        if args.openai_key:
            kwargs["openai_api_key"] = args.openai_key
        if args.hf_key:
            kwargs["hf_api_key"] = args.hf_key
        if args.aws_profile:
            kwargs["aws_profile"] = args.aws_profile
        if args.aws_region:
            kwargs["aws_region"] = args.aws_region
        if args.ollama_url:
            kwargs["ollama_base_url"] = args.ollama_url
        
        test_embedding(args.provider, args.model, **kwargs)
    
    elif args.command == "setup":
        kwargs = {}
        if args.openai_key:
            kwargs["openai_api_key"] = args.openai_key
        if args.hf_key:
            kwargs["hf_api_key"] = args.hf_key
        if args.aws_profile:
            kwargs["aws_profile"] = args.aws_profile
        if args.aws_region:
            kwargs["aws_region"] = args.aws_region
        if args.ollama_url:
            kwargs["ollama_base_url"] = args.ollama_url
        
        setup_environment(args.provider, args.model, **kwargs)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
