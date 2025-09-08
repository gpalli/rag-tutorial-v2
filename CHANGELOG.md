# Changelog

## Version 2.0 - Multi-Provider Embedding Support

### New Features

- **Multi-Provider Support**: Added support for 5 different embedding providers:
  - Ollama (local models)
  - OpenAI (cloud API)
  - Hugging Face (cloud/local models)
  - AWS Bedrock (cloud API)
  - Sentence Transformers (local models)

- **Flexible Model Selection**: Easy switching between different embedding models via:
  - Command-line arguments
  - Environment variables
  - Configuration files

- **Embedding Manager**: New utility script (`embedding_manager.py`) for:
  - Listing available models
  - Testing embedding configurations
  - Setting up environment variables

- **Enhanced Configuration**: 
  - `embedding_config.py` - Centralized model configurations
  - `config.yaml` - YAML configuration file
  - Environment variable support

### New Files

- `embedding_config.py` - Embedding model configurations and provider management
- `embedding_manager.py` - Utility for managing embedding models
- `config.yaml` - Configuration file for easy setup
- `example_usage.py` - Example usage of different embedding providers
- `CHANGELOG.md` - This changelog

### Updated Files

- `get_embedding_function.py` - Enhanced with multi-provider support
- `populate_database.py` - Added embedding provider/model parameters
- `query_data.py` - Added embedding provider/model parameters
- `test_rag.py` - Updated to work with new parameterized system
- `requirements.txt` - Added new dependencies
- `README.md` - Comprehensive documentation update

### Supported Models

#### Ollama (Local)
- `nomic-embed-text` (768D) - Default
- `mxbai-embed-large` (1024D)
- `all-minilm` (384D)

#### OpenAI (Cloud)
- `text-embedding-3-small` (1536D) - Default
- `text-embedding-3-large` (3072D)
- `text-embedding-ada-002` (1536D)

#### Hugging Face (Cloud/Local)
- `sentence-transformers/all-MiniLM-L6-v2` (384D) - Default
- `sentence-transformers/all-mpnet-base-v2` (768D)
- `intfloat/e5-large-v2` (1024D)

#### AWS Bedrock (Cloud)
- `amazon.titan-embed-text-v1` (1536D) - Default
- `amazon.titan-embed-text-v2` (1024D)

#### Sentence Transformers (Local)
- `all-MiniLM-L6-v2` (384D) - Default
- `all-mpnet-base-v2` (768D)

### Usage Examples

#### Command Line
```bash
# List available models
python3 embedding_manager.py list

# Test a model
python3 embedding_manager.py test --provider ollama --model nomic-embed-text

# Populate database with specific model
python3 populate_database.py --reset --provider ollama --model nomic-embed-text

# Query with specific model
python3 query_data.py "Your question" --provider ollama --model nomic-embed-text
```

#### Environment Variables
```bash
export EMBEDDING_PROVIDER=ollama
export EMBEDDING_MODEL=nomic-embed-text
python3 populate_database.py --reset
python3 query_data.py "Your question"
```

### Backward Compatibility

- All existing functionality remains unchanged
- Default behavior uses Ollama with `nomic-embed-text` model
- Existing scripts work without modification

### Dependencies Added

- `langchain-openai` - OpenAI integration
- `sentence-transformers` - Hugging Face sentence transformers
- `torch` - PyTorch for local models
- `transformers` - Hugging Face transformers

### Migration Guide

No migration required! The system is fully backward compatible. To use new features:

1. Install new dependencies: `pip install -r requirements.txt`
2. Use new command-line options or environment variables
3. Explore available models: `python3 embedding_manager.py list`
