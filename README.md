# RAG Tutorial v2

A flexible Retrieval-Augmented Generation (RAG) system that supports multiple embedding models and providers. Load and query PDF, Markdown, and text files using LangChain, ChromaDB, and various embedding services.

## Features

- **Multi-format support**: Loads PDF, Markdown (`.md`), text (`.txt`), and RTF (`.rtf`) files
- **Multiple embedding providers**: Ollama, OpenAI, Hugging Face, AWS Bedrock, Sentence Transformers
- **Flexible model selection**: Easy switching between different embedding models
- **Vector search**: Uses ChromaDB for semantic similarity search
- **Local LLM**: Powered by Ollama with configurable models
- **Chunking**: Intelligent text splitting for better retrieval
- **Source attribution**: Shows which documents were used for answers
- **Environment-based configuration**: Easy setup via environment variables

## Prerequisites

- Python 3.8+ (recommended: Python 3.10)
- [pyenv](https://github.com/pyenv/pyenv) (recommended for Python version management)
- [Ollama](https://ollama.ai) installed (for local embedding models)
- Optional: API keys for cloud providers (OpenAI, Hugging Face, AWS)

## Installation

### 1. Python Environment Setup

**Option A: Using pyenv (Recommended)**
```bash
# Install pyenv if not already installed
# Follow instructions at https://github.com/pyenv/pyenv#installation

# Install Python 3.10.0
pyenv install 3.10.0

# Set local Python version for this project
pyenv local 3.10.0

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

**Option B: Using system Python**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Python Dependencies

```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

### 3. Install Ollama and Download Models (Optional)

```bash
# Install Ollama from https://ollama.ai
# Then download embedding and LLM models
ollama pull nomic-embed-text  # Embedding model
ollama pull mistral          # LLM model
```

### 4. Set Up API Keys (Optional)

For cloud-based embedding providers, set up your API keys:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Hugging Face
export HUGGINGFACE_API_KEY="your-hf-api-key"

# AWS (for Bedrock)
export AWS_PROFILE="your-aws-profile"
export AWS_REGION="us-east-1"
```

## Usage

### 1. Choose Your Embedding Model

The system supports multiple embedding providers. Use the embedding manager to explore options:

```bash
# List all available models
python embedding_manager.py list

# List models for a specific provider
python embedding_manager.py list --provider ollama

# Test a specific model
python embedding_manager.py test --provider ollama --model nomic-embed-text

# Set up environment for a model
python embedding_manager.py setup --provider ollama --model nomic-embed-text
```

### 2. Populate the Database

Load and process your documents into the vector database:

```bash
# Using Ollama (default)
python populate_database.py --reset

# Using OpenAI
python populate_database.py --reset --provider openai --model text-embedding-3-small --openai-key your-key

# Using Hugging Face
python populate_database.py --reset --provider huggingface --model sentence-transformers/all-MiniLM-L6-v2

# Using environment variables
export EMBEDDING_PROVIDER=ollama
export EMBEDDING_MODEL=nomic-embed-text
python populate_database.py --reset
```

This will:
- Load all PDF, Markdown (`.md`), text (`.txt`), and RTF (`.rtf`) files from the `data/` directory
- Split them into chunks
- Generate embeddings using your chosen model
- Store everything in a Chroma vector database

### 3. Query the RAG System

Ask questions about your documents:

```bash
# Using default settings
python query_data.py "How much money does a player start with in Monopoly?"

# Using specific embedding model
python query_data.py "What file types does the RAG system support?" --provider ollama --model nomic-embed-text

# Using different LLM model
python query_data.py "What are the features mentioned?" --llm-model llama2

# Using OpenAI embeddings
python query_data.py "Your question" --provider openai --model text-embedding-3-small --openai-key your-key
```

### 4. Run Tests (Optional)

Test the system with predefined questions:

```bash
# Using default settings
python test_rag.py

# Using specific configuration
python -c "
from test_rag import query_and_validate
query_and_validate('How much total money does a player start with?', '$1500', 'ollama', 'nomic-embed-text')
"
```

## File Structure

```
rag-tutorial-v2/
├── data/                    # Place your documents here
│   ├── *.pdf               # PDF files
│   ├── *.md                # Markdown files
│   ├── *.txt               # Text files
│   └── *.rtf               # RTF files
├── chroma/                 # Vector database (created automatically)
├── populate_database.py    # Script to load documents
├── query_data.py          # Script to query the system
├── test_rag.py            # Test script
├── get_embedding_function.py  # Embedding function factory
├── embedding_config.py     # Embedding model configurations
├── embedding_manager.py    # Utility for managing embedding models
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (created by setup)
```

## Supported Embedding Providers

### Ollama (Local)
- **nomic-embed-text**: High-quality text embeddings (768D)
- **mxbai-embed-large**: Large embedding model (1024D)
- **all-minilm**: Lightweight model (384D)

### OpenAI (Cloud)
- **text-embedding-3-small**: Latest small model (1536D)
- **text-embedding-3-large**: Latest large model (3072D)
- **text-embedding-ada-002**: Previous generation (1536D)

### Hugging Face (Cloud/Local)
- **sentence-transformers/all-MiniLM-L6-v2**: Popular model (384D)
- **sentence-transformers/all-mpnet-base-v2**: High-quality (768D)
- **intfloat/e5-large-v2**: Microsoft E5 model (1024D)

### AWS Bedrock (Cloud)
- **amazon.titan-embed-text-v1**: AWS Titan v1 (1536D)
- **amazon.titan-embed-text-v2**: AWS Titan v2 (1024D)

### Sentence Transformers (Local)
- **all-MiniLM-L6-v2**: Lightweight (384D)
- **all-mpnet-base-v2**: High-quality (768D)

## Example Workflows

### Basic Workflow (Ollama)

```bash
# 1. Set up Python environment
pyenv local 3.10.0
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama and models
ollama pull nomic-embed-text
ollama pull mistral

# 4. Add your documents to the data/ directory
# (PDF, .md, .txt, or .rtf files)

# 5. Populate database
python populate_database.py --reset

# 6. Query the system
python query_data.py "What are the rules for Monopoly?"
```

### OpenAI Workflow

```bash
# 1. Set up environment
export OPENAI_API_KEY="your-api-key"

# 2. Populate database with OpenAI embeddings
python populate_database.py --reset --provider openai --model text-embedding-3-small

# 3. Query with OpenAI embeddings
python query_data.py "Your question" --provider openai --model text-embedding-3-small
```

### Hugging Face Workflow

```bash
# 1. Set up environment (optional for public models)
export HUGGINGFACE_API_KEY="your-api-key"

# 2. Populate database with Hugging Face embeddings
python populate_database.py --reset --provider huggingface --model sentence-transformers/all-MiniLM-L6-v2

# 3. Query with Hugging Face embeddings
python query_data.py "Your question" --provider huggingface --model sentence-transformers/all-MiniLM-L6-v2
```

### Environment-Based Configuration

```bash
# 1. Set up environment variables
export EMBEDDING_PROVIDER=ollama
export EMBEDDING_MODEL=nomic-embed-text
export OLLAMA_BASE_URL=http://localhost:11434

# 2. Use scripts without additional parameters
python populate_database.py --reset
python query_data.py "Your question"
```

## Supported File Types

- **PDF files**: Uses PyPDFDirectoryLoader
- **Markdown files**: Uses DirectoryLoader with TextLoader
- **Text files**: Uses DirectoryLoader with TextLoader
- **RTF files**: Uses DirectoryLoader with TextLoader

The system will automatically detect and process all supported file types in the `data/` directory.

## Dependencies

- `pypdf` - PDF processing
- `langchain` - Document processing and LLM integration
- `langchain-ollama` - Ollama integration
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community integrations
- `langchain-chroma` - ChromaDB integration
- `chromadb` - Vector storage
- `sentence-transformers` - Hugging Face sentence transformers
- `torch` - PyTorch for local models
- `transformers` - Hugging Face transformers
- `pytest` - Testing
- `boto3` - AWS integration (for Bedrock)

## Troubleshooting

### Common Issues

1. **Ollama connection errors**: Make sure Ollama is running (`ollama serve`)
2. **API key errors**: Verify your API keys are set correctly
3. **Model not found**: Ensure the model is downloaded (for Ollama) or available (for cloud providers)
4. **Memory issues**: Use smaller models or reduce chunk size for large documents

### Getting Help

- Check the embedding manager: `python embedding_manager.py list`
- Test your configuration: `python embedding_manager.py test --provider your-provider`
- View available models: `python embedding_manager.py list --provider your-provider`
