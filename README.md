# RAG Tutorial v2

A Retrieval-Augmented Generation (RAG) system that can load and query PDF, Markdown, and text files using LangChain, ChromaDB, and Ollama.

## Features

- **Multi-format support**: Loads PDF, Markdown (`.md`), and text (`.txt`) files
- **Vector search**: Uses ChromaDB for semantic similarity search
- **Local LLM**: Powered by Ollama with Mistral model
- **Chunking**: Intelligent text splitting for better retrieval
- **Source attribution**: Shows which documents were used for answers

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed
- Mistral model downloaded

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama and download Mistral model**:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull mistral
   ```

## Usage

### 1. Populate the Database

Load and process your documents into the vector database:

```bash
# First time - reset and populate the database
python populate_database.py --reset

# Or just add new documents without resetting
python populate_database.py
```

This will:
- Load all PDF files from the `data/` directory
- Load all Markdown (`.md`) and text (`.txt`) files from the `data/` directory
- Split them into chunks
- Generate embeddings
- Store everything in a Chroma vector database

### 2. Query the RAG System

Ask questions about your documents:

```bash
python query_data.py "How much money does a player start with in Monopoly?"
python query_data.py "What file types does the RAG system support?"
python query_data.py "What are the features mentioned in the sample markdown file?"
```

### 3. Run Tests (Optional)

Test the system with predefined questions:

```bash
python test_rag.py
```

## File Structure

```
rag-tutorial-v2/
├── data/                    # Place your documents here
│   ├── *.pdf               # PDF files
│   ├── *.md                # Markdown files
│   └── *.txt               # Text files
├── chroma/                 # Vector database (created automatically)
├── populate_database.py    # Script to load documents
├── query_data.py          # Script to query the system
├── test_rag.py            # Test script
├── get_embedding_function.py
└── requirements.txt
```

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama and Mistral model
ollama pull mistral

# 3. Add your documents to the data/ directory
# (PDF, .md, or .txt files)

# 4. Populate database
python populate_database.py --reset

# 5. Query the system
python query_data.py "What are the rules for Monopoly?"
python query_data.py "How many points does the longest train get in Ticket to Ride?"
```

## Supported File Types

- **PDF files**: Uses PyPDFDirectoryLoader
- **Markdown files**: Uses DirectoryLoader with TextLoader
- **Text files**: Uses DirectoryLoader with TextLoader

The system will automatically detect and process all supported file types in the `data/` directory.

## Dependencies

- `pypdf` - PDF processing
- `langchain` - Document processing and LLM integration
- `chromadb` - Vector storage
- `pytest` - Testing
- `boto3` - AWS integration (if needed)
