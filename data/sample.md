# Sample Markdown Document

This is a sample Markdown file to test the RAG system.

## Features

- Support for PDF files
- Support for Markdown files
- Vector database storage
- Semantic search capabilities

## Usage

The populate_database.py script can now load both PDF and Markdown files from the data directory.

### Code Example

```python
# Load documents
documents = load_documents()
chunks = split_documents(documents)
add_to_chroma(chunks)
```

This ensures that all your documents are properly indexed and searchable.
