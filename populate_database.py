import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser(description="Populate the vector database with documents")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--provider", type=str, default="ollama", 
                       choices=["ollama", "openai", "huggingface", "bedrock", "sentence_transformers"],
                       help="Embedding provider to use")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific embedding model to use (uses default for provider if not specified)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                       help="Ollama base URL (only used with ollama provider)")
    parser.add_argument("--openai-key", type=str, default=None,
                       help="OpenAI API key (only used with openai provider)")
    parser.add_argument("--hf-key", type=str, default=None,
                       help="Hugging Face API key (only used with huggingface provider)")
    parser.add_argument("--aws-profile", type=str, default="default",
                       help="AWS profile name (only used with bedrock provider)")
    parser.add_argument("--aws-region", type=str, default="us-east-1",
                       help="AWS region (only used with bedrock provider)")
    
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Prepare embedding configuration
    embedding_kwargs = {}
    if args.provider == "ollama":
        embedding_kwargs["ollama_base_url"] = args.ollama_url
    elif args.provider == "openai":
        embedding_kwargs["openai_api_key"] = args.openai_key
    elif args.provider == "huggingface":
        embedding_kwargs["hf_api_key"] = args.hf_key
    elif args.provider == "bedrock":
        embedding_kwargs["aws_profile"] = args.aws_profile
        embedding_kwargs["aws_region"] = args.aws_region

    print(f"🔧 Using embedding provider: {args.provider}")
    if args.model:
        print(f"🔧 Using model: {args.model}")
    else:
        print(f"🔧 Using default model for {args.provider}")

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks, args.provider, args.model, **embedding_kwargs)


def load_documents():
    documents = []
    
    # Load PDF files
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    pdf_documents = pdf_loader.load()
    documents.extend(pdf_documents)
    
    # Load Markdown files (recursively)
    md_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    md_documents = md_loader.load()
    documents.extend(md_documents)
    
    # Load TXT files (recursively)
    txt_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    txt_documents = txt_loader.load()
    documents.extend(txt_documents)
    
    # Load RTF files (recursively)
    rtf_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.rtf",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    rtf_documents = rtf_loader.load()
    documents.extend(rtf_documents)
    
    print(f"Loaded {len(pdf_documents)} PDF documents, {len(md_documents)} Markdown documents, {len(txt_documents)} TXT documents, and {len(rtf_documents)} RTF documents")
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], provider: str = "ollama", model: str = None, **kwargs):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(provider, model, **kwargs)
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("✅ Documents added successfully")
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
