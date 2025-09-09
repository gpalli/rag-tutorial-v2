import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function, get_embedding_function_from_config
from config_loader import get_database_config, get_llm_config, get_query_config, get_embedding_config

# Load configuration from config.yaml
db_config = get_database_config()
llm_config = get_llm_config()
query_config = get_query_config()

CHROMA_PATH = db_config.get("chroma_path", "chroma")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--use-config", action="store_true", default=True,
                       help="Use configuration from config.yaml (default)")
    parser.add_argument("--provider", type=str, default=None, 
                       choices=["ollama", "openai", "huggingface", "bedrock", "sentence_transformers"],
                       help="Embedding provider to use (overrides config.yaml)")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific embedding model to use (overrides config.yaml)")
    parser.add_argument("--ollama-url", type=str, default=None,
                       help="Ollama base URL (overrides config.yaml)")
    parser.add_argument("--openai-key", type=str, default=None,
                       help="OpenAI API key (overrides config.yaml)")
    parser.add_argument("--hf-key", type=str, default=None,
                       help="Hugging Face API key (overrides config.yaml)")
    parser.add_argument("--aws-profile", type=str, default=None,
                       help="AWS profile name (overrides config.yaml)")
    parser.add_argument("--aws-region", type=str, default=None,
                       help="AWS region (overrides config.yaml)")
    parser.add_argument("--llm-model", type=str, default=None,
                       help="LLM model to use for generating responses (overrides config.yaml)")
    
    args = parser.parse_args()
    query_text = args.query_text
    
    # Prepare embedding configuration
    try:
        if args.use_config:
            # Use config.yaml as primary source
            embedding_function = get_embedding_function_from_config()
            
            # Override with command line arguments if provided
            if args.provider or args.model or any([args.ollama_url, args.openai_key, args.hf_key, args.aws_profile, args.aws_region]):
                embedding_kwargs = {}
                provider = args.provider
                model = args.model
                
                if args.provider == "ollama":
                    embedding_kwargs["ollama_base_url"] = args.ollama_url
                elif args.provider == "openai":
                    embedding_kwargs["openai_api_key"] = args.openai_key
                elif args.provider == "huggingface":
                    embedding_kwargs["hf_api_key"] = args.hf_key
                elif args.provider == "bedrock":
                    embedding_kwargs["aws_profile"] = args.aws_profile
                    embedding_kwargs["aws_region"] = args.aws_region
                
                if provider or model or embedding_kwargs:
                    # If only model is provided, use the provider from config
                    if not provider:
                        config = get_embedding_config()
                        provider = config.get("provider", "ollama")
                    embedding_function = get_embedding_function(provider, model, **embedding_kwargs)
        else:
            # Use command line arguments only
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
            
            embedding_function = get_embedding_function(args.provider, args.model, **embedding_kwargs)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return
    
    # Prepare LLM configuration
    llm_model = args.llm_model or llm_config.get("model", "llama3.1:latest")
    
    # Get provider and model info for display
    if hasattr(embedding_function, 'model_name'):
        model_name = embedding_function.model_name
    else:
        model_name = "Unknown"
    
    print(f"🔧 Using embedding function: {type(embedding_function).__name__}")
    print(f"🔧 Using embedding model: {model_name}")
    print(f"🔧 Using LLM model: {llm_model}")
    
    query_rag(query_text, embedding_function, llm_model)


def query_rag(query_text: str, embedding_function, llm_model: str = "llama3.1:latest"):
    try:
        # Prepare the DB.
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    except Exception as e:
        print(f"❌ Error: Failed to initialize database with embedding function: {str(e)}")
        return None

    # Search the DB.
    max_results = query_config.get("max_results", 5)
    results = db.similarity_search_with_score(query_text, k=max_results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # Configure LLM with parameters from config
    llm_kwargs = {}
    if llm_config.get("temperature"):
        llm_kwargs["temperature"] = llm_config.get("temperature")
    if llm_config.get("max_tokens"):
        llm_kwargs["num_predict"] = llm_config.get("max_tokens")
    
    model = OllamaLLM(model=llm_model, **llm_kwargs)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
