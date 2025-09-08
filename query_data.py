import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

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
    parser.add_argument("--llm-model", type=str, default="mistral",
                       help="LLM model to use for generating responses")
    
    args = parser.parse_args()
    query_text = args.query_text
    
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
        print(f"🔧 Using embedding model: {args.model}")
    else:
        print(f"🔧 Using default embedding model for {args.provider}")
    print(f"🔧 Using LLM model: {args.llm_model}")
    
    query_rag(query_text, args.provider, args.model, args.llm_model, **embedding_kwargs)


def query_rag(query_text: str, provider: str = "ollama", model: str = None, 
              llm_model: str = "mistral", **kwargs):
    # Prepare the DB.
    embedding_function = get_embedding_function(provider, model, **kwargs)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model=llm_model)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
