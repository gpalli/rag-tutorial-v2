#!/usr/bin/env python3
"""
Compare different Ollama embedding models to show their differences.
"""

import time
from get_embedding_function import get_embedding_function
from query_data import query_rag


def compare_models():
    """Compare different Ollama embedding models."""
    
    models_to_test = [
        ("nomic-embed-text", "Nomic's text embedding model"),
        ("mxbai-embed-large", "Mixedbread AI's large embedding model")
    ]
    
    test_queries = [
        "What is Sumo Logic?",
        "How does search work in Sumo Logic?",
        "What are the main features of Sumo Logic?"
    ]
    
    print("🔍 Ollama Embedding Model Comparison")
    print("=" * 60)
    
    for model_name, description in models_to_test:
        print(f"\n📊 Testing Model: {model_name}")
        print(f"Description: {description}")
        print("-" * 40)
        
        try:
            # Test embedding generation speed and dimensions
            start_time = time.time()
            embeddings = get_embedding_function("ollama", model_name)
            test_text = "This is a test document for embedding generation."
            result = embeddings.embed_query(test_text)
            embedding_time = time.time() - start_time
            
            print(f"✅ Embedding Dimensions: {len(result)}")
            print(f"⏱️  Embedding Time: {embedding_time:.3f} seconds")
            print(f"📏 Vector Range: [{min(result):.4f}, {max(result):.4f}]")
            
            # Test with a query
            print(f"\n🔍 Testing Query: '{test_queries[0]}'")
            query_start = time.time()
            response = query_rag(test_queries[0], "ollama", model_name, "mistral")
            query_time = time.time() - query_start
            
            print(f"⏱️  Query Time: {query_time:.3f} seconds")
            print(f"📝 Response Length: {len(response)} characters")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
        
        print()


def analyze_embedding_quality():
    """Analyze embedding quality differences."""
    
    print("\n🔬 Embedding Quality Analysis")
    print("=" * 60)
    
    # Test with similar and different texts
    test_texts = [
        "Sumo Logic is a cloud-based log management platform",
        "Sumo Logic provides log analytics and monitoring services", 
        "The weather is sunny today",
        "Machine learning algorithms process data"
    ]
    
    models = ["nomic-embed-text", "mxbai-embed-large"]
    
    for model_name in models:
        print(f"\n📊 Model: {model_name}")
        print("-" * 30)
        
        try:
            embeddings = get_embedding_function("ollama", model_name)
            
            # Generate embeddings for all test texts
            embedding_vectors = []
            for text in test_texts:
                vector = embeddings.embed_query(text)
                embedding_vectors.append(vector)
            
            # Calculate similarity between similar texts (Sumo Logic related)
            similar_texts = embedding_vectors[0], embedding_vectors[1]
            different_texts = embedding_vectors[0], embedding_vectors[2]
            
            # Simple cosine similarity calculation
            def cosine_similarity(a, b):
                import numpy as np
                dot_product = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                return dot_product / (norm_a * norm_b)
            
            similar_sim = cosine_similarity(similar_texts[0], similar_texts[1])
            different_sim = cosine_similarity(different_texts[0], different_texts[1])
            
            print(f"📈 Similar texts similarity: {similar_sim:.4f}")
            print(f"📉 Different texts similarity: {different_sim:.4f}")
            print(f"🎯 Discrimination ratio: {similar_sim/different_sim:.2f}")
            
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {e}")


def main():
    """Run the complete comparison."""
    print("🚀 Starting Ollama Embedding Model Comparison")
    print("This will show you the practical differences between models")
    print()
    
    compare_models()
    analyze_embedding_quality()
    
    print("\n" + "=" * 60)
    print("📋 Summary of Key Differences:")
    print("=" * 60)
    print("1. DIMENSIONS:")
    print("   - nomic-embed-text: 768 dimensions (balanced)")
    print("   - mxbai-embed-large: 1024 dimensions (more detailed)")
    print()
    print("2. PERFORMANCE:")
    print("   - Higher dimensions = more memory usage")
    print("   - Larger models = slower processing")
    print("   - More dimensions = potentially better accuracy")
    print()
    print("3. USE CASES:")
    print("   - nomic-embed-text: Good balance of speed and quality")
    print("   - mxbai-embed-large: Better for complex semantic understanding")
    print()
    print("4. STORAGE:")
    print("   - 768D vectors: ~3KB per embedding")
    print("   - 1024D vectors: ~4KB per embedding")
    print()
    print("💡 Recommendation: Start with nomic-embed-text for most use cases,")
    print("   upgrade to mxbai-embed-large if you need higher accuracy.")


if __name__ == "__main__":
    main()
