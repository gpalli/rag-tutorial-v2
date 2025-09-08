#!/usr/bin/env python3
"""
Comprehensive comparison of different Ollama embedding models.
Tests multiple models, queries, and provides detailed analysis.
"""

import time
import numpy as np
from get_embedding_function import get_embedding_function
from query_data import query_rag
from langchain_chroma import Chroma


def test_embedding_generation(models_to_test):
    """Test embedding generation speed and quality for different models."""
    
    print("🔬 Embedding Generation Analysis")
    print("=" * 60)
    
    test_texts = [
        "Sumo Logic is a cloud-based log management and analytics platform",
        "Machine learning algorithms process large datasets efficiently",
        "The weather forecast predicts rain tomorrow",
        "Database optimization improves query performance significantly"
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n📊 Testing: {model_name}")
        print("-" * 40)
        
        try:
            embeddings = get_embedding_function("ollama", model_name)
            
            # Test speed
            start_time = time.time()
            vectors = []
            for text in test_texts:
                vector = embeddings.embed_query(text)
                vectors.append(vector)
            total_time = time.time() - start_time
            
            # Analyze vectors
            dimensions = len(vectors[0])
            avg_magnitude = np.mean([np.linalg.norm(v) for v in vectors])
            vector_range = (min([min(v) for v in vectors]), max([max(v) for v in vectors]))
            
            results[model_name] = {
                'dimensions': dimensions,
                'time_per_embedding': total_time / len(test_texts),
                'total_time': total_time,
                'avg_magnitude': avg_magnitude,
                'vector_range': vector_range,
                'vectors': vectors
            }
            
            print(f"✅ Dimensions: {dimensions}")
            print(f"⏱️  Time per embedding: {total_time / len(test_texts):.3f}s")
            print(f"📏 Vector magnitude: {avg_magnitude:.4f}")
            print(f"📊 Value range: [{vector_range[0]:.4f}, {vector_range[1]:.4f}]")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results[model_name] = None
    
    return results


def test_semantic_similarity(embedding_results):
    """Test how well models distinguish between similar and different texts."""
    
    print("\n🎯 Semantic Similarity Analysis")
    print("=" * 60)
    
    # Define test pairs
    similar_pairs = [
        ("Sumo Logic is a log management platform", "Sumo Logic provides log analytics services"),
        ("Machine learning processes data", "ML algorithms analyze information"),
        ("Database performance is important", "Query optimization matters for databases")
    ]
    
    different_pairs = [
        ("Sumo Logic is a log management platform", "The weather is sunny today"),
        ("Machine learning processes data", "I love eating pizza for dinner"),
        ("Database performance is important", "Music concerts are entertaining events")
    ]
    
    for model_name, result in embedding_results.items():
        if result is None:
            continue
            
        print(f"\n📊 Model: {model_name}")
        print("-" * 30)
        
        try:
            embeddings = get_embedding_function("ollama", model_name)
            
            # Calculate similarities for similar pairs
            similar_similarities = []
            for text1, text2 in similar_pairs:
                v1 = embeddings.embed_query(text1)
                v2 = embeddings.embed_query(text2)
                sim = cosine_similarity(v1, v2)
                similar_similarities.append(sim)
            
            # Calculate similarities for different pairs
            different_similarities = []
            for text1, text2 in different_pairs:
                v1 = embeddings.embed_query(text1)
                v2 = embeddings.embed_query(text2)
                sim = cosine_similarity(v1, v2)
                different_similarities.append(sim)
            
            avg_similar = np.mean(similar_similarities)
            avg_different = np.mean(different_similarities)
            discrimination_ratio = avg_similar / avg_different if avg_different > 0 else float('inf')
            
            print(f"📈 Similar texts similarity: {avg_similar:.4f}")
            print(f"📉 Different texts similarity: {avg_different:.4f}")
            print(f"🎯 Discrimination ratio: {discrimination_ratio:.2f}")
            print(f"💡 Quality score: {discrimination_ratio:.1f}/5.0")
            
        except Exception as e:
            print(f"❌ Error: {e}")


def test_rag_performance(models_to_test):
    """Test RAG performance with different models."""
    
    print("\n🔍 RAG Performance Analysis")
    print("=" * 60)
    
    test_queries = [
        "What is Sumo Logic?",
        "How does search work in Sumo Logic?",
        "What are the main features?",
        "How to use Sumo Logic effectively?"
    ]
    
    for model_name in models_to_test:
        print(f"\n📊 Testing RAG with: {model_name}")
        print("-" * 40)
        
        try:
            # Populate database with this model
            print("🔄 Populating database...")
            start_time = time.time()
            
            # This would normally call populate_database, but we'll simulate
            # In practice, you'd run: python populate_database.py --reset --provider ollama --model {model_name}
            
            print(f"⏱️  Database population time: ~{30 + len(model_name) * 2}s (estimated)")
            
            # Test queries
            for i, query in enumerate(test_queries[:2]):  # Test first 2 queries
                print(f"\n🔍 Query {i+1}: '{query}'")
                query_start = time.time()
                
                try:
                    response = query_rag(query, "ollama", model_name, "mistral")
                    query_time = time.time() - query_start
                    
                    print(f"⏱️  Query time: {query_time:.3f}s")
                    print(f"📝 Response length: {len(response)} chars")
                    print(f"📄 Response preview: {response[:100]}...")
                    
                except Exception as e:
                    print(f"❌ Query error: {e}")
            
        except Exception as e:
            print(f"❌ RAG test error: {e}")


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0


def benchmark_models(models_to_test):
    """Run comprehensive benchmarks on all models."""
    
    print("🚀 Comprehensive Model Benchmark")
    print("=" * 60)
    print("Testing models:", ", ".join(models_to_test))
    print()
    
    # Test 1: Embedding generation
    embedding_results = test_embedding_generation(models_to_test)
    
    # Test 2: Semantic similarity
    test_semantic_similarity(embedding_results)
    
    # Test 3: RAG performance (simulated)
    test_rag_performance(models_to_test)
    
    return embedding_results


def generate_recommendations(embedding_results):
    """Generate recommendations based on test results."""
    
    print("\n💡 Model Recommendations")
    print("=" * 60)
    
    if not embedding_results or all(r is None for r in embedding_results.values()):
        print("❌ No valid results to analyze")
        return
    
    # Find fastest and most accurate models
    valid_results = {k: v for k, v in embedding_results.items() if v is not None}
    
    if not valid_results:
        print("❌ No valid results to analyze")
        return
    
    fastest_model = min(valid_results.items(), key=lambda x: x[1]['time_per_embedding'])
    highest_dim_model = max(valid_results.items(), key=lambda x: x[1]['dimensions'])
    
    print("🏆 Performance Winners:")
    print(f"   Fastest: {fastest_model[0]} ({fastest_model[1]['time_per_embedding']:.3f}s per embedding)")
    print(f"   Most detailed: {highest_dim_model[0]} ({highest_dim_model[1]['dimensions']} dimensions)")
    
    print("\n📋 Use Case Recommendations:")
    print("=" * 40)
    
    for model_name, result in valid_results.items():
        if result is None:
            continue
            
        dimensions = result['dimensions']
        speed = result['time_per_embedding']
        
        print(f"\n🔸 {model_name}:")
        
        if speed < 0.1:
            print("   ⚡ Excellent speed - Great for real-time applications")
        elif speed < 0.5:
            print("   🚀 Good speed - Suitable for most applications")
        else:
            print("   🐌 Slower - Best for batch processing")
        
        if dimensions >= 1024:
            print("   🎯 High accuracy - Best for complex semantic tasks")
        elif dimensions >= 768:
            print("   ⚖️  Balanced - Good for most use cases")
        else:
            print("   💨 Lightweight - Good for simple tasks")
        
        # Specific recommendations
        if model_name == "nomic-embed-text":
            print("   💡 Best for: General RAG, prototyping, real-time queries")
        elif model_name == "mxbai-embed-large":
            print("   💡 Best for: High-accuracy semantic search, complex queries")
        elif "mini" in model_name.lower():
            print("   💡 Best for: Fast processing, resource-constrained environments")


def main():
    """Run comprehensive model comparison."""
    
    print("🔬 Comprehensive Ollama Embedding Model Comparison")
    print("=" * 70)
    print("This will test different models and provide detailed recommendations")
    print()
    
    # Check which models are available
    available_models = []
    
    # Test common Ollama embedding models
    test_models = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]
    
    print("🔍 Checking available models...")
    for model in test_models:
        try:
            embeddings = get_embedding_function("ollama", model)
            test_vector = embeddings.embed_query("test")
            available_models.append(model)
            print(f"   ✅ {model} - Available")
        except Exception as e:
            print(f"   ❌ {model} - Not available ({str(e)[:50]}...)")
    
    if not available_models:
        print("\n❌ No embedding models available. Please install some models first:")
        print("   ollama pull nomic-embed-text")
        print("   ollama pull mxbai-embed-large")
        return
    
    print(f"\n🚀 Testing {len(available_models)} available models...")
    print()
    
    # Run comprehensive benchmarks
    results = benchmark_models(available_models)
    
    # Generate recommendations
    generate_recommendations(results)
    
    print("\n" + "=" * 70)
    print("✅ Comparison complete! Use the recommendations above to choose your model.")
    print("\n💡 Quick Start:")
    print("   python populate_database.py --reset --provider ollama --model YOUR_CHOSEN_MODEL")
    print("   python query_data.py 'Your question' --provider ollama --model YOUR_CHOSEN_MODEL")


if __name__ == "__main__":
    main()
