#!/usr/bin/env python3
"""
Side-by-side comparison of different Ollama embedding models.
Tests the same queries with different models to show practical differences.
"""

import time
from query_data import query_rag


def compare_models_side_by_side():
    """Compare models side by side with the same queries."""
    
    print("🔄 Side-by-Side Model Comparison")
    print("=" * 70)
    print("Testing the same queries with different embedding models")
    print()
    
    # Test queries
    test_queries = [
        "What is Sumo Logic?",
        "How does search work in Sumo Logic?", 
        "What are the main features of Sumo Logic?",
        "How to use Sumo Logic effectively?",
        "What are the security features?"
    ]
    
    models_to_test = [
        ("nomic-embed-text", "Nomic's text embedding (768D)"),
        ("mxbai-embed-large", "Mixedbread AI large (1024D)")
    ]
    
    results = {}
    
    for model_name, description in models_to_test:
        print(f"\n🔧 Testing with: {model_name}")
        print(f"Description: {description}")
        print("=" * 50)
        
        model_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                response = query_rag(query, "ollama", model_name, "mistral")
                query_time = time.time() - start_time
                
                # Extract key metrics
                response_length = len(response)
                sources_count = response.count("Sources:")
                
                model_results.append({
                    'query': query,
                    'response': response,
                    'time': query_time,
                    'length': response_length,
                    'sources': sources_count
                })
                
                print(f"⏱️  Time: {query_time:.2f}s")
                print(f"📏 Length: {response_length} chars")
                print(f"📄 Sources: {sources_count}")
                print(f"💬 Response: {response[:150]}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                model_results.append({
                    'query': query,
                    'response': f"Error: {e}",
                    'time': 0,
                    'length': 0,
                    'sources': 0
                })
        
        results[model_name] = model_results
        print(f"\n✅ Completed testing {model_name}")
    
    return results


def analyze_results(results):
    """Analyze and compare the results."""
    
    print("\n📊 Analysis & Comparison")
    print("=" * 70)
    
    for model_name, model_results in results.items():
        print(f"\n🔸 {model_name.upper()}")
        print("-" * 30)
        
        # Calculate averages
        avg_time = sum(r['time'] for r in model_results) / len(model_results)
        avg_length = sum(r['length'] for r in model_results) / len(model_results)
        avg_sources = sum(r['sources'] for r in model_results) / len(model_results)
        
        print(f"⏱️  Average query time: {avg_time:.2f}s")
        print(f"📏 Average response length: {avg_length:.0f} chars")
        print(f"📄 Average sources: {avg_sources:.1f}")
        
        # Find best and worst performing queries
        times = [r['time'] for r in model_results if r['time'] > 0]
        if times:
            fastest_query = min(model_results, key=lambda x: x['time'] if x['time'] > 0 else float('inf'))
            slowest_query = max(model_results, key=lambda x: x['time'])
            
            print(f"🚀 Fastest query: {fastest_query['query'][:30]}... ({fastest_query['time']:.2f}s)")
            print(f"🐌 Slowest query: {slowest_query['query'][:30]}... ({slowest_query['time']:.2f}s)")


def compare_responses(results):
    """Compare responses for the same queries."""
    
    print("\n🔍 Response Quality Comparison")
    print("=" * 70)
    
    # Get the first model's queries as reference
    first_model = list(results.keys())[0]
    queries = [r['query'] for r in results[first_model]]
    
    for i, query in enumerate(queries):
        print(f"\n📝 Query {i+1}: {query}")
        print("=" * 50)
        
        for model_name, model_results in results.items():
            if i < len(model_results):
                result = model_results[i]
                print(f"\n🔸 {model_name}:")
                print(f"   Time: {result['time']:.2f}s | Length: {result['length']} chars")
                print(f"   Response: {result['response'][:200]}...")
                if result['response'].startswith("Error:"):
                    print(f"   ❌ {result['response']}")


def generate_final_recommendations(results):
    """Generate final recommendations based on all tests."""
    
    print("\n💡 Final Recommendations")
    print("=" * 70)
    
    if len(results) < 2:
        print("❌ Need at least 2 models to compare")
        return
    
    # Calculate overall performance metrics
    model_metrics = {}
    
    for model_name, model_results in results.items():
        valid_results = [r for r in model_results if r['time'] > 0]
        if not valid_results:
            continue
            
        avg_time = sum(r['time'] for r in valid_results) / len(valid_results)
        avg_length = sum(r['length'] for r in valid_results) / len(valid_results)
        success_rate = len(valid_results) / len(model_results)
        
        model_metrics[model_name] = {
            'avg_time': avg_time,
            'avg_length': avg_length,
            'success_rate': success_rate
        }
    
    # Find winners
    fastest = min(model_metrics.items(), key=lambda x: x[1]['avg_time'])
    most_detailed = max(model_metrics.items(), key=lambda x: x[1]['avg_length'])
    most_reliable = max(model_metrics.items(), key=lambda x: x[1]['success_rate'])
    
    print("🏆 Performance Winners:")
    print(f"   🚀 Fastest: {fastest[0]} ({fastest[1]['avg_time']:.2f}s avg)")
    print(f"   📝 Most detailed: {most_detailed[0]} ({most_detailed[1]['avg_length']:.0f} chars avg)")
    print(f"   ✅ Most reliable: {most_reliable[0]} ({most_reliable[1]['success_rate']*100:.0f}% success)")
    
    print("\n🎯 When to use each model:")
    print("-" * 40)
    
    for model_name, metrics in model_metrics.items():
        print(f"\n🔸 {model_name}:")
        
        if metrics['avg_time'] < 5:
            print("   ⚡ Very fast - Great for real-time applications")
        elif metrics['avg_time'] < 10:
            print("   🚀 Fast - Good for interactive applications")
        else:
            print("   🐌 Slower - Better for batch processing")
        
        if metrics['avg_length'] > 300:
            print("   📝 Detailed responses - Good for complex queries")
        else:
            print("   💬 Concise responses - Good for quick answers")
        
        if metrics['success_rate'] > 0.9:
            print("   ✅ Very reliable - Production ready")
        elif metrics['success_rate'] > 0.7:
            print("   ⚠️  Mostly reliable - Good for testing")
        else:
            print("   ❌ Less reliable - Needs troubleshooting")
        
        # Specific recommendations
        if "nomic" in model_name.lower():
            print("   💡 Best for: General use, prototyping, real-time queries")
        elif "mxbai" in model_name.lower():
            print("   💡 Best for: High-accuracy search, complex semantic tasks")


def main():
    """Run the complete side-by-side comparison."""
    
    print("🔬 Side-by-Side Ollama Model Comparison")
    print("=" * 70)
    print("This will test the same queries with different models")
    print("to show you the practical differences in real usage.")
    print()
    
    # Run the comparison
    results = compare_models_side_by_side()
    
    # Analyze results
    analyze_results(results)
    
    # Compare responses
    compare_responses(results)
    
    # Generate recommendations
    generate_final_recommendations(results)
    
    print("\n" + "=" * 70)
    print("✅ Side-by-side comparison complete!")
    print("\n💡 Next steps:")
    print("   1. Choose your preferred model based on the analysis above")
    print("   2. Use it consistently for your RAG system")
    print("   3. Monitor performance in your specific use case")


if __name__ == "__main__":
    main()
