#!/usr/bin/env python3
"""
Basic ToW Architecture Usage Example
===================================

This example demonstrates how to use the Thoughts of Words (ToW) architecture
to improve multilingual LLM accuracy through English intermediary reasoning.

Features demonstrated:
- Model adapter initialization (DeepSeek, Llama, Qwen)
- ToW Engine setup and configuration
- English thought generation
- Cognitive bridging to target language
- Multilingual output generation
- Quality assessment and validation
"""

import os
import sys
import time
from pathlib import Path

# Add the tow_architecture to Python path
sys.path.append(str(Path(__file__).parent.parent))

from tow_architecture import ToWEngine
from tow_architecture.models import ModelAdapterFactory, ModelConfig, ModelType
from tow_architecture.core import ToWRequest
from tow_architecture.utils import setup_tow_logging, ToWConfig


def main():
    """Main demonstration function"""
    print("🚀 Thoughts of Words (ToW) Architecture Demo")
    print("=" * 60)
    
    # Setup logging
    logger = setup_tow_logging(level="INFO", log_dir="logs")
    
    # Configuration
    model_path = "/scratch/jsong132/Increase_MLLM_Ability/DeepSeek_R1_Distill_Llama_70B"
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("Please update the model_path variable to point to your model.")
        return
    
    try:
        # Step 1: Create and load model adapter
        print("\n🔧 Step 1: Initializing Model Adapter")
        print("-" * 40)
        
        model_adapter = ModelAdapterFactory.create_optimized_config(
            model_path=model_path,
            quantization="4bit",  # Use 4-bit quantization for memory efficiency
            max_memory={0: "70GB", 1: "70GB"}  # Adjust for your GPU setup
        )
        
        # Create the adapter
        adapter = ModelAdapterFactory.create_adapter(
            model_path=model_path,
            model_type=ModelType.DEEPSEEK,
            **model_adapter.__dict__
        )
        
        print(f"✅ Created {type(adapter).__name__}")
        
        # Load the model
        print("🔄 Loading model... (this may take a few minutes)")
        if not adapter.load_model():
            print("❌ Failed to load model")
            return
        
        print("✅ Model loaded successfully")
        print(f"📊 Memory usage: {adapter.get_memory_usage()}")
        
        # Step 2: Initialize ToW Engine
        print("\n🧠 Step 2: Initializing ToW Engine")
        print("-" * 40)
        
        # Create ToW configuration
        tow_config = ToWConfig()
        tow_config.max_thought_tokens = 3  # Generate 3 thought tokens
        tow_config.thought_processor.max_thought_length = 150
        tow_config.cognitive_bridge.enable_cultural_adaptation = True
        
        # Create ToW Engine
        tow_engine = ToWEngine(
            model_adapter=adapter,
            config=tow_config
        )
        
        print("✅ ToW Engine initialized")
        
        # Step 3: Demonstrate ToW processing
        print("\n🎯 Step 3: ToW Processing Examples")
        print("-" * 40)
        
        # Example 1: Korean translation with reasoning
        example_1()
        
        # Example 2: Mathematical problem in Chinese
        example_2()
        
        # Example 3: Creative writing in Japanese
        example_3()
        
        # Step 4: Performance statistics
        print("\n📈 Step 4: Performance Statistics")
        print("-" * 40)
        
        stats = tow_engine.get_statistics()
        print_statistics(stats)
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
    
    finally:
        # Cleanup
        if 'adapter' in locals():
            adapter.clear_cache()
        print("\n🧹 Cleanup completed")


def example_1():
    """Example 1: Korean translation with reasoning"""
    print("\n📝 Example 1: Korean Translation with Reasoning")
    print("Input: 'The weather is really nice today, so I want to go for a walk in the park.'")
    
    # Create request
    request = ToWRequest(
        text="The weather is really nice today, so I want to go for a walk in the park.",
        source_language="en",
        target_language="ko",
        task_type="translation",
        max_tokens=512,
        temperature=0.7
    )
    
    # Process with ToW
    start_time = time.time()
    response = tow_engine.process(request)
    processing_time = time.time() - start_time
    
    print(f"⏱️  Processing time: {processing_time:.2f}s")
    print(f"🤔 Thought tokens:")
    for i, thought in enumerate(response.thought_tokens, 1):
        print(f"   {i}. {thought}")
    
    print(f"🎯 Korean output: {response.output_text}")
    print(f"📊 Confidence: {response.confidence_score:.3f}")
    print(f"🏆 Quality metrics:")
    print(f"   - Language quality: {response.metadata.get('validation', {}).get('language_consistency', 0.0):.3f}")
    print(f"   - Thought alignment: {response.metadata.get('validation', {}).get('thought_alignment', 0.0):.3f}")


def example_2():
    """Example 2: Mathematical problem in Chinese"""
    print("\n🧮 Example 2: Mathematical Problem in Chinese")
    print("Input: 'Solve the quadratic equation: 2x² + 5x - 3 = 0'")
    
    request = ToWRequest(
        text="Solve the quadratic equation: 2x² + 5x - 3 = 0",
        source_language="en", 
        target_language="zh",
        task_type="reasoning",
        max_tokens=768,
        temperature=0.5  # Lower temperature for mathematical precision
    )
    
    start_time = time.time()
    response = tow_engine.process(request)
    processing_time = time.time() - start_time
    
    print(f"⏱️  Processing time: {processing_time:.2f}s")
    print(f"🤔 Thought tokens:")
    for i, thought in enumerate(response.thought_tokens, 1):
        print(f"   {i}. {thought}")
    
    print(f"🎯 Chinese output: {response.output_text}")
    print(f"📊 Confidence: {response.confidence_score:.3f}")


def example_3():
    """Example 3: Creative writing in Japanese"""
    print("\n✍️  Example 3: Creative Writing in Japanese")
    print("Input: 'Write a short story about a robot learning to paint'")
    
    request = ToWRequest(
        text="Write a short story about a robot learning to paint",
        source_language="en",
        target_language="ja", 
        task_type="generation",
        max_tokens=1024,
        temperature=0.8  # Higher temperature for creativity
    )
    
    start_time = time.time()
    response = tow_engine.process(request)
    processing_time = time.time() - start_time
    
    print(f"⏱️  Processing time: {processing_time:.2f}s")
    print(f"🤔 Thought tokens:")
    for i, thought in enumerate(response.thought_tokens, 1):
        print(f"   {i}. {thought}")
    
    print(f"🎯 Japanese output: {response.output_text}")
    print(f"📊 Confidence: {response.confidence_score:.3f}")


def print_statistics(stats):
    """Print ToW processing statistics"""
    print(f"📊 Total requests: {stats['total_requests']}")
    print(f"✅ Successful requests: {stats['successful_requests']}")
    print(f"📈 Success rate: {stats['success_rate']:.1%}")
    print(f"⏱️  Average processing time: {stats['avg_processing_time']:.2f}s")
    
    print(f"\n🔄 Stage performance:")
    for stage, times in stats['stage_times'].items():
        if isinstance(times, dict) and times['count'] > 0:
            print(f"   {stage}: {times['avg']:.3f}s (avg)")


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n🔄 Batch Processing Demo")
    print("-" * 30)
    
    # Multiple requests
    requests = [
        ToWRequest(
            text="Hello, how are you today?",
            target_language="ko"
        ),
        ToWRequest(
            text="What is artificial intelligence?",
            target_language="zh"
        ),
        ToWRequest(
            text="Explain quantum computing simply",
            target_language="ja"
        )
    ]
    
    start_time = time.time()
    
    # Process batch
    results = []
    for i, request in enumerate(requests, 1):
        print(f"Processing request {i}/{len(requests)}...")
        response = tow_engine.process(request)
        results.append(response)
    
    total_time = time.time() - start_time
    
    print(f"\n📊 Batch results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average per request: {total_time/len(requests):.2f}s")
    
    for i, result in enumerate(results, 1):
        print(f"   Request {i}: {result.confidence_score:.3f} confidence")


if __name__ == "__main__":
    main()