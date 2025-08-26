#!/usr/bin/env python3
"""
Voice Cloning Performance Benchmark
Test tốc độ voice cloning với các settings khác nhau
"""

import os
import time
import torch
from voice_cloner import VoiceCloner

def benchmark_voice_cloning():
    """Benchmark voice cloning performance"""
    print("🚀 Voice Cloning Performance Benchmark")
    print("=" * 50)
    
    # Test texts of different lengths
    test_texts = [
        ("Short", "Hello world"),
        ("Medium", "This is a medium length text for testing voice cloning performance."),
        ("Long", "This is a much longer text that will take more time to process. It contains multiple sentences and should give us a better understanding of how the system performs with different input lengths. The goal is to measure the time it takes for the voice cloning process to complete."),
    ]
    
    # Test devices
    devices = ["auto"]
    if torch.cuda.is_available():
        devices.append("cuda")
    
    for device in devices:
        print(f"\n🎯 Testing device: {device}")
        print("-" * 30)
        
        try:
            # Initialize voice cloner
            print(f"🔄 Initializing voice cloner on {device}...")
            start_time = time.time()
            cloner = VoiceCloner(device=device)
            init_time = time.time() - start_time
            print(f"✅ Initialization time: {init_time:.2f}s")
            
            # Add a test voice sample (you need to provide an audio file)
            test_audio = "test_voice.wav"  # Change this to your test audio file
            if os.path.exists(test_audio):
                cloner.add_voice_sample("test_voice", test_audio, "Test voice")
                print(f"✅ Added test voice sample")
                
                # Benchmark each text length
                for text_type, text in test_texts:
                    print(f"\n📝 Testing {text_type} text ({len(text)} chars):")
                    print(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
                    
                    # Warm up
                    print("🔥 Warming up...")
                    try:
                        output_path = f"benchmark_{text_type}_{device}.wav"
                        start_time = time.time()
                        cloner.clone_voice(text, "test_voice", output_path)
                        warmup_time = time.time() - start_time
                        print(f"🔥 Warmup time: {warmup_time:.2f}s")
                    except Exception as e:
                        print(f"❌ Warmup failed: {e}")
                        continue
                    
                    # Actual benchmark
                    print("⚡ Running benchmark...")
                    try:
                        start_time = time.time()
                        cloner.clone_voice(text, "test_voice", output_path)
                        benchmark_time = time.time() - start_time
                        
                        # Calculate performance metrics
                        chars_per_second = len(text) / benchmark_time
                        print(f"⚡ Benchmark time: {benchmark_time:.2f}s")
                        print(f"📊 Performance: {chars_per_second:.1f} chars/second")
                        
                        # Clean up
                        if os.path.exists(output_path):
                            os.remove(output_path)
                            
                    except Exception as e:
                        print(f"❌ Benchmark failed: {e}")
                        
            else:
                print(f"⚠️ Test audio file not found: {test_audio}")
                print("💡 Please provide a test audio file to run benchmarks")
                
        except Exception as e:
            print(f"❌ Failed to initialize voice cloner on {device}: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Benchmark completed!")
    
    # Performance recommendations
    print("\n💡 Performance Recommendations:")
    if torch.cuda.is_available():
        print("🚀 Use GPU (CUDA) for best performance")
        print("⚡ Enable half-precision (FP16) if memory allows")
    else:
        print("💻 CPU-only mode detected")
        print("🔧 Consider using a machine with GPU for better performance")
    
    print("📝 Shorter texts are faster to process")
    print("🎯 Use the optimized runner: python run_optimized.py")

if __name__ == "__main__":
    benchmark_voice_cloning() 