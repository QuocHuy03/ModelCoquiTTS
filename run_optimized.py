#!/usr/bin/env python3
"""
Optimized Voice Cloning Runner
Chạy với performance settings tốt nhất
"""

import os
import sys
import torch
from app import app, init_voice_cloner

def check_system_requirements():
    """Kiểm tra system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PyTorch
    print(f"🔥 PyTorch: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"🚀 CUDA: {cuda_version}")
        print(f"🎯 GPU Devices: {device_count}")
        print(f"💾 GPU: {device_name}")
        print(f"💾 GPU Memory: {memory_gb:.1f} GB")
        
        # Set optimal CUDA settings
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking CUDA
        os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8
        torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark
        
        return "cuda"
    else:
        print("💻 CUDA not available - using CPU")
        return "cpu"

def set_environment_optimizations():
    """Set environment variables for optimal performance"""
    print("⚡ Setting performance optimizations...")
    
    # Coqui TTS optimizations
    os.environ['COQUI_TOS_AGREED'] = '1'  # Auto-accept license
    
    # PyTorch optimizations
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())  # Use all CPU cores
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
    
    # Memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Disable warnings for cleaner output
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    print("✅ Environment optimizations applied")

def main():
    """Main function"""
    print("🎵 Starting Optimized Voice Cloning Web Interface...")
    print("=" * 60)
    
    # Check system
    device = check_system_requirements()
    
    # Set optimizations
    set_environment_optimizations()
    
    # Set device for voice cloner
    os.environ['VOICE_CLONER_DEVICE'] = device
    
    print("=" * 60)
    
    # Initialize voice cloner with device
    if init_voice_cloner():
        print("✅ Voice cloner initialized successfully")
    else:
        print("❌ Failed to initialize voice cloner")
        print("⚠️ Some features may not work properly")
    
    # Get port from environment
    port = int(os.getenv('PORT', '7000'))
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"🚀 Starting server on http://{host}:{port}")
    print(f"🎯 Device: {device}")
    print(f"⚡ Performance mode: Enabled")
    
    # Run the app
    app.run(
        host=host,
        port=port,
        debug=False,  # Disable debug for production
        threaded=True,
        use_reloader=False  # Disable reloader for production
    )

if __name__ == '__main__':
    main() 