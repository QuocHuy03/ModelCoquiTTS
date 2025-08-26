#!/usr/bin/env python3
"""
Performance Configuration for Voice Cloning
Các profile tối ưu hóa khác nhau cho voice cloning
"""

import os
import torch

class PerformanceConfig:
    """Performance configuration profiles"""
    
    @staticmethod
    def ultra_fast():
        """Ultra-fast mode - prioritize speed over quality"""
        config = {
            'device': 'auto',
            'use_half_precision': True,
            'enable_cache': True,
            'fast_inference': True,
            'model_variant': 'xtts_v1.1',  # Faster than v2
            'inference_settings': {
                'speed': 1.5,
                'enable_text_normalization': False,
                'enable_phonemizer': False,
                'enable_emotion_control': False,
                'enable_voice_cloning': True,
            },
            'text_processing': {
                'max_length': 250,  # Shorter text = faster
                'skip_language_detection': True,
            }
        }
        return config
    
    @staticmethod
    def balanced():
        """Balanced mode - good balance of speed and quality"""
        config = {
            'device': 'auto',
            'use_half_precision': True,
            'enable_cache': True,
            'fast_inference': True,
            'model_variant': 'xtts_v2',
            'inference_settings': {
                'speed': 1.2,
                'enable_text_normalization': True,
                'enable_phonemizer': False,
                'enable_emotion_control': False,
                'enable_voice_cloning': True,
            },
            'text_processing': {
                'max_length': 500,
                'skip_language_detection': False,
            }
        }
        return config
    
    @staticmethod
    def high_quality():
        """High quality mode - prioritize quality over speed"""
        config = {
            'device': 'auto',
            'use_half_precision': False,
            'enable_cache': True,
            'fast_inference': False,
            'model_variant': 'xtts_v2',
            'inference_settings': {
                'speed': 1.0,
                'enable_text_normalization': True,
                'enable_phonemizer': True,
                'enable_emotion_control': True,
                'enable_voice_cloning': True,
            },
            'text_processing': {
                'max_length': 1000,
                'skip_language_detection': False,
            }
        }
        return config
    
    @staticmethod
    def gpu_optimized():
        """GPU-optimized mode for CUDA devices"""
        if not torch.cuda.is_available():
            return PerformanceConfig.balanced()
        
        config = {
            'device': 'cuda',
            'use_half_precision': True,
            'enable_cache': True,
            'fast_inference': True,
            'model_variant': 'xtts_v2',
            'inference_settings': {
                'speed': 1.3,
                'enable_text_normalization': True,
                'enable_phonemizer': False,
                'enable_emotion_control': False,
                'enable_voice_cloning': True,
            },
            'text_processing': {
                'max_length': 750,
                'skip_language_detection': False,
            },
            'cuda_settings': {
                'memory_fraction': 0.8,
                'enable_tf32': True,
                'enable_cudnn_benchmark': True,
            }
        }
        return config
    
    @staticmethod
    def get_config(profile='auto'):
        """Get configuration based on profile"""
        if profile == 'auto':
            # Auto-detect best profile
            if torch.cuda.is_available():
                return PerformanceConfig.gpu_optimized()
            else:
                return PerformanceConfig.balanced()
        elif profile == 'ultra_fast':
            return PerformanceConfig.ultra_fast()
        elif profile == 'balanced':
            return PerformanceConfig.balanced()
        elif profile == 'high_quality':
            return PerformanceConfig.high_quality()
        elif profile == 'gpu_optimized':
            return PerformanceConfig.gpu_optimized()
        else:
            return PerformanceConfig.balanced()
    
    @staticmethod
    def apply_environment_optimizations(config):
        """Apply environment optimizations based on config"""
        print("⚡ Applying performance optimizations...")
        
        # Coqui TTS
        os.environ['COQUI_TOS_AGREED'] = '1'
        
        # PyTorch optimizations
        if config.get('device') == 'cuda':
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
            
            cuda_settings = config.get('cuda_settings', {})
            if cuda_settings.get('enable_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            if cuda_settings.get('enable_cudnn_benchmark'):
                torch.backends.cudnn.benchmark = True
        
        # CPU optimizations
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
        
        # Memory optimizations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        print("✅ Performance optimizations applied")

# Usage example:
if __name__ == "__main__":
    print("🎯 Performance Configuration Profiles")
    print("=" * 40)
    
    profiles = ['ultra_fast', 'balanced', 'high_quality', 'gpu_optimized', 'auto']
    
    for profile in profiles:
        config = PerformanceConfig.get_config(profile)
        print(f"\n📊 Profile: {profile}")
        print(f"   Device: {config['device']}")
        print(f"   Half Precision: {config['use_half_precision']}")
        print(f"   Fast Inference: {config['fast_inference']}")
        print(f"   Model: {config['model_variant']}")
        print(f"   Max Text Length: {config['text_processing']['max_length']}")
    
    print(f"\n🚀 Auto-detected profile:")
    auto_config = PerformanceConfig.get_config('auto')
    print(f"   Device: {auto_config['device']}")
    print(f"   Model: {auto_config['model_variant']}") 