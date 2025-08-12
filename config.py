#!/usr/bin/env python3
"""
Configuration file cho Voice Cloning Model
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR = BASE_DIR / "uploads"

# Voice samples configuration
VOICE_SAMPLES_DIR = DATA_DIR / "voice_samples"
METADATA_FILE = DATA_DIR / "metadata.csv"

# Model configuration
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
CUSTOM_MODEL_PATTERN = "*_config.json"

# Audio configuration
SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.flac', '.m4a'}
RECOMMENDED_SAMPLE_RATE = 22050
RECOMMENDED_DURATION_MIN = 2.0  # seconds
RECOMMENDED_DURATION_MAX = 15.0  # seconds

# Training configuration
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 1000
DEFAULT_LEARNING_RATE = 1e-4

# Web interface configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
FLASK_THREADED = True

# API configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

# Voice cloning parameters
DEFAULT_LANGUAGE = "en"  # Tiếng Anh (XTTS không hỗ trợ tiếng Việt)
DEFAULT_VOICE_ID_PREFIX = "voice_"

# Output configuration
AUDIO_OUTPUT_FORMAT = "wav"
AUDIO_OUTPUT_QUALITY = "high"  # high, medium, low

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "logs" / "voice_cloning.log"

# Performance configuration
USE_GPU = True  # Auto-detect GPU
MAX_CONCURRENT_REQUESTS = 5
REQUEST_TIMEOUT = 300  # seconds

# Security configuration
ENABLE_CORS = True
ALLOWED_ORIGINS = ["*"]  # Configure for production

# Validation configuration
MIN_VOICE_SAMPLES = 5
MAX_VOICE_SAMPLES = 100
MIN_AUDIO_DURATION = 1.0  # seconds
MAX_AUDIO_DURATION = 30.0  # seconds

# File naming configuration
AUDIO_FILENAME_PATTERN = "output_{voice_id}_{hash}.wav"
BATCH_FILENAME_PATTERN = "batch_{voice_id}_{index:03d}.wav"

# Error messages
ERROR_MESSAGES = {
    'voice_not_found': 'Voice ID "{voice_id}" không tìm thấy',
    'invalid_audio': 'File audio không hợp lệ',
    'file_too_large': 'File quá lớn (tối đa {max_size}MB)',
    'unsupported_format': 'Định dạng file không được hỗ trợ',
    'model_not_ready': 'Model chưa sẵn sàng',
    'training_failed': 'Training thất bại',
    'invalid_text': 'Text không hợp lệ'
}

# Success messages
SUCCESS_MESSAGES = {
    'voice_added': 'Voice sample đã được thêm thành công: {voice_id}',
    'voice_removed': 'Voice "{voice_id}" đã được xóa',
    'voice_cloned': 'Voice cloning hoàn tất thành công',
    'config_exported': 'Cấu hình đã được export',
    'training_completed': 'Training hoàn tất thành công'
}

# Warning messages
WARNING_MESSAGES = {
    'low_sample_rate': 'Sample rate thấp, khuyến nghị: {recommended}Hz',
    'short_duration': 'Độ dài audio ngắn, khuyến nghị: {min}-{max}s',
    'few_samples': 'Ít voice samples, khuyến nghị: ít nhất {min} samples'
}

# Create necessary directories
def create_directories():
    """Tạo các thư mục cần thiết"""
    directories = [
        DATA_DIR,
        VOICE_SAMPLES_DIR,
        MODELS_DIR,
        OUTPUT_DIR,
        UPLOADS_DIR,
        LOG_FILE.parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Configuration validation
def validate_config():
    """Kiểm tra tính hợp lệ của cấu hình"""
    errors = []
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        errors.append("Python 3.8+ required")
    
    # Check directories
    if not DATA_DIR.exists():
        errors.append(f"Data directory not found: {DATA_DIR}")
    
    # Check audio formats
    if not SUPPORTED_AUDIO_FORMATS:
        errors.append("No supported audio formats defined")
    
    # Check model configuration
    if not DEFAULT_MODEL:
        errors.append("Default model not specified")
    
    return errors

# Get configuration value
def get_config(key, default=None):
    """Lấy giá trị cấu hình"""
    return globals().get(key, default)

# Set configuration value
def set_config(key, value):
    """Đặt giá trị cấu hình"""
    globals()[key] = value

# Load environment variables
def load_env_config():
    """Load cấu hình từ environment variables"""
    env_mapping = {
        'FLASK_HOST': 'FLASK_HOST',
        'FLASK_PORT': 'FLASK_PORT',
        'FLASK_DEBUG': 'FLASK_DEBUG',
        'LOG_LEVEL': 'LOG_LEVEL',
        'USE_GPU': 'USE_GPU',
        'MAX_CONCURRENT_REQUESTS': 'MAX_CONCURRENT_REQUESTS'
    }
    
    for env_var, config_key in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            # Convert string values to appropriate types
            if config_key == 'FLASK_PORT':
                value = int(value)
            elif config_key == 'FLASK_DEBUG':
                value = value.lower() in ('true', '1', 'yes')
            elif config_key == 'USE_GPU':
                value = value.lower() in ('true', '1', 'yes')
            elif config_key == 'MAX_CONCURRENT_REQUESTS':
                value = int(value)
            
            set_config(config_key, value)

# Initialize configuration
if __name__ == "__main__":
    print("🔧 Voice Cloning Configuration")
    print("=" * 40)
    
    # Load environment variables
    load_env_config()
    
    # Create directories
    create_directories()
    
    # Validate configuration
    errors = validate_config()
    
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration is valid")
        print(f"📁 Data directory: {DATA_DIR}")
        print(f"🤖 Models directory: {MODELS_DIR}")
        print(f"🌐 Web interface: http://{FLASK_HOST}:{FLASK_PORT}")
        print(f"🎵 Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}") 