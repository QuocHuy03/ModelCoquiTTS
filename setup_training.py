#!/usr/bin/env python3
"""
Setup Script for Voice Cloning Training
Chuẩn bị môi trường và cấu trúc dữ liệu cho training
"""

import os
import shutil
from pathlib import Path


def setup_training_environment():
    """Thiết lập môi trường training"""
    print("🔧 Setting up training environment...")
    
    # Tạo cấu trúc thư mục
    directories = [
        "data",
        "data/audio",
        "data/audio/speaker1",
        "data/audio/speaker2",
        "models",
        "output",
        "checkpoints",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    # Tạo file metadata mẫu
    create_sample_metadata()
    
    # Tạo config mẫu
    create_sample_config()
    
    # Tạo requirements.txt
    create_requirements()
    
    print("✅ Training environment setup completed!")


def create_sample_metadata():
    """Tạo file metadata mẫu"""
    metadata_content = """filename|text
audio/speaker1/sample1.wav|Xin chào, tôi là người nói tiếng Việt
audio/speaker1/sample2.wav|Hôm nay trời đẹp quá
audio/speaker1/sample3.wav|Cảm ơn bạn đã lắng nghe
audio/speaker1/sample4.wav|Chúc bạn một ngày tốt lành
audio/speaker1/sample5.wav|Tôi rất vui được gặp bạn
audio/speaker2/sample1.wav|Hello, I am an English speaker
audio/speaker2/sample2.wav|The weather is beautiful today
audio/speaker2/sample3.wav|Thank you for listening
audio/speaker2/sample4.wav|Have a great day
audio/speaker2/sample5.wav|I am happy to meet you"""
    
    with open("data/metadata.csv", "w", encoding="utf-8") as f:
        f.write(metadata_content)
    
    print("📝 Created sample metadata.csv")


def create_sample_config():
    """Tạo file config mẫu"""
    config_content = """{
    "model": "xtts",
    "run_name": "custom_voice_cloning",
    "run_description": "Custom Vietnamese Voice Cloning Model",
    "epochs": 1000,
    "batch_size": 4,
    "lr": 1e-4,
    "audio": {
        "sample_rate": 22050,
        "hop_length": 256,
        "win_length": 1024
    },
    "text": {
        "cleaner_name": "multilingual_cleaners"
    }
}"""
    
    with open("data/sample_config.json", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("⚙️ Created sample config.json")


def create_requirements():
    """Tạo file requirements.txt"""
    requirements_content = """# Voice Cloning Training Requirements
coqui-tts[all]>=0.14.0
torch>=1.13.0
torchaudio>=0.13.0
numpy>=1.21.0
scipy>=1.9.0
librosa>=0.9.2
soundfile>=0.12.1
matplotlib>=3.5.0
tqdm>=4.64.0
tensorboard>=2.10.0
wandb>=0.13.0
"""
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    
    print("📦 Created requirements.txt")


def create_training_guide():
    """Tạo hướng dẫn training"""
    guide_content = """# 🚀 Voice Cloning Training Guide

## 📋 Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 50GB+ free disk space

## 🔧 Installation
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python -c "import TTS; print('✅ TTS installed successfully')"
```

## 📁 Data Preparation
1. **Audio Files**: Place your audio files in `data/audio/speaker1/`, `data/audio/speaker2/`, etc.
2. **Metadata**: Update `data/metadata.csv` with your audio files and corresponding text
3. **Format**: WAV, MP3, FLAC (22050Hz recommended)

## 🎯 Training Steps

### Step 1: Prepare Data
```bash
python train_custom_model.py --mode prepare --audio_dir your_audio_folder
```

### Step 2: Start Training
```bash
python train_custom_model.py --mode train --config custom_xtts_config.json
```

### Step 3: Export Model
```bash
python train_custom_model.py --mode export --checkpoint path/to/checkpoint
```

## 📊 Training Parameters
- **Epochs**: 1000 (adjust based on data size)
- **Batch Size**: 4 (reduce if out of memory)
- **Learning Rate**: 1e-4 (adjust if training is unstable)
- **Save Checkpoints**: Every 1000 steps

## 🎵 Audio Requirements
- **Duration**: 3-10 seconds per sample
- **Quality**: Clear, minimal noise
- **Quantity**: 50-100 samples per speaker minimum
- **Language**: Vietnamese + English (or your target languages)

## 🔍 Monitoring Training
- **Tensorboard**: `tensorboard --logdir output/`
- **Checkpoints**: Saved in `checkpoints/` folder
- **Logs**: Training logs in `logs/` folder

## ⚠️ Common Issues
1. **Out of Memory**: Reduce batch_size
2. **Training Slow**: Check GPU usage, reduce num_workers
3. **Poor Quality**: Increase training data, adjust learning rate

## 🎉 After Training
1. Model saved in `models/` folder
2. Update `voice_cloner.py` to use custom model
3. Test with new voice samples

## 📞 Support
- Check Coqui TTS documentation
- Monitor training logs for errors
- Adjust parameters based on your data
"""
    
    with open("TRAINING_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide_content)
    
    print("📚 Created TRAINING_GUIDE.md")


def main():
    """Main function"""
    print("🎵 Voice Cloning Training Setup")
    print("=" * 40)
    
    setup_training_environment()
    create_training_guide()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add your audio files to data/audio/ folders")
    print("2. Update data/metadata.csv with your data")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Start training: python train_custom_model.py --mode prepare")
    print("\n📚 Read TRAINING_GUIDE.md for detailed instructions")


if __name__ == "__main__":
    main() 