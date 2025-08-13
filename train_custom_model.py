#!/usr/bin/env python3
"""
Custom Voice Cloning Model Training Script
Sử dụng Coqui TTS để train model voice cloning riêng
"""

import os
import json
import argparse
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import torch


class VoiceModelTrainer:
    def __init__(self, config_path: str = None):
        """
        Khởi tạo Voice Model Trainer
        
        Args:
            config_path: Đường dẫn đến config file (optional)
        """
        self.config_path = config_path
        self.model = None
        self.config = None
        
    def create_training_config(self, output_path: str = "custom_xtts_config.json"):
        """
        Tạo config file cho training
        
        Args:
            output_path: Đường dẫn output config
        """
        # Base XTTS config
        base_config = XttsConfig()
        
        # Customize config cho training
        custom_config = {
            "model": "xtts",
            "run_name": "custom_voice_cloning",
            "run_description": "Custom Vietnamese Voice Cloning Model",
            
            # Training parameters
            "epochs": 1000,
            "batch_size": 4,
            "eval_batch_size": 4,
            "num_loader_workers": 4,
            "num_eval_loader_workers": 4,
            
            # Learning rate
            "lr": 1e-4,
            "lr_decay": 0.999875,
            "lr_decay_step": 1,
            
            # Optimizer
            "optimizer": "AdamW",
            "optimizer_params": {
                "betas": [0.9, 0.998],
                "eps": 1e-9,
                "weight_decay": 1e-2
            },
            
            # Scheduler
            "lr_scheduler": "ExponentialLR",
            "lr_scheduler_params": {
                "gamma": 0.999875
            },
            
            # Model specific
            "use_speaker_encoder_as_loss": True,
            "speaker_encoder_model_path": None,
            "speaker_encoder_config_path": None,
            
            # Audio processing
            "audio": {
                "sample_rate": 22050,
                "hop_length": 256,
                "win_length": 1024,
                "fft_size": 1024,
                "mel_channels": 80,
                "mel_fmin": 0.0,
                "mel_fmax": 8000.0
            },
            
            # Text processing
            "text": {
                "cleaner_name": "multilingual_cleaners",
                "add_blank": True,
                "phoneme_cache_path": None
            },
            
            # Training data
            "datasets": [
                {
                    "name": "custom_voice_dataset",
                    "path": "data/",
                    "meta_file_train": "data/metadata.csv",
                    "meta_file_val": "data/metadata_val.csv",
                    "language": "vi"
                }
            ],
            
            # Checkpointing
            "save_step": 1000,
            "save_n_checkpoints": 5,
            "save_best_after": 10000,
            "save_checkpoints": True,
            
            # Logging
            "print_step": 25,
            "print_eval": True,
            "mixed_precision": True,
            "output_path": "output/",
            "datasets": "data/"
        }
        
        # Merge configs
        for key, value in custom_config.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
            else:
                base_config.__dict__[key] = value
        
        # Save config
        base_config.save_json(output_path)
        print(f"✅ Training config created: {output_path}")
        
        return output_path
    
    def prepare_training_data(self, audio_dir: str, metadata_file: str):
        """
        Chuẩn bị dữ liệu training
        
        Args:
            audio_dir: Thư mục chứa audio files
            metadata_file: File metadata
        """
        print("🔧 Preparing training data...")
        
        # Tạo thư mục data nếu chưa có
        os.makedirs("data", exist_ok=True)
        
        # Copy audio files
        if os.path.exists(audio_dir):
            import shutil
            for root, dirs, files in os.walk(audio_dir):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac')):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join("data", file)
                        shutil.copy2(src_path, dst_path)
                        print(f"📁 Copied: {file}")
        
        # Tạo metadata file nếu chưa có
        if not os.path.exists(metadata_file):
            self._create_sample_metadata()
        
        # Tạo validation metadata
        self._create_validation_metadata(metadata_file)
        
        print("✅ Training data prepared successfully!")
    
    def _create_sample_metadata(self):
        """Tạo file metadata mẫu"""
        metadata_content = """filename|text
sample1.wav|Xin chào, tôi là người nói tiếng Việt
sample2.wav|Hôm nay trời đẹp quá
sample3.wav|Cảm ơn bạn đã lắng nghe
sample4.wav|Chúc bạn một ngày tốt lành
sample5.wav|Tôi rất vui được gặp bạn"""
        
        with open("data/metadata.csv", "w", encoding="utf-8") as f:
            f.write(metadata_content)
        
        print("📝 Created sample metadata.csv")
    
    def _create_validation_metadata(self, train_metadata: str):
        """Tạo file validation metadata"""
        if not os.path.exists(train_metadata):
            return
        
        # Đọc training metadata
        with open(train_metadata, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Tạo validation metadata (20% của training data)
        header = lines[0]
        data_lines = lines[1:]
        
        # Chia data
        split_index = len(data_lines) // 5  # 20%
        val_lines = data_lines[:split_index]
        
        # Lưu validation metadata
        with open("data/metadata_val.csv", "w", encoding="utf-8") as f:
            f.write(header)
            f.writelines(val_lines)
        
        print(f"✅ Created validation metadata: {len(val_lines)} samples")
    
    def start_training(self, config_path: str, resume_from: str = None):
        """
        Bắt đầu training
        
        Args:
            config_path: Đường dẫn config file
            resume_from: Checkpoint để resume (optional)
        """
        print("🚀 Starting voice cloning model training...")
        
        try:
            # Load config
            self.config = XttsConfig()
            self.config.load_json(config_path)
            
            # Initialize model
            self.model = Xtts.init_from_config(self.config)
            
            # Move to GPU nếu có
            if torch.cuda.is_available():
                self.model.cuda()
                print("✅ Using CUDA for training")
            else:
                print("⚠️ CUDA not available, using CPU")
            
            # Start training
            self.model.fit(
                config_path=config_path,
                resume_from=resume_from
            )
            
            print("🎉 Training completed successfully!")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def export_model(self, checkpoint_path: str, output_dir: str = "models"):
        """
        Export model đã train
        
        Args:
            checkpoint_path: Đường dẫn checkpoint
            output_dir: Thư mục output
        """
        print("📦 Exporting trained model...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Export model
            model_path = os.path.join(output_dir, "custom_xtts_model.pth")
            torch.save(checkpoint, model_path)
            
            # Export config
            config_path = os.path.join(output_dir, "custom_xtts_config.json")
            if self.config:
                self.config.save_json(config_path)
            
            print(f"✅ Model exported to: {output_dir}")
            print(f"📁 Model file: {model_path}")
            print(f"📁 Config file: {config_path}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Custom Voice Cloning Model")
    parser.add_argument("--mode", choices=["prepare", "train", "export"], 
                       default="prepare", help="Training mode")
    parser.add_argument("--audio_dir", type=str, default="audio_samples",
                       help="Directory containing audio files")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv",
                       help="Path to metadata file")
    parser.add_argument("--config", type=str, default="custom_xtts_config.json",
                       help="Path to training config")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for export")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = VoiceModelTrainer()
    
    if args.mode == "prepare":
        print("🔧 Mode: Prepare Training Data")
        trainer.prepare_training_data(args.audio_dir, args.metadata)
        trainer.create_training_config(args.config)
        
    elif args.mode == "train":
        print("🚀 Mode: Start Training")
        if not os.path.exists(args.config):
            print("❌ Config file not found. Run 'prepare' mode first.")
            return
        trainer.start_training(args.config, args.resume)
        
    elif args.mode == "export":
        print("📦 Mode: Export Model")
        if not args.checkpoint:
            print("❌ Checkpoint path required for export mode")
            return
        trainer.export_model(args.checkpoint)
    
    print("✅ Operation completed successfully!")


if __name__ == "__main__":
    main() 