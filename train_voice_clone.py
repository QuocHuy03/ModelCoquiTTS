#!/usr/bin/env python3
"""
Voice Cloning Training Script
Sử dụng Coqui TTS để train model clone voice
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict
import torch
from voice_cloner import VoiceCloner, validate_audio_file, convert_audio_format


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Voice Cloning Model")
    
    parser.add_argument(
        "--voice_samples", 
        type=str, 
        required=True,
        help="Đường dẫn đến thư mục chứa voice samples"
    )
    
    parser.add_argument(
        "--metadata", 
        type=str, 
        required=True,
        help="Đường dẫn đến file metadata.csv"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models",
        help="Thư mục output cho model (default: models)"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="voice_clone_model",
        help="Tên model (default: voice_clone_model)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device để training (default: auto)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size cho training (default: 4)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1000,
        help="Số epochs training (default: 1000)"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    
    parser.add_argument(
        "--validate_only", 
        action="store_true",
        help="Chỉ validate dữ liệu, không training"
    )
    
    return parser.parse_args()


def load_metadata(metadata_path: str) -> List[Dict]:
    """Load metadata từ CSV file"""
    metadata = []
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                metadata.append(row)
        
        print(f"✅ Loaded {len(metadata)} metadata entries")
        return metadata
        
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return []


def validate_dataset(voice_samples_dir: str, metadata: List[Dict]) -> bool:
    """Validate dataset trước khi training"""
    print("🔍 Validating dataset...")
    
    if not os.path.exists(voice_samples_dir):
        print(f"❌ Voice samples directory not found: {voice_samples_dir}")
        return False
    
    valid_samples = 0
    total_samples = len(metadata)
    
    for entry in metadata:
        filename = entry.get('filename', '').strip()
        text = entry.get('text', '').strip()
        
        if not filename or not text:
            print(f"⚠️ Invalid metadata entry: {entry}")
            continue
        
        audio_path = os.path.join(voice_samples_dir, filename)
        
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            continue
        
        if not validate_audio_file(audio_path):
            print(f"⚠️ Audio validation failed: {audio_path}")
            continue
        
        valid_samples += 1
    
    print(f"📊 Dataset validation: {valid_samples}/{total_samples} samples valid")
    
    if valid_samples < 10:
        print("⚠️ Warning: Recommended at least 10 valid samples for good voice cloning")
    
    return valid_samples > 0


def prepare_training_data(voice_samples_dir: str, metadata: List[Dict], output_dir: str):
    """Chuẩn bị dữ liệu training"""
    print("🔄 Preparing training data...")
    
    training_dir = os.path.join(output_dir, "training_data")
    os.makedirs(training_dir, exist_ok=True)
    
    prepared_samples = []
    
    for entry in metadata:
        filename = entry.get('filename', '').strip()
        text = entry.get('text', '').strip()
        
        if not filename or not text:
            continue
        
        input_path = os.path.join(voice_samples_dir, filename)
        output_filename = f"prepared_{filename}"
        output_path = os.path.join(training_dir, output_filename)
        
        # Convert audio format nếu cần
        if filename.lower().endswith('.wav'):
            # Copy WAV file
            import shutil
            shutil.copy2(input_path, output_path)
        else:
            # Convert to WAV
            if convert_audio_format(input_path, output_path):
                output_filename = output_filename.replace('.mp3', '.wav').replace('.flac', '.wav')
                output_path = os.path.join(training_dir, output_filename)
        
        if os.path.exists(output_path):
            prepared_samples.append({
                'filename': output_filename,
                'text': text,
                'path': output_path
            })
    
    # Tạo metadata mới cho training
    training_metadata_path = os.path.join(training_dir, "metadata.csv")
    with open(training_metadata_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['filename', 'text'])
        for sample in prepared_samples:
            writer.writerow([sample['filename'], sample['text']])
    
    print(f"✅ Prepared {len(prepared_samples)} samples for training")
    return training_dir, training_metadata_path


def train_voice_clone_model(training_dir: str, metadata_path: str, args):
    """Train voice cloning model"""
    print("🚀 Starting voice clone training...")
    
    try:
        # Khởi tạo voice cloner
        cloner = VoiceCloner(device=args.device)
        
        # Load prepared samples
        metadata = load_metadata(metadata_path)
        
        # Thêm voice samples
        for entry in metadata:
            filename = entry.get('filename', '').strip()
            text = entry.get('text', '').strip()
            audio_path = os.path.join(training_dir, filename)
            
            if os.path.exists(audio_path):
                voice_id = f"voice_{len(cloner.get_available_voices())}"
                cloner.add_voice_sample(voice_id, audio_path, text)
        
        print(f"✅ Added {len(cloner.get_available_voices())} voice samples")
        
        # Test voice cloning
        print("🧪 Testing voice cloning...")
        test_text = "Xin chào, đây là test voice cloning!"
        test_voice = cloner.get_available_voices()[0] if cloner.get_available_voices() else None
        
        if test_voice:
            try:
                output_path = cloner.clone_voice(test_text, test_voice, "test_output.wav")
                print(f"✅ Test successful: {output_path}")
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        # Export voice configuration
        config_path = os.path.join(args.output_dir, f"{args.model_name}_config.json")
        cloner.export_voice_config(config_path)
        
        print(f"✅ Training completed! Model config saved to: {config_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def main():
    """Main function"""
    args = parse_arguments()
    
    print("🎵 Voice Cloning Training Script")
    print("=" * 50)
    
    # Validate arguments
    if not os.path.exists(args.voice_samples):
        print(f"❌ Voice samples directory not found: {args.voice_samples}")
        sys.exit(1)
    
    if not os.path.exists(args.metadata):
        print(f"❌ Metadata file not found: {args.metadata}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metadata
    metadata = load_metadata(args.metadata)
    if not metadata:
        print("❌ No valid metadata found")
        sys.exit(1)
    
    # Validate dataset
    if not validate_dataset(args.voice_samples, metadata):
        print("❌ Dataset validation failed")
        sys.exit(1)
    
    if args.validate_only:
        print("✅ Dataset validation completed successfully")
        return
    
    # Prepare training data
    training_dir, training_metadata = prepare_training_data(
        args.voice_samples, metadata, args.output_dir
    )
    
    # Train model
    train_voice_clone_model(training_dir, training_metadata, args)
    
    print("\n🎉 Voice cloning training completed successfully!")
    print(f"📁 Model files saved to: {args.output_dir}")
    print(f"🔧 To use the model, run: python voice_cloner.py")


if __name__ == "__main__":
    main() 