#!/usr/bin/env python3
"""
Demo Voice Cloning Script
Script mẫu để test voice cloning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_cloner import VoiceCloner


def main():
    print("🎵 Demo Voice Cloning")
    print("=" * 40)
    
    try:
        # Khởi tạo voice cloner
        print("🚀 Đang khởi tạo voice cloner...")
        cloner = VoiceCloner()
        
        # Thêm voice samples (nếu có)
        voice_samples_dir = "data/voice_samples"
        if os.path.exists(voice_samples_dir):
            print(f"📁 Tìm thấy thư mục voice samples: {voice_samples_dir}")
            
            # Liệt kê các file audio
            audio_files = [f for f in os.listdir(voice_samples_dir) 
                          if f.lower().endswith(('.wav', '.mp3', '.flac'))]
            
            if audio_files:
                print(f"🎵 Tìm thấy {len(audio_files)} file audio:")
                for audio_file in audio_files[:3]:  # Chỉ hiển thị 3 file đầu
                    print(f"   - {audio_file}")
                
                # Thêm voice sample đầu tiên
                first_audio = os.path.join(voice_samples_dir, audio_files[0])
                cloner.add_voice_sample("demo_voice", first_audio, "Voice sample demo")
                
                # Test voice cloning
                print("\n🧪 Testing voice cloning...")
                test_text = "Xin chào, đây là test voice cloning!"
                
                output_path = cloner.clone_voice(test_text, "demo_voice")
                print(f"✅ Voice cloning thành công: {output_path}")
                
            else:
                print("⚠️  Không tìm thấy file audio nào trong thư mục voice_samples")
                print("   Hãy thêm các file audio (.wav, .mp3, .flac) vào thư mục này")
        else:
            print(f"❌ Không tìm thấy thư mục voice samples: {voice_samples_dir}")
            print("   Hãy chạy script setup_demo.py trước")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("\n💡 Gợi ý:")
        print("   1. Chạy: pip install -r requirements.txt")
        print("   2. Thêm audio files vào data/voice_samples/")
        print("   3. Chạy lại script này")


if __name__ == "__main__":
    main()
