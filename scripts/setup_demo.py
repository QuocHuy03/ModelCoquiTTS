#!/usr/bin/env python3
"""
Setup Demo Script
Tạo dữ liệu mẫu để test voice cloning
"""

import os
import csv
import shutil
from pathlib import Path


def create_demo_structure():
    """Tạo cấu trúc thư mục demo"""
    print("🔧 Tạo cấu trúc thư mục demo...")
    
    # Tạo thư mục data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    voice_samples_dir = data_dir / "voice_samples"
    voice_samples_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục uploads
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    print("✅ Cấu trúc thư mục đã được tạo")


def create_sample_metadata():
    """Tạo file metadata mẫu"""
    print("📝 Tạo file metadata mẫu...")
    
    metadata_file = Path("data/metadata.csv")
    
    # Sample metadata cho voice cloning
    sample_data = [
        ["sample_001.wav", "Xin chào, tôi là giọng nói mẫu số một"],
        ["sample_002.wav", "Đây là giọng nói mẫu thứ hai"],
        ["sample_003.wav", "Chào mừng bạn đến với hệ thống voice cloning"],
        ["sample_004.wav", "Hôm nay là một ngày đẹp trời"],
        ["sample_005.wav", "Công nghệ AI đang phát triển rất nhanh"],
        ["sample_006.wav", "Voice cloning là một ứng dụng thú vị"],
        ["sample_007.wav", "Chúng ta có thể clone giọng nói của bất kỳ ai"],
        ["sample_008.wav", "Chất lượng audio rất quan trọng"],
        ["sample_009.wav", "Sample rate nên là 22050Hz"],
        ["sample_010.wav", "Độ dài audio nên từ 3-10 giây"]
    ]
    
    with open(metadata_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['filename', 'text'])
        writer.writerows(sample_data)
    
    print(f"✅ File metadata mẫu đã được tạo: {metadata_file}")
    print(f"📊 Số lượng samples: {len(sample_data)}")


def create_sample_audio_files():
    """Tạo file audio mẫu (placeholder)"""
    print("🎵 Tạo file audio mẫu...")
    
    voice_samples_dir = Path("data/voice_samples")
    
    # Tạo file placeholder cho audio samples
    for i in range(1, 11):
        filename = f"sample_{i:03d}.wav"
        file_path = voice_samples_dir / filename
        
        # Tạo file placeholder (1KB)
        with open(file_path, 'wb') as f:
            f.write(b'RIFF' + b'\x00' * 1020)  # WAV header placeholder
        
        print(f"   ✅ {filename}")
    
    print("⚠️  Lưu ý: Đây chỉ là file placeholder. Bạn cần thay thế bằng audio thật!")


def create_demo_script():
    """Tạo script demo để test voice cloning"""
    print("📜 Tạo script demo...")
    
    demo_script = Path("scripts/demo_voice_cloning.py")
    
    demo_code = '''#!/usr/bin/env python3
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
                print("\\n🧪 Testing voice cloning...")
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
        print("\\n💡 Gợi ý:")
        print("   1. Chạy: pip install -r requirements.txt")
        print("   2. Thêm audio files vào data/voice_samples/")
        print("   3. Chạy lại script này")


if __name__ == "__main__":
    main()
'''
    
    with open(demo_script, 'w', encoding='utf-8') as f:
        f.write(demo_code)
    
    print(f"✅ Script demo đã được tạo: {demo_script}")


def create_requirements_check():
    """Tạo script kiểm tra requirements"""
    print("🔍 Tạo script kiểm tra requirements...")
    
    check_script = Path("scripts/check_requirements.py")
    
    check_code = '''#!/usr/bin/env python3
"""
Check Requirements Script
Kiểm tra các dependencies cần thiết
"""

import sys
import subprocess
import importlib


def check_python_version():
    """Kiểm tra phiên bản Python"""
    print("🐍 Kiểm tra phiên bản Python...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Cần Python >= 3.8")
        return False


def check_package(package_name, import_name=None):
    """Kiểm tra package đã được cài đặt"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError:
        print(f"❌ {package_name} - Chưa cài đặt")
        return False


def check_requirements():
    """Kiểm tra tất cả requirements"""
    print("📦 Kiểm tra các packages...")
    
    packages = [
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("TTS", "TTS"),
        ("numpy", "numpy"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("flask", "flask"),
        ("flask-cors", "flask_cors")
    ]
    
    all_ok = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            all_ok = False
    
    return all_ok


def main():
    print("🔍 Kiểm tra Requirements cho Voice Cloning")
    print("=" * 50)
    
    python_ok = check_python_version()
    packages_ok = check_requirements()
    
    print("\\n📊 Kết quả kiểm tra:")
    if python_ok and packages_ok:
        print("🎉 Tất cả requirements đã được đáp ứng!")
        print("   Bạn có thể chạy voice cloning model")
    else:
        print("⚠️  Một số requirements chưa được đáp ứng")
        print("\\n💡 Để cài đặt, chạy:")
        print("   pip install -r requirements.txt")
    
    print("\\n🚀 Để chạy web interface:")
    print("   python app.py")
    
    print("\\n🎵 Để test voice cloning:")
    print("   python scripts/demo_voice_cloning.py")


if __name__ == "__main__":
    main()
'''
    
    with open(check_script, 'w', encoding='utf-8') as f:
        f.write(check_code)
    
    print(f"✅ Script kiểm tra requirements đã được tạo: {check_script}")


def main():
    """Main function"""
    print("🎵 Setup Demo cho Voice Cloning Model")
    print("=" * 50)
    
    # Tạo cấu trúc thư mục
    create_demo_structure()
    
    # Tạo metadata mẫu
    create_sample_metadata()
    
    # Tạo audio samples mẫu
    create_sample_audio_files()
    
    # Tạo script demo
    create_demo_script()
    
    # Tạo script kiểm tra requirements
    create_requirements_check()
    
    print("\\n🎉 Setup demo hoàn tất!")
    print("\\n📋 Hướng dẫn sử dụng:")
    print("1. Cài đặt dependencies: pip install -r requirements.txt")
    print("2. Kiểm tra requirements: python scripts/check_requirements.py")
    print("3. Thay thế audio samples mẫu bằng audio thật")
    print("4. Chạy demo: python scripts/demo_voice_cloning.py")
    print("5. Chạy web interface: python app.py")
    print("\\n⚠️  Lưu ý: Audio samples mẫu chỉ là placeholder!")
    print("   Bạn cần thay thế bằng audio thật để test voice cloning")


if __name__ == "__main__":
    main() 