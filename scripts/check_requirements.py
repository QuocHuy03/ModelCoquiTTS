#!/usr/bin/env python3
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
    
    print("\n📊 Kết quả kiểm tra:")
    if python_ok and packages_ok:
        print("🎉 Tất cả requirements đã được đáp ứng!")
        print("   Bạn có thể chạy voice cloning model")
    else:
        print("⚠️  Một số requirements chưa được đáp ứng")
        print("\n💡 Để cài đặt, chạy:")
        print("   pip install -r requirements.txt")
    
    print("\n🚀 Để chạy web interface:")
    print("   python app.py")
    
    print("\n🎵 Để test voice cloning:")
    print("   python scripts/demo_voice_cloning.py")


if __name__ == "__main__":
    main()
