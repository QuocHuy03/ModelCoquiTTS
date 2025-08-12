#!/usr/bin/env python3
"""
Vietnamese Voice Cloning Demo
Hướng dẫn sử dụng tiếng Việt với Coqui TTS XTTS
"""

from voice_cloner import VoiceCloner
import os

def demo_vietnamese_voice_cloning():
    """
    Demo cách sử dụng tiếng Việt với voice cloning
    """
    print("🇻🇳 VIETNAMESE VOICE CLONING DEMO")
    print("=" * 50)
    
    # Khởi tạo voice cloner
    print("🚀 Initializing Voice Cloner...")
    cloner = VoiceCloner()
    
    print("\n📋 SCENARIO: Clone voice tiếng Việt")
    print("=" * 50)
    
    # Demo 1: Cách SAI - Text tiếng Việt
    print("\n❌ CÁCH SAI - Text tiếng Việt:")
    print("Input: 'Xin chào, tôi là người Việt Nam'")
    print("Result: XTTS không thể đọc được, sẽ bị lỗi hoặc phát âm sai")
    
    # Demo 2: Cách ĐÚNG - Voice sample tiếng Việt + Text tiếng Anh
    print("\n✅ CÁCH ĐÚNG - Voice sample tiếng Việt + Text tiếng Anh:")
    print("1. Upload voice sample tiếng Việt (giọng nói tiếng Việt)")
    print("2. Nhập text tiếng Anh: 'Hello, I am Vietnamese'")
    print("3. Kết quả: Giọng tiếng Anh với accent tiếng Việt")
    
    print("\n💡 CÁCH HOẠT ĐỘNG:")
    print("• Voice sample tiếng Việt → Giữ accent, tone, đặc điểm giọng nói")
    print("• Text tiếng Anh → XTTS có thể đọc được tự nhiên")
    print("• Kết hợp → Giọng tiếng Anh với accent tiếng Việt")
    
    print("\n🎯 VÍ DỤ THỰC TẾ:")
    print("=" * 50)
    
    examples = [
        {
            "voice_sample": "Giọng nam Hà Nội",
            "text_english": "Hello, welcome to Vietnam!",
            "result": "Giọng tiếng Anh với accent Hà Nội"
        },
        {
            "voice_sample": "Giọng nữ Sài Gòn", 
            "text_english": "I love Vietnamese food and culture",
            "result": "Giọng tiếng Anh với accent Sài Gòn"
        },
        {
            "voice_sample": "Giọng trẻ em miền Tây",
            "text_english": "Vietnam is a beautiful country",
            "result": "Giọng tiếng Anh với accent miền Tây"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Voice Sample: {example['voice_sample']}")
        print(f"   Text: '{example['text_english']}'")
        print(f"   Kết quả: {example['result']}")
    
    print("\n🔧 HƯỚNG DẪN THỰC HÀNH:")
    print("=" * 50)
    print("1. 📱 Upload voice sample tiếng Việt (WAV, MP3, FLAC)")
    print("2. 🌍 Chọn language: 'Vietnamese' hoặc để auto-detect")
    print("3. 📝 Nhập text tiếng Anh (không phải tiếng Việt)")
    print("4. 🎵 Clone voice thành công!")
    
    print("\n⚠️  LƯU Ý QUAN TRỌNG:")
    print("• XTTS model gốc KHÔNG hỗ trợ tiếng Việt")
    print("• Chỉ có thể clone accent/giọng nói tiếng Việt")
    print("• Text phải là tiếng Anh để XTTS đọc được")
    
    print("\n🚀 ĐỂ HỖ TRỢ TIẾNG VIỆT HOÀN TOÀN:")
    print("• Fine-tune XTTS model với dataset tiếng Việt lớn")
    print("• Cần GPU mạnh và thời gian training dài")
    print("• Hoặc sử dụng model tiếng Việt khác (như VITS, FastSpeech2)")
    
    print("\n🎉 KẾT LUẬN:")
    print("Voice cloning tiếng Việt hoạt động tốt với:")
    print("✅ Voice sample tiếng Việt + Text tiếng Anh")
    print("❌ Voice sample tiếng Việt + Text tiếng Việt (không hoạt động)")

if __name__ == "__main__":
    demo_vietnamese_voice_cloning() 