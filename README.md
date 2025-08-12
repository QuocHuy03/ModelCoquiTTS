# Voice Cloning Model với Coqui TTS

Dự án này sử dụng Coqui TTS để tạo model clone voice từ audio samples.

## Tính năng

- 🎵 **Voice Cloning**: Clone giọng nói từ audio samples
- 🎤 **Text-to-Speech**: Chuyển text thành giọng nói đã clone
- 🌐 **Web Interface**: Giao diện web để test model
- 📊 **Training Pipeline**: Quy trình huấn luyện model
- 🔧 **Easy Setup**: Cài đặt và sử dụng đơn giản

## Cài đặt

1. **Clone repository:**
```bash
git clone <your-repo-url>
cd CloneVoiceModel
```

2. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

3. **Chuẩn bị dữ liệu:**
- Tạo thư mục `data/voice_samples/` 
- Thêm audio files (.wav) của giọng nói muốn clone
- Tạo file `data/metadata.csv` với format: `filename|text`

## Sử dụng

### 1. Huấn luyện model
```bash
python train_voice_clone.py --voice_samples data/voice_samples/ --metadata data/metadata.csv
```

### 2. Chạy web interface
```bash
python app.py
```

### 3. Sử dụng API
```bash
curl -X POST http://localhost:5000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Xin chào, đây là giọng nói đã clone!", "voice_id": "cloned_voice"}'
```

## Cấu trúc dự án

```
CloneVoiceModel/
├── data/                   # Dữ liệu training
├── models/                 # Model đã train
├── scripts/                # Scripts tiện ích
├── web/                    # Web interface
├── train_voice_clone.py    # Script training chính
├── app.py                  # Flask web app
├── voice_cloner.py         # Core voice cloning logic
└── requirements.txt        # Dependencies
```

## Lưu ý

- Cần ít nhất 10-20 audio samples chất lượng cao để clone voice tốt
- Audio nên có độ dài 3-10 giây, rõ ràng, không nhiễu
- Training có thể mất vài giờ tùy thuộc vào hardware

## Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. Phiên bản Python (>=3.8)
2. CUDA support cho GPU training
3. Định dạng audio files (.wav, 22050Hz) 