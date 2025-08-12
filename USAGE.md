# Hướng dẫn sử dụng Voice Cloning Model

## 🎯 Tổng quan

Dự án này sử dụng **Coqui TTS** để tạo model clone voice từ audio samples. Bạn có thể clone giọng nói của bất kỳ ai chỉ cần cung cấp một số audio samples chất lượng cao.

## 🚀 Cài đặt nhanh

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup demo
```bash
python scripts/setup_demo.py
```

### 3. Kiểm tra requirements
```bash
python scripts/check_requirements.py
```

### 4. Chạy web interface
```bash
python app.py
```

Truy cập: http://localhost:5000

## 📁 Cấu trúc dự án

```
CloneVoiceModel/
├── 📁 data/                   # Dữ liệu training
│   ├── 📁 voice_samples/      # Audio samples
│   └── 📄 metadata.csv        # Metadata cho samples
├── 📁 models/                 # Model đã train
├── 📁 scripts/                # Scripts tiện ích
│   ├── 📄 setup_demo.py       # Setup demo
│   ├── 📄 check_requirements.py # Kiểm tra requirements
│   └── 📄 demo_voice_cloning.py # Demo voice cloning
├── 📁 web/                    # Web interface
│   └── 📁 templates/          # HTML templates
├── 📁 output/                 # Audio output
├── 📁 uploads/                # Uploaded files
├── 📄 voice_cloner.py         # Core voice cloning logic
├── 📄 train_voice_clone.py    # Training script
├── 📄 app.py                  # Flask web app
├── 📄 config.py               # Configuration
└── 📄 requirements.txt        # Dependencies
```

## 🎵 Chuẩn bị dữ liệu

### Audio Samples

**Yêu cầu:**
- **Định dạng**: WAV, MP3, FLAC, M4A (khuyến nghị: WAV)
- **Sample rate**: 22050Hz (khuyến nghị)
- **Độ dài**: 3-10 giây (tối thiểu: 2s, tối đa: 15s)
- **Chất lượng**: Rõ ràng, không nhiễu, không echo
- **Số lượng**: Ít nhất 10-20 samples

**Tips:**
- Ghi âm trong môi trường yên tĩnh
- Sử dụng microphone chất lượng tốt
- Đảm bảo giọng nói rõ ràng, tự nhiên
- Tránh background noise

### Metadata

Tạo file `data/metadata.csv` với format:
```csv
filename|text
sample_001.wav|Xin chào, tôi là giọng nói mẫu
sample_002.wav|Đây là câu thứ hai để training
sample_003.wav|Câu thứ ba với nội dung khác
```

**Lưu ý:**
- Sử dụng dấu `|` làm delimiter
- Text phải khớp với nội dung audio
- Không sử dụng dấu `|` trong text

## 🏋️ Training Model

### 1. Training cơ bản
```bash
python train_voice_clone.py \
  --voice_samples data/voice_samples/ \
  --metadata data/metadata.csv
```

### 2. Training với tùy chọn
```bash
python train_voice_clone.py \
  --voice_samples data/voice_samples/ \
  --metadata data/metadata.csv \
  --output_dir models/ \
  --model_name my_voice_model \
  --device cuda \
  --epochs 2000 \
  --learning_rate 5e-5
```

### 3. Chỉ validate dữ liệu
```bash
python train_voice_clone.py \
  --voice_samples data/voice_samples/ \
  --metadata data/metadata.csv \
  --validate_only
```

## 🎭 Sử dụng Voice Cloning

### 1. Sử dụng Python API

```python
from voice_cloner import VoiceCloner

# Khởi tạo
cloner = VoiceCloner()

# Thêm voice sample
cloner.add_voice_sample("my_voice", "path/to/audio.wav", "Mô tả voice")

# Clone voice
output_path = cloner.clone_voice(
    "Đây là text cần clone thành giọng nói!", 
    "my_voice"
)

print(f"Audio đã tạo: {output_path}")
```

### 2. Batch processing

```python
# Clone nhiều text cùng lúc
texts = [
    "Xin chào các bạn",
    "Hôm nay là ngày đẹp trời",
    "Cảm ơn bạn đã lắng nghe"
]

output_paths = cloner.batch_clone(texts, "my_voice", "output/")
```

### 3. Quản lý voices

```python
# Lấy danh sách voices
voices = cloner.get_available_voices()

# Lấy thông tin voice
voice_info = cloner.get_voice_info("my_voice")

# Xóa voice
cloner.remove_voice("my_voice")

# Export/Import config
cloner.export_voice_config("my_config.json")
cloner.import_voice_config("my_config.json")
```

## 🌐 Web Interface

### 1. Khởi động
```bash
python app.py
```

### 2. Sử dụng

**Thêm Voice Sample:**
1. Mở http://localhost:5000
2. Chọn tab "Thêm Voice Sample"
3. Nhập Voice ID (optional)
4. Nhập mô tả
5. Chọn file audio
6. Click "Upload Voice Sample"

**Clone Voice:**
1. Chọn tab "Clone Voice"
2. Chọn voice từ dropdown
3. Nhập text cần clone
4. Click "Clone Voice"
5. Nghe và tải xuống audio

**Quản lý Voices:**
- Xem danh sách voices
- Xóa voice không cần thiết
- Export cấu hình

## 🔧 Cấu hình

### Environment Variables

```bash
# Web interface
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
export FLASK_DEBUG=true

# Performance
export USE_GPU=true
export MAX_CONCURRENT_REQUESTS=5

# Logging
export LOG_LEVEL=INFO
```

### File config.py

Chỉnh sửa `config.py` để thay đổi:
- Đường dẫn thư mục
- Tham số audio
- Cấu hình training
- Giới hạn file size
- Thông báo lỗi

## 📊 Monitoring & Debugging

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Logs
```bash
tail -f logs/voice_cloning.log
```

### Performance
- Monitor GPU usage: `nvidia-smi`
- Monitor memory: `htop`
- Monitor disk space: `df -h`

## 🚨 Troubleshooting

### Lỗi thường gặp

**1. "Model not found"**
```bash
# Kiểm tra TTS installation
python -c "import TTS; print(TTS.__version__)"

# Reinstall TTS
pip uninstall TTS
pip install TTS
```

**2. "CUDA out of memory"**
```bash
# Giảm batch size
python train_voice_clone.py --batch_size 2

# Sử dụng CPU
python train_voice_clone.py --device cpu
```

**3. "Audio file not found"**
```bash
# Kiểm tra đường dẫn
ls -la data/voice_samples/

# Kiểm tra permissions
chmod 644 data/voice_samples/*.wav
```

**4. "Flask app not starting"**
```bash
# Kiểm tra port
netstat -tulpn | grep 5000

# Thay đổi port
export FLASK_PORT=5001
python app.py
```

### Debug Mode

```python
# Trong voice_cloner.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Trong app.py
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📈 Performance Optimization

### GPU Training
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Sử dụng GPU
export CUDA_VISIBLE_DEVICES=0
python train_voice_clone.py --device cuda
```

### Memory Optimization
```bash
# Giảm batch size
--batch_size 2

# Sử dụng gradient accumulation
--gradient_accumulation_steps 4
```

### Audio Processing
```bash
# Convert audio trước training
python -c "
from voice_cloner import convert_audio_format
convert_audio_format('input.mp3', 'output.wav', 22050)
"
```

## 🔒 Security Considerations

### Production Deployment
```bash
# Disable debug mode
export FLASK_DEBUG=false

# Restrict CORS
export ALLOWED_ORIGINS="https://yourdomain.com"

# Enable HTTPS
# Sử dụng reverse proxy (nginx) + SSL
```

### File Upload Security
```bash
# Giới hạn file size
export MAX_FILE_SIZE=10485760  # 10MB

# Validate file types
# Chỉ cho phép audio files
```

## 📚 Advanced Usage

### Custom Model Training
```python
# Fine-tune pre-trained model
from TTS.trainer import Trainer
from TTS.config import load_config

config = load_config("path/to/config.json")
trainer = Trainer(config)
trainer.fit()
```

### Voice Cloning API
```python
# REST API
import requests

response = requests.post("http://localhost:5000/api/synthesize", json={
    "text": "Hello world",
    "voice_id": "my_voice"
})

audio_url = response.json()["audio_url"]
```

### Integration với ứng dụng khác
```python
# Webhook integration
@app.route('/webhook/synthesize', methods=['POST'])
def webhook_synthesize():
    data = request.get_json()
    # Process webhook data
    # Return audio URL
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repo-url>
cd CloneVoiceModel

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
isort .
```

### Code Style
- Sử dụng Python 3.8+
- Follow PEP 8
- Type hints cho functions
- Docstrings cho classes/methods
- Error handling đầy đủ

## 📄 License

Dự án này sử dụng MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## 🆘 Hỗ trợ

### Documentation
- [Coqui TTS Documentation](https://tts.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)

### Community
- [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- [Discord Community](https://discord.gg/5qXaJ6Q)

### Issues
- Tạo issue trên GitHub
- Mô tả chi tiết vấn đề
- Đính kèm logs và error messages
- Cung cấp steps to reproduce

---

**🎉 Chúc bạn thành công với Voice Cloning Model!** 