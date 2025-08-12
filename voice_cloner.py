import os
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class VoiceCloner:
    """
    Voice Cloning class sử dụng Coqui TTS XTTS model
    """
    
    # Supported languages for XTTS
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'pl': 'Polish',
        'tr': 'Turkish',
        'ru': 'Russian',
        'nl': 'Dutch',
        'cs': 'Czech',
        'ar': 'Arabic',
        'zh-cn': 'Chinese (Simplified)',
        'hu': 'Hungarian',
        'ko': 'Korean',
        'ja': 'Japanese',
        'hi': 'Hindi'
    }
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Khởi tạo Voice Cloner
        
        Args:
            model_path: Đường dẫn đến model đã train
            device: Device để chạy model (auto, cpu, cuda)
        """
        self.device = device
        self.model = None
        self.model_path = model_path
        self.voice_samples = {}
        self.voice_embeddings = {}
        
        # Khởi tạo model
        self._load_model()
    
    def _load_model(self):
        """Load XTTS model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load custom model
                self.model = TTS(model_path=self.model_path)
                print(f"✅ Loaded custom model from: {self.model_path}")
            else:
                # Load pre-trained XTTS model
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                print("✅ Loaded pre-trained XTTS v2 model")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def add_voice_sample(self, voice_id: str, audio_path: str, text: str = None):
        """
        Thêm voice sample để clone
        
        Args:
            voice_id: ID của voice
            audio_path: Đường dẫn đến file audio
            text: Text tương ứng với audio (optional)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Validate audio format
        if not audio_path.lower().endswith(('.wav', '.mp3', '.flac')):
            raise ValueError("Audio file must be .wav, .mp3, or .flac")
        
        self.voice_samples[voice_id] = {
            'audio_path': audio_path,
            'text': text
        }
        print(f"✅ Added voice sample: {voice_id} -> {audio_path}")
    
    def auto_detect_language(self, text: str) -> str:
        """
        Auto-detect language từ text (đơn giản dựa trên ký tự)
        
        Args:
            text: Text cần detect language
            
        Returns:
            Language code (default: 'en')
        """
        # Simple language detection based on characters
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh-cn'  # Chinese
        elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return 'ja'  # Japanese
        elif any('\uac00' <= char <= '\ud7af' for char in text):
            return 'ko'  # Korean
        elif any('\u0600' <= char <= '\u06ff' for char in text):
            return 'ar'  # Arabic
        elif any(char in 'áéíóúñü' for char in text.lower()):
            return 'es'  # Spanish
        elif any(char in 'àâäéèêëïîôùûüÿç' for char in text.lower()):
            return 'fr'  # French
        elif any(char in 'äöüß' for char in text.lower()):
            return 'de'  # German
        elif any(char in 'àèéìíîòóù' for char in text.lower()):
            return 'it'  # Italian
        elif any(char in 'àáâãçéêíóôõú' for char in text.lower()):
            return 'pt'  # Portuguese
        elif any(char in 'ąćęłńóśźż' for char in text.lower()):
            return 'pl'  # Polish
        elif any(char in 'çğıöşü' for char in text.lower()):
            return 'tr'  # Turkish
        elif any(char in 'аеёиоуыэюя' for char in text.lower()):
            return 'ru'  # Russian
        elif any(char in 'àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ' for char in text.lower()):
            return 'nl'  # Dutch
        elif any(char in 'áčďéěíňóřšťúůýž' for char in text.lower()):
            return 'cs'  # Czech
        elif any(char in 'áéíóöőúüű' for char in text.lower()):
            return 'hu'  # Hungarian
        elif any(char in 'अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळक्षज्ञ' for char in text):
            return 'hi'  # Hindi
        else:
            return 'en'  # Default to English
    
    def clone_voice(self, text: str, voice_id: str, output_path: str = None, language: str = None) -> str:
        """
        Clone voice và tạo audio từ text
        
        Args:
            text: Text cần chuyển thành giọng nói
            voice_id: ID của voice đã đăng ký
            output_path: Đường dẫn output (optional)
            language: Ngôn ngữ (optional, auto-detect nếu không chỉ định)
            
        Returns:
            Đường dẫn đến file audio đã tạo
        """
        if voice_id not in self.voice_samples:
            raise ValueError(f"Voice ID '{voice_id}' not found. Please add voice sample first.")
        
        # Auto-detect language if not specified
        if language is None:
            language = self.auto_detect_language(text)
            print(f"🌍 Auto-detected language: {language} ({self.SUPPORTED_LANGUAGES.get(language, 'Unknown')})")
        
        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            print(f"⚠️ Warning: Language '{language}' not supported, using 'en' instead")
            language = 'en'
        
        # Validate text length (XTTS limit: 250 characters)
        if len(text) > 250:
            print(f"⚠️ Warning: Text too long ({len(text)} chars), truncating to 250 characters")
            text = text[:250]
        
        if not output_path:
            output_path = f"output_{voice_id}_{hash(text) % 10000}.wav"
        
        try:
            # Sử dụng XTTS để clone voice
            audio_path = self.voice_samples[voice_id]['audio_path']
            
            # Tạo audio với voice cloning
            self.model.tts_to_file(
                text=text,
                speaker_wav=audio_path,
                language=language,
                file_path=output_path
            )
            
            print(f"✅ Voice cloned successfully: {output_path}")
            print(f"🌍 Language used: {language} ({self.SUPPORTED_LANGUAGES.get(language, 'Unknown')})")
            return output_path
            
        except Exception as e:
            print(f"❌ Error cloning voice: {e}")
            raise
    
    def clone_voice_with_effects(self, text: str, voice_id: str, output_path: str = None, 
                                language: str = None, speed: float = 1.0, pitch_shift: float = 0.0) -> str:
        """
        Clone voice với audio effects
        
        Args:
            text: Text cần chuyển thành giọng nói
            voice_id: ID của voice đã đăng ký
            output_path: Đường dẫn output (optional)
            language: Ngôn ngữ (optional)
            speed: Tốc độ phát (0.5 = chậm, 2.0 = nhanh)
            pitch_shift: Thay đổi pitch (-12 = thấp, +12 = cao)
            
        Returns:
            Đường dẫn đến file audio đã tạo
        """
        if voice_id not in self.voice_samples:
            raise ValueError(f"Voice ID '{voice_id}' not found. Please add voice sample first.")
        
        # Validate parameters
        if speed < 0.5 or speed > 2.0:
            print(f"⚠️ Warning: Speed {speed} out of range [0.5, 2.0], using 1.0")
            speed = 1.0
        
        if pitch_shift < -12 or pitch_shift > 12:
            print(f"⚠️ Warning: Pitch shift {pitch_shift} out of range [-12, 12], using 0.0")
            pitch_shift = 0.0
        
        # Auto-detect language if not specified
        if language is None:
            language = self.auto_detect_language(text)
            print(f"🌍 Auto-detected language: {language} ({self.SUPPORTED_LANGUAGES.get(language, 'Unknown')})")
        
        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            print(f"⚠️ Warning: Language '{language}' not supported, using 'en' instead")
            language = 'en'
        
        # Validate text length (XTTS limit: 250 characters)
        if len(text) > 250:
            print(f"⚠️ Warning: Text too long ({len(text)} chars), truncating to 250 characters")
            text = text[:250]
        
        if not output_path:
            output_path = f"output_{voice_id}_{hash(text) % 10000}.wav"
        
        try:
            # Sử dụng XTTS để clone voice
            audio_path = self.voice_samples[voice_id]['audio_path']
            
            # Tạo audio với voice cloning
            self.model.tts_to_file(
                text=text,
                speaker_wav=audio_path,
                language=language,
                file_path=output_path
            )
            
            # Apply audio effects if needed
            if speed != 1.0 or pitch_shift != 0.0:
                output_path = self._apply_audio_effects(output_path, speed, pitch_shift)
            
            print(f"✅ Voice cloned successfully: {output_path}")
            print(f"🌍 Language used: {language} ({self.SUPPORTED_LANGUAGES.get(language, 'Unknown')})")
            if speed != 1.0:
                print(f"⚡ Speed: {speed}x")
            if pitch_shift != 0.0:
                print(f"🎵 Pitch shift: {pitch_shift:+d} semitones")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error cloning voice: {e}")
            raise
    
    def _apply_audio_effects(self, audio_path: str, speed: float, pitch_shift: float) -> str:
        """
        Apply audio effects (speed, pitch) to audio file
        
        Args:
            audio_path: Đường dẫn file audio
            speed: Tốc độ phát
            pitch_shift: Thay đổi pitch (semitones)
            
        Returns:
            Đường dẫn file audio đã được xử lý
        """
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Apply speed change
            if speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed)
            
            # Apply pitch shift
            if pitch_shift != 0.0:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            
            # Save processed audio
            output_path = audio_path.replace('.wav', f'_effects_s{speed}_p{pitch_shift}.wav')
            sf.write(output_path, audio, sr)
            
            print(f"🎵 Applied audio effects: speed={speed}x, pitch={pitch_shift:+d} semitones")
            return output_path
            
        except Exception as e:
            print(f"⚠️ Warning: Could not apply audio effects: {e}")
            return audio_path
    
    def get_available_voices(self) -> List[str]:
        """Lấy danh sách voice đã đăng ký"""
        return list(self.voice_samples.keys())
    
    def remove_voice(self, voice_id: str):
        """Xóa voice sample"""
        if voice_id in self.voice_samples:
            del self.voice_samples[voice_id]
            print(f"✅ Removed voice: {voice_id}")
        else:
            print(f"⚠️ Voice ID '{voice_id}' not found")
    
    def batch_clone(self, texts: List[str], voice_id: str, output_dir: str = "output") -> List[str]:
        """
        Clone voice cho nhiều text cùng lúc
        
        Args:
            texts: Danh sách text cần clone
            voice_id: ID của voice
            output_dir: Thư mục output
            
        Returns:
            Danh sách đường dẫn file audio đã tạo
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"batch_{voice_id}_{i:03d}.wav")
            try:
                self.clone_voice(text, voice_id, output_path)
                output_paths.append(output_path)
            except Exception as e:
                print(f"❌ Error processing text {i}: {e}")
                continue
        
        print(f"✅ Batch processing completed: {len(output_paths)}/{len(texts)} files created")
        return output_paths
    
    def get_voice_info(self, voice_id: str) -> Dict:
        """Lấy thông tin chi tiết của voice"""
        if voice_id not in self.voice_samples:
            return None
        
        voice_info = self.voice_samples[voice_id].copy()
        
        # Thêm thông tin file audio
        audio_path = voice_info['audio_path']
        if os.path.exists(audio_path):
            try:
                audio_info = sf.info(audio_path)
                voice_info.update({
                    'duration': audio_info.duration,
                    'sample_rate': audio_info.samplerate,
                    'channels': audio_info.channels,
                    'file_size': os.path.getsize(audio_path)
                })
            except Exception:
                pass
        
        return voice_info
    
    def export_voice_config(self, output_path: str = "voice_config.json"):
        """Export cấu hình voice samples"""
        config = {
            'voices': self.voice_samples,
            'model_path': self.model_path,
            'device': self.device
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Voice configuration exported to: {output_path}")
    
    def import_voice_config(self, config_path: str):
        """Import cấu hình voice samples"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.voice_samples = config.get('voices', {})
        print(f"✅ Voice configuration imported: {len(self.voice_samples)} voices loaded")


# Utility functions
def validate_audio_file(audio_path: str) -> bool:
    """Kiểm tra file audio có hợp lệ không"""
    try:
        info = sf.info(audio_path)
        # Kiểm tra sample rate (nên là 22050Hz cho XTTS)
        if info.samplerate != 22050:
            print(f"⚠️ Warning: Audio sample rate is {info.samplerate}Hz, recommended: 22050Hz")
        
        # Kiểm tra duration (nên từ 3-10 giây)
        if info.duration < 2 or info.duration > 15:
            print(f"⚠️ Warning: Audio duration is {info.duration:.1f}s, recommended: 3-10s")
        
        return True
    except Exception as e:
        print(f"❌ Invalid audio file: {e}")
        return False


def convert_audio_format(input_path: str, output_path: str, target_sr: int = 22050):
    """Chuyển đổi format audio sang WAV với sample rate mong muốn"""
    try:
        import librosa
        
        # Load audio
        audio, sr = librosa.load(input_path, sr=target_sr)
        
        # Save as WAV
        sf.write(output_path, audio, target_sr)
        
        print(f"✅ Audio converted: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting audio: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    cloner = VoiceCloner()
    
    # Thêm voice sample
    cloner.add_voice_sample("my_voice", "path/to/audio.wav", "Xin chào")
    
    # Clone voice cơ bản
    output = cloner.clone_voice("Đây là giọng nói đã clone!", "my_voice")
    print(f"Basic output: {output}")
    
    # Clone voice với ngôn ngữ cụ thể
    output_en = cloner.clone_voice("Hello, this is cloned voice!", "my_voice", language="en")
    print(f"English output: {output_en}")
    
    # Clone voice với audio effects
    output_effects = cloner.clone_voice_with_effects(
        "Xin chào với hiệu ứng!", 
        "my_voice", 
        speed=1.2, 
        pitch_shift=2.0
    )
    print(f"Effects output: {output_effects}")
    
    # Demo auto-language detection
    texts = [
        "Hello world",  # English
        "Bonjour le monde",  # French
        "Hola mundo",  # Spanish
        "こんにちは世界",  # Japanese
        "안녕하세요 세계",  # Korean
        "你好世界",  # Chinese
    ]
    
    for text in texts:
        detected_lang = cloner.auto_detect_language(text)
        print(f"Text: {text} -> Language: {detected_lang}") 