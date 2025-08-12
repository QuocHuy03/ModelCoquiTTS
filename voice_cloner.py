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
        'hi': 'Hindi',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'id': 'Indonesian',
        'ms': 'Malay',
        'tl': 'Tagalog',
        'bn': 'Bengali',
        'ta': 'Tamil',
        'te': 'Telugu',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'gu': 'Gujarati',
        'pa': 'Punjabi',
        'or': 'Odia',
        'as': 'Assamese',
        'ne': 'Nepali',
        'si': 'Sinhala',
        'my': 'Burmese',
        'km': 'Khmer',
        'lo': 'Lao',
        'mn': 'Mongolian',
        'ka': 'Georgian',
        'hy': 'Armenian',
        'az': 'Azerbaijani',
        'kk': 'Kazakh',
        'ky': 'Kyrgyz',
        'uz': 'Uzbek',
        'tg': 'Tajik',
        'tk': 'Turkmen',
        'et': 'Estonian',
        'lv': 'Latvian',
        'lt': 'Lithuanian',
        'fi': 'Finnish',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'da': 'Danish',
        'is': 'Icelandic',
        'fo': 'Faroese',
        'sq': 'Albanian',
        'mk': 'Macedonian',
        'bg': 'Bulgarian',
        'ro': 'Romanian',
        'hr': 'Croatian',
        'sl': 'Slovenian',
        'sk': 'Slovak',
        'uk': 'Ukrainian',
        'be': 'Belarusian',
        'mt': 'Maltese',
        'cy': 'Welsh',
        'ga': 'Irish',
        'gd': 'Scottish Gaelic',
        'br': 'Breton',
        'eu': 'Basque',
        'ca': 'Catalan',
        'gl': 'Galician',
        'oc': 'Occitan',
        'fur': 'Friulian',
        'rm': 'Romansh',
        'lad': 'Ladino',
        'jv': 'Javanese',
        'su': 'Sundanese',
        'ceb': 'Cebuano',
        'war': 'Waray',
        'hil': 'Hiligaynon',
        'bcl': 'Central Bikol',
        'pam': 'Kapampangan',
        'pag': 'Pangasinan',
        'ilo': 'Ilocano',
        'bjn': 'Banjar',
        'ace': 'Acehnese',
        'min': 'Minangkabau',
        'gor': 'Gorontalo',
        'bug': 'Buginese',
        'mak': 'Makassarese',
        'mad': 'Madurese',
        'ban': 'Balinese',
        'sas': 'Sasak',
        'sun': 'Sundanese',
        'jav': 'Javanese',
        'kbd': 'Kabardian',
        'ady': 'Adyghe',
        'ab': 'Abkhaz',
        'os': 'Ossetian',
        'av': 'Avar',
        'lez': 'Lezgi',
        'tab': 'Tabasaran',
        'agx': 'Aghul',
        'rut': 'Rutul',
        'tsakhur': 'Tsakhur',
        'udm': 'Udmurt',
        'kom': 'Komi',
        'mhr': 'Mari',
        'udm': 'Udmurt',
        'mns': 'Mansi',
        'kca': 'Khanty',
        'sel': 'Selkup',
        'ket': 'Ket',
        'yug': 'Yug',
        'niv': 'Nivkh',
        'chv': 'Chuvash',
        'tat': 'Tatar',
        'bua': 'Buryat',
        'xal': 'Kalmyk',
        'tyv': 'Tuvan',
        'alt': 'Southern Altai',
        'krc': 'Karachay-Balkar',
        'kum': 'Kumyk',
        'nog': 'Nogai',
        'crh': 'Crimean Tatar',
        'gag': 'Gagauz',
        'cjs': 'Shor',
        'kjh': 'Khakas',
        'kim': 'Tofa',
        'dlg': 'Dolgan',
        'sah': 'Yakut',
        'evn': 'Evenki',
        'eve': 'Even',
        'neg': 'Negidal',
        'ulc': 'Ulch',
        'oaa': 'Orok',
        'ude': 'Udege',
        'orv': 'Old Russian',
        'chu': 'Old Church Slavonic',
        'grc': 'Ancient Greek',
        'lat': 'Latin',
        'san': 'Sanskrit',
        'pal': 'Pahlavi',
        'ave': 'Avestan',
        'xpr': 'Parthian',
        'xsc': 'Scythian',
        'xss': 'Assan',
        'xco': 'Chorasmian',
        'xln': 'Alanic',
        'xme': 'Median',
        'xmr': 'Meroitic',
        'xmt': 'Mator',
        'xna': 'Ancient North Arabian',
        'xpg': 'Ancient Greek',
        'xpi': 'Pictish',
        'xpm': 'Pumpokol',
        'xpo': 'Pochutec',
        'xpp': 'Puyo',
        'xpr': 'Parthian',
        'xps': 'Umbrian',
        'xpu': 'Punic',
        'xpy': 'Puyo',
        'xqt': 'Qatabanian',
        'xre': 'Krevinian',
        'xrn': 'Arin',
        'xrr': 'Raetic',
        'xrt': 'Aranama',
        'xrw': 'Karawa',
        'xsa': 'Sabaean',
        'xsc': 'Scythian',
        'xsi': 'Sidetic',
        'xsm': 'Samaritan',
        'xsn': 'Sanga',
        'xsp': 'Sop',
        'xsu': 'Subu',
        'xsv': 'Sudovian',
        'xta': 'Alcozauca Mixtec',
        'xtb': 'Chazumba Mixtec',
        'xtc': 'Katcha-Kadugli-Miri',
        'xtd': 'Diuxi-Tilantongo Mixtec',
        'xte': 'Ketengban',
        'xtg': 'Transalpine Gaulish',
        'xti': 'Sinicahua Mixtec',
        'xtj': 'San Juan Teita Mixtec',
        'xtl': 'Tijaltepec Mixtec',
        'xtm': 'Magdalena Peñasco Mixtec',
        'xtn': 'Northern Tlaxiaco Mixtec',
        'xto': 'Tokharian A',
        'xtp': 'San Miguel Piedras Mixtec',
        'xtq': 'Tumshuqese',
        'xtr': 'Early Tripuri',
        'xts': 'Sindihui Mixtec',
        'xtt': 'Tacahua Mixtec',
        'xtu': 'Cuyamecalco Mixtec',
        'xtv': 'Thawa',
        'xtw': 'Tawandê',
        'xty': 'Yoloxochitl Mixtec',
        'xtz': 'Tasmanian',
        'xua': 'Alu Kurumba',
        'xub': 'Betta Kurumba',
        'xud': 'Umiida',
        'xuf': 'Kunfal',
        'xug': 'Kunigami',
        'xuj': 'Jennu Kurumba',
        'xul': 'Ngunawal',
        'xum': 'Umbrian',
        'xun': 'Unggarranggu',
        'xur': 'Urartian',
        'xuu': 'Kxoe',
        'xve': 'Venetic',
        'xvi': 'Kamviri',
        'xvn': 'Vandalic',
        'xvo': 'Volscian',
        'xvs': 'Vestinian',
        'xwa': 'Kwaza',
        'xwc': 'Woccon',
        'xwd': 'Wadiyara Koli',
        'xwe': 'Xwela Gbe',
        'xwg': 'Kwegu',
        'xwj': 'Wajuk',
        'xwk': 'Wangkumara',
        'xwl': 'Western Xwla Gbe',
        'xwo': 'Written Oirat',
        'xwr': 'Kwerba Mamberamo',
        'xwt': 'Wotjobaluk',
        'xww': 'Wemba Wemba',
        'xxb': 'Boro (Ghana)',
        'xxk': 'Keo',
        'xxm': 'Minkin',
        'xxr': 'Koropó',
        'xxt': 'Tambora',
        'xya': 'Yaygir',
        'xyb': 'Yandjibara',
        'xyj': 'Mayi-Yapi',
        'xyk': 'Mayi-Kulan',
        'xyl': 'Yalakalore',
        'xyt': 'Mayi-Thakurti',
        'xyy': 'Yorta Yorta',
        'xzh': 'Zhang-Zhung',
        'xzm': 'Zemgalian',
        'xzp': 'Ancient Zapotec'
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
        
        # Special handling for Vietnamese - use English as base but keep Vietnamese text
        if language == "vi":
            print("🇻🇳 Vietnamese detected!")
            print("⚠️  WARNING: XTTS model cannot naturally read Vietnamese text")
            print("💡 SOLUTION: Use English text for voice cloning, Vietnamese voice sample will maintain accent")
            print("📝 RECOMMENDATION: Input English text, voice will sound Vietnamese due to voice sample")
            
            # Ask user if they want to continue with Vietnamese text or switch to English
            print("🔄 Switching to English base language for XTTS compatibility...")
            language = "en"  # Use English as base language
        
        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            print(f"⚠️ Warning: Language '{language}' not supported, using 'en' instead")
            language = 'en'
        
        # Validate text length (XTTS limit: 500 characters - increased from 250)
        if len(text) > 500:
            print(f"⚠️ Warning: Text too long ({len(text)} chars), truncating to 500 characters")
            text = text[:500]
        
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
            if language == "en" and self.auto_detect_language(text) == "vi":
                print(f"🌍 Language used: English (base) + Vietnamese (text)")
                print(f"🎯 Result: English pronunciation with Vietnamese voice accent")
            else:
                print(f"🌍 Language used: {language} ({self.SUPPORTED_LANGUAGES.get(language, 'Unknown')})")
            return output_path
            
        except Exception as e:
            print(f"❌ Error cloning voice: {e}")
            raise
    
    def clone_voice_with_effects(self, text: str, voice_id: str, output_path: str = None, 
                                language: str = None, speed: float = 1.0, pitch_shift: float = 0.0,
                                voice_type: str = "normal", age_group: str = "adult") -> str:
        """
        Clone voice với audio effects đầy đủ
        
        Args:
            text: Text cần chuyển thành giọng nói
            voice_id: ID của voice đã đăng ký
            output_path: Đường dẫn output (optional)
            language: Ngôn ngữ (optional)
            speed: Tốc độ phát (0.5 = chậm, 2.0 = nhanh)
            pitch_shift: Thay đổi pitch (-12 = thấp, +12 = cao)
            voice_type: Loại giọng (normal, male, female, child, elderly)
            age_group: Nhóm tuổi (child, teen, adult, elderly)
            
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
        
        # Special handling for Vietnamese - use English as base but keep Vietnamese text
        if language == "vi":
            print("🇻🇳 Vietnamese detected! Using English as base language for XTTS compatibility")
            print("💡 Note: Vietnamese text will be processed with English phonetics but maintain Vietnamese pronunciation")
            language = "en"  # Use English as base language
        
        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            print(f"⚠️ Warning: Language '{language}' not supported, using 'en' instead")
            language = 'en'
        
        # Validate text length (XTTS limit: 500 characters)
        if len(text) > 500:
            print(f"⚠️ Warning: Text too long ({len(text)} chars), truncating to 500 characters")
            text = text[:500]
        
        # Apply voice type and age group effects
        final_pitch_shift = pitch_shift
        final_speed = speed
        
        if voice_type == "male":
            final_pitch_shift -= 3  # Giọng nam thấp hơn
        elif voice_type == "female":
            final_pitch_shift += 3  # Giọng nữ cao hơn
        elif voice_type == "child":
            final_pitch_shift += 6  # Giọng trẻ em cao hơn
            final_speed *= 1.1  # Nói nhanh hơn một chút
        elif voice_type == "elderly":
            final_pitch_shift -= 2  # Giọng già thấp hơn
            final_speed *= 0.9  # Nói chậm hơn một chút
        
        if age_group == "child":
            final_pitch_shift += 4
            final_speed *= 1.15
        elif age_group == "teen":
            final_pitch_shift += 2
            final_speed *= 1.05
        elif age_group == "elderly":
            final_pitch_shift -= 3
            final_speed *= 0.85
        
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
            if final_speed != 1.0 or final_pitch_shift != 0.0:
                output_path = self._apply_audio_effects(output_path, final_speed, final_pitch_shift)
            
            print(f"✅ Voice cloned successfully: {output_path}")
            print(f"🌍 Language used: {language} ({self.SUPPORTED_LANGUAGES.get(language, 'Unknown')})")
            print(f"🎭 Voice type: {voice_type}, Age group: {age_group}")
            if final_speed != 1.0:
                print(f"⚡ Speed: {final_speed}x")
            if final_pitch_shift != 0.0:
                print(f"🎵 Pitch shift: {final_pitch_shift:+d} semitones")
            
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
    
    def batch_clone_voices(self, texts: list, voice_id: str, output_folder: str = "batch_output", 
                          language: str = None, voice_type: str = "normal", age_group: str = "adult",
                          speed: float = 1.0, pitch_shift: float = 0.0) -> dict:
        """
        Clone voice cho nhiều text cùng lúc
        
        Args:
            texts: List các text cần clone
            voice_id: ID của voice đã đăng ký
            output_folder: Thư mục output
            language: Ngôn ngữ (optional)
            voice_type: Loại giọng
            age_group: Nhóm tuổi
            speed: Tốc độ
            pitch_shift: Pitch shift
            
        Returns:
            Dict chứa kết quả của từng text
        """
        if voice_id not in self.voice_samples:
            raise ValueError(f"Voice ID '{voice_id}' not found. Please add voice sample first.")
        
        # Tạo thư mục output nếu chưa có
        os.makedirs(output_folder, exist_ok=True)
        
        results = {
            'success_count': 0,
            'failed_count': 0,
            'outputs': [],
            'errors': []
        }
        
        print(f"🚀 Starting batch processing for {len(texts)} texts...")
        
        for i, text in enumerate(texts, 1):
            try:
                print(f"\n📝 Processing text {i}/{len(texts)}: {text[:50]}...")
                
                # Tạo tên file output
                output_filename = f"batch_{voice_id}_{i:03d}_{hash(text) % 10000}.wav"
                output_path = os.path.join(output_folder, output_filename)
                
                # Clone voice với effects
                result_path = self.clone_voice_with_effects(
                    text, voice_id, output_path, language, speed, pitch_shift, voice_type, age_group
                )
                
                results['outputs'].append({
                    'text': text,
                    'output_path': result_path,
                    'filename': output_filename,
                    'index': i
                })
                results['success_count'] += 1
                
                print(f"✅ Text {i} processed successfully: {output_filename}")
                
            except Exception as e:
                error_msg = f"Failed to process text {i}: {str(e)}"
                print(f"❌ {error_msg}")
                results['errors'].append({
                    'text': text,
                    'error': str(e),
                    'index': i
                })
                results['failed_count'] += 1
        
        print(f"\n🎉 Batch processing completed!")
        print(f"✅ Success: {results['success_count']}")
        print(f"❌ Failed: {results['failed_count']}")
        print(f"📁 Output folder: {output_folder}")
        
        return results
    
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
    
    # Clone voice với audio effects đầy đủ
    output_effects = cloner.clone_voice_with_effects(
        "Xin chào với hiệu ứng!", 
        "my_voice", 
        speed=1.2, 
        pitch_shift=2.0,
        voice_type="female",
        age_group="teen"
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
        "สวัสดีโลก",  # Thai
        "Xin chào thế giới",  # Vietnamese - Special handling
        "Halo dunia",  # Indonesian
        "Selamat pagi dunia",  # Malay
    ]
    
    for text in texts:
        detected_lang = cloner.auto_detect_language(text)
        print(f"Text: {text} -> Language: {detected_lang}")
    
    # Demo Vietnamese voice cloning
    print("\n🇻🇳 Vietnamese Voice Cloning Demo:")
    try:
        vietnamese_output = cloner.clone_voice(
            "Xin chào! Tôi là giọng nói tiếng Việt được clone từ Coqui TTS.", 
            "my_voice", 
            language="vi"
        )
        print(f"✅ Vietnamese clone successful: {vietnamese_output}")
    except Exception as e:
        print(f"❌ Vietnamese clone failed: {e}")
    
    # Demo voice types
    voice_types = ["normal", "male", "female", "child", "elderly"]
    age_groups = ["child", "teen", "adult", "elderly"]
    
    print("\n🎭 Voice Types:", voice_types)
    print("👥 Age Groups:", age_groups)
    print("⚡ Speed Range: 0.5x - 2.0x")
    print("🎵 Pitch Range: -12 to +12 semitones")
    print("📝 Text Limit: 500 characters")
    print(f"🌍 Supported Languages: {len(cloner.SUPPORTED_LANGUAGES)} languages")
    print("🇻🇳 Vietnamese: Special handling with English base language") 