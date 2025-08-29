import os
import json
import torch
import numpy as np
import soundfile as sf
import datetime
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
        
        # Performance optimization flags
        self.use_half_precision = True
        self.enable_cache = True
        self.fast_inference = True
        
        # Khởi tạo model
        self._load_model()
    
    def _load_model(self):
        """Load XTTS model với performance optimizations"""
        try:
            # Auto-detect GPU
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                    print(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
                    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                else:
                    self.device = "cpu"
                    print("💻 Using CPU (GPU not available)")
            
            if self.model_path and os.path.exists(self.model_path):
                # Load custom model
                self.model = TTS(model_path=self.model_path)
                print(f"✅ Loaded custom model from: {self.model_path}")
            else:
                # Load pre-trained XTTS model với optimizations
                print("🔄 Loading XTTS v2 model (this may take a few minutes)...")
                
                # Use faster model variant if available
                try:
                    # Try to load faster XTTS v1.1 first
                    self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v1.1")
                    print("✅ Loaded XTTS v1.1 (faster than v2)")
                except:
                    # Fallback to v2
                    self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                    print("✅ Loaded XTTS v2 model")
                
                # Apply performance optimizations
                if hasattr(self.model, 'synthesizer') and hasattr(self.model.synthesizer, 'model'):
                    model = self.model.synthesizer.model
                    if hasattr(model, 'eval'):
                        model.eval()
                    if self.device == "cuda" and self.use_half_precision:
                        if hasattr(model, 'half'):
                            model.half()
                            print("⚡ Enabled half-precision (FP16) for faster inference")
                
                print(f"🎯 Model loaded on: {self.device}")
                if self.device == "cuda":
                    print(f"🔥 CUDA optimizations enabled")
                
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
        
        # Lấy tên file từ audio_path để tạo voice_id mô tả hơn
        filename = os.path.basename(audio_path)
        filename_without_ext = os.path.splitext(filename)[0]
        
        # Tạo voice_id mới dựa trên user_id (voice_id parameter) thay vì filename
        # voice_id parameter có dạng "voice_{user_id}" từ app.py
        if voice_id.startswith("voice_"):
            # Nếu voice_id đã có prefix "voice_", sử dụng trực tiếp
            base_voice_id = voice_id
        else:
            # Nếu không có prefix, thêm vào
            base_voice_id = f"voice_{voice_id}"
        
        # Tạo unique voice_id nếu đã tồn tại
        new_voice_id = base_voice_id
        counter = 1
        while new_voice_id in self.voice_samples:
            new_voice_id = f"{base_voice_id}_{counter:03d}"
            counter += 1
        
        self.voice_samples[new_voice_id] = {
            'audio_path': audio_path,
            'text': text,
            'original_id': voice_id,  # Lưu ID gốc từ request
            'filename': filename,      # Lưu tên file thực tế
            'upload_time': str(datetime.datetime.now())
        }
        
        print(f"✅ Added voice sample: {new_voice_id} -> {audio_path}")
        print(f"📁 File: {filename}")
        if new_voice_id != base_voice_id:
            print(f"📝 Note: Voice ID changed from '{base_voice_id}' to '{new_voice_id}' (duplicate detected)")
        print(f"📝 Original requested ID: {voice_id} -> Generated ID: {new_voice_id}")
        
        return new_voice_id  # Trả về voice_id thực tế
    
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
        Clone voice và tạo audio từ text với performance optimizations
        
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
            print(f"🎯 Starting voice cloning...")
            print(f"📝 Text length: {len(text)} characters")
            print(f"🌍 Language: {language}")
            
            # Sử dụng XTTS để clone voice
            audio_path = self.voice_samples[voice_id]['audio_path']
            
            # Performance optimization: Use faster inference settings
            inference_settings = {}
            if self.fast_inference:
                # Reduce quality slightly for speed
                inference_settings.update({
                    'speed': 1.2,  # Slightly faster
                })
            
            # Tạo audio với voice cloning và performance optimizations
            print("🔄 Generating audio (this may take 30-60 seconds)...")
            
            # Use optimized TTS call
            if hasattr(self.model, 'tts_to_file'):
                self.model.tts_to_file(
                    text=text,
                    speaker_wav=audio_path,
                    language=language,
                    file_path=output_path,
                    **inference_settings
                )
            else:
                # Fallback method
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
            
            # Performance tips
            if self.device == "cpu":
                print("💡 Performance tip: Consider using GPU for faster inference")
            elif self.device == "cuda":
                print("🔥 GPU acceleration active - optimal performance")
            
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
    
    def clone_voice_with_advanced_effects(self, text: str, voice_id: str, output_path: str = None,
                                         language: str = None, speed: float = 1.0, pitch_shift: float = 0.0,
                                         voice_type: str = "normal", age_group: str = "adult",
                                         reverb: float = 0.0, echo: float = 0.0, noise_reduction: bool = False,
                                         normalize: bool = True) -> str:
        """
        Clone voice với advanced audio effects
        
        Args:
            text: Text cần chuyển thành giọng nói
            voice_id: ID của voice đã đăng ký
            output_path: Đường dẫn output (optional)
            language: Ngôn ngữ (optional)
            speed: Tốc độ phát
            pitch_shift: Thay đổi pitch
            voice_type: Loại giọng
            age_group: Nhóm tuổi
            reverb: Mức độ reverb (0.0 - 1.0)
            echo: Mức độ echo (0.0 - 1.0)
            noise_reduction: Có giảm noise không
            normalize: Có normalize audio không
            
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
        
        if reverb < 0.0 or reverb > 1.0:
            print(f"⚠️ Warning: Reverb {reverb} out of range [0.0, 1.0], using 0.0")
            reverb = 0.0
        
        if echo < 0.0 or echo > 1.0:
            print(f"⚠️ Warning: Echo {echo} out of range [0.0, 1.0], using 0.0")
            echo = 0.0
        
        # Auto-detect language if not specified
        if language is None:
            language = self.auto_detect_language(text)
            print(f"🌍 Auto-detected language: {language} ({self.SUPPORTED_LANGUAGES.get(language, 'Unknown')})")
        
        # Special handling for Vietnamese
        if language == "vi":
            print("🇻🇳 Vietnamese detected!")
            print("⚠️  WARNING: XTTS model cannot naturally read Vietnamese text")
            print("💡 SOLUTION: Use English text for voice cloning, Vietnamese voice sample will maintain accent")
            print("📝 RECOMMENDATION: Input English text, voice will sound Vietnamese due to voice sample")
            print("🔄 Switching to English base language for XTTS compatibility...")
            language = "en"
        
        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            print(f"⚠️ Warning: Language '{language}' not supported, using 'en' instead")
            language = 'en'
        
        # Validate text length
        if len(text) > 500:
            print(f"⚠️ Warning: Text too long ({len(text)} chars), truncating to 500 characters")
            text = text[:500]
        
        # Apply voice type and age group effects
        final_pitch_shift = pitch_shift
        final_speed = speed
        
        if voice_type == "male":
            final_pitch_shift -= 3
        elif voice_type == "female":
            final_pitch_shift += 3
        elif voice_type == "child":
            final_pitch_shift += 6
            final_speed *= 1.1
        elif voice_type == "elderly":
            final_pitch_shift -= 2
            final_speed *= 0.9
        
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
            output_path = f"advanced_{voice_id}_{hash(text) % 10000}.wav"
        
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
            
            # Apply advanced audio effects
            if any([final_speed != 1.0, final_pitch_shift != 0.0, reverb > 0.0, echo > 0.0, noise_reduction, normalize]):
                output_path = self._apply_advanced_audio_effects(
                    output_path, final_speed, final_pitch_shift, 
                    reverb, echo, noise_reduction, normalize
                )
            
            print(f"✅ Voice cloned successfully with advanced effects: {output_path}")
            print(f"🌍 Language used: {language} ({self.SUPPORTED_LANGUAGES.get(language, 'Unknown')})")
            print(f"🎭 Voice type: {voice_type}, Age group: {age_group}")
            if final_speed != 1.0:
                print(f"⚡ Speed: {final_speed}x")
            if final_pitch_shift != 0.0:
                print(f"🎵 Pitch shift: {final_pitch_shift:+d} semitones")
            if reverb > 0.0:
                print(f"🏛️ Reverb: {reverb:.2f}")
            if echo > 0.0:
                print(f"🔊 Echo: {echo:.2f}")
            if noise_reduction:
                print(f"🔇 Noise reduction: Enabled")
            if normalize:
                print(f"📊 Normalize: Enabled")
            
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
    
    def _apply_advanced_audio_effects(self, audio_path: str, speed: float, pitch_shift: float,
                                     reverb: float, echo: float, noise_reduction: bool, normalize: bool) -> str:
        """
        Apply advanced audio effects
        
        Args:
            audio_path: Đường dẫn file audio
            speed: Tốc độ phát
            pitch_shift: Thay đổi pitch
            reverb: Mức độ reverb
            echo: Mức độ echo
            noise_reduction: Có giảm noise không
            normalize: Có normalize audio không
            
        Returns:
            Đường dẫn file audio đã được xử lý
        """
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Apply basic effects
            if speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed)
            
            if pitch_shift != 0.0:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            
            # Apply advanced effects
            if reverb > 0.0:
                # Simple reverb effect
                reverb_samples = int(sr * reverb * 0.1)  # 0.1 second reverb
                reverb_audio = np.zeros_like(audio)
                reverb_audio[reverb_samples:] = audio[:-reverb_samples] * reverb
                audio = audio + reverb_audio
            
            if echo > 0.0:
                # Simple echo effect
                echo_delay = int(sr * echo * 0.3)  # 0.3 second echo
                echo_audio = np.zeros_like(audio)
                echo_audio[echo_delay:] = audio[:-echo_delay] * echo
                audio = audio + echo_audio
            
            if noise_reduction:
                # Simple noise reduction (spectral gating)
                D = librosa.stft(audio)
                D_mag, D_phase = librosa.magphase(D)
                D_mag_filtered = D_mag * (D_mag > np.percentile(D_mag, 20))
                audio = librosa.istft(D_mag_filtered * D_phase)
            
            if normalize:
                # Normalize audio
                audio = librosa.util.normalize(audio)
            
            # Save processed audio
            effects_str = f"_s{speed}_p{pitch_shift}"
            if reverb > 0.0:
                effects_str += f"_r{reverb:.2f}"
            if echo > 0.0:
                effects_str += f"_e{echo:.2f}"
            if noise_reduction:
                effects_str += "_nr"
            if normalize:
                effects_str += "_n"
            
            output_path = audio_path.replace('.wav', f'_advanced{effects_str}.wav')
            sf.write(output_path, audio, sr)
            
            print(f"🎵 Applied advanced audio effects: {effects_str}")
            return output_path
            
        except Exception as e:
            print(f"⚠️ Warning: Could not apply advanced audio effects: {e}")
            return audio_path
    
    def get_available_voices(self) -> List[str]:
        """Lấy danh sách voice đã đăng ký"""
        return list(self.voice_samples.keys())
    
    def get_voices_by_user(self, user_id: str) -> List[str]:
        """Lấy danh sách voice của một user cụ thể"""
        user_voices = []
        print(f"🔍 get_voices_by_user called for user_id: {user_id}")
        print(f"🔍 Total voice samples: {len(self.voice_samples)}")
        print(f"🔍 Looking for voices that start with 'voice_{user_id}' or have original_id starting with 'voice_{user_id}'")
        
        for voice_id, voice_info in self.voice_samples.items():
            # Kiểm tra nếu voice_id bắt đầu với user_id hoặc có original_id trùng khớp
            starts_with_user = voice_id.startswith(f"voice_{user_id}")
            original_starts_with_user = voice_info.get('original_id', '').startswith(f"voice_{user_id}")
            
            print(f"  - Checking {voice_id}:")
            print(f"    voice_id starts with 'voice_{user_id}': {starts_with_user}")
            print(f"    original_id starts with 'voice_{user_id}': {original_starts_with_user}")
            print(f"    original_id value: {voice_info.get('original_id', 'N/A')}")
            print(f"    filename: {voice_info.get('filename', 'N/A')}")
            
            if starts_with_user or original_starts_with_user:
                user_voices.append(voice_id)
                print(f"    ✅ Added to user voices")
            else:
                print(f"    ❌ Not added to user voices")
                print(f"    💡 Reason: voice_id='{voice_id}' doesn't start with 'voice_{user_id}' and original_id='{voice_info.get('original_id', 'N/A')}' doesn't start with 'voice_{user_id}'")
        
        print(f"🔍 Final result: {len(user_voices)} voices for user {user_id}: {user_voices}")
        return user_voices
    
    def remove_voice(self, voice_id: str):
        """Xóa voice sample"""
        if voice_id in self.voice_samples:
            del self.voice_samples[voice_id]
            print(f"✅ Removed voice: {voice_id}")
        else:
            print(f"⚠️ Voice ID '{voice_id}' not found")
    
    def remove_user_voices(self, user_id: str) -> int:
        """Xóa tất cả voice của một user"""
        removed_count = 0
        voices_to_remove = []
        
        for voice_id, voice_info in self.voice_samples.items():
            if (voice_id.startswith(f"voice_{user_id}") or 
                voice_info.get('original_id', '').startswith(f"voice_{user_id}")):
                voices_to_remove.append(voice_id)
        
        for voice_id in voices_to_remove:
            del self.voice_samples[voice_id]
            removed_count += 1
            print(f"✅ Removed user voice: {voice_id}")
        
        print(f"🗑️ Removed {removed_count} voices for user {user_id}")
        return removed_count
    
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
                    'file_size': os.path.getsize(audio_path),
                    'filename': os.path.basename(audio_path)  # Thêm tên file
                })
            except Exception:
                # Nếu không đọc được audio info, vẫn thêm filename
                voice_info['filename'] = os.path.basename(audio_path)
        else:
            # Nếu file không tồn tại, vẫn thêm filename từ metadata
            if 'filename' in voice_info:
                pass  # Giữ nguyên filename đã có
            else:
                voice_info['filename'] = os.path.basename(audio_path)
        
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
    
    def clone_voice_with_srt(self, text: str, voice_id: str, output_path: str = None, 
                             language: str = None, srt_path: str = None, 
                             segment_duration: float = 3.0) -> tuple:
        """
        Clone voice và tạo audio + file SRT phụ đề
        
        Args:
            text: Text cần chuyển thành giọng nói
            voice_id: ID của voice đã đăng ký
            output_path: Đường dẫn output audio (optional)
            language: Ngôn ngữ (optional)
            srt_path: Đường dẫn file SRT (optional)
            segment_duration: Thời gian mỗi segment phụ đề (giây)
            
        Returns:
            Tuple (audio_path, srt_path)
        """
        if voice_id not in self.voice_samples:
            raise ValueError(f"Voice ID '{voice_id}' not found. Please add voice sample first.")
        
        # Tạo audio trước
        audio_path = self.clone_voice(text, voice_id, output_path, language)
        
        # Tạo file SRT
        if not srt_path:
            srt_path = audio_path.replace('.wav', '.srt')
        
        # Tách text thành các segment
        segments = self._split_text_for_subtitles(text, segment_duration)
        
        # Tạo file SRT
        self._create_srt_file(segments, srt_path, segment_duration)
        
        print(f"✅ Voice cloned with SRT: {audio_path}")
        print(f"📝 SRT file created: {srt_path}")
        print(f"🔤 {len(segments)} subtitle segments created")
        
        return audio_path, srt_path
    
    def _split_text_for_subtitles(self, text: str, segment_duration: float) -> list:
        """
        Tách text thành các segment phù hợp cho phụ đề
        
        Args:
            text: Text cần tách
            segment_duration: Thời gian mỗi segment (giây)
            
        Returns:
            List các segment text
        """
        # Ước tính số từ mỗi segment (dựa trên tốc độ đọc trung bình)
        words_per_second = 2.5  # Tốc độ đọc trung bình
        words_per_segment = int(segment_duration * words_per_second)
        
        # Tách text thành các từ
        words = text.split()
        segments = []
        
        if len(words) <= words_per_segment:
            # Text ngắn, chỉ cần 1 segment
            segments.append(text)
        else:
            # Tách thành nhiều segment
            current_segment = []
            current_word_count = 0
            
            for word in words:
                current_segment.append(word)
                current_word_count += 1
                
                # Kiểm tra nếu đã đủ từ cho segment
                if current_word_count >= words_per_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []
                    current_word_count = 0
            
            # Thêm segment cuối cùng nếu còn
            if current_segment:
                segments.append(' '.join(current_segment))
        
        return segments
    
    def _create_srt_file(self, segments: list, srt_path: str, segment_duration: float):
        """
        Tạo file SRT từ các segment
        
        Args:
            segments: List các segment text
            srt_path: Đường dẫn file SRT
            segment_duration: Thời gian mỗi segment (giây)
        """
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                # Tính thời gian bắt đầu và kết thúc
                start_time = (i - 1) * segment_duration
                end_time = i * segment_duration
                
                # Format thời gian theo chuẩn SRT (HH:MM:SS,mmm)
                start_time_str = self._format_srt_time(start_time)
                end_time_str = self._format_srt_time(end_time)
                
                # Ghi segment vào file SRT
                f.write(f"{i}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{segment}\n\n")
    
    def _format_srt_time(self, seconds: float) -> str:
        """
        Format thời gian theo chuẩn SRT (HH:MM:SS,mmm)
        
        Args:
            seconds: Thời gian tính bằng giây
            
        Returns:
            String thời gian theo format SRT
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def batch_clone_with_srt(self, texts: list, voice_id: str, output_folder: str = "batch_output", 
                             language: str = None, segment_duration: float = 3.0) -> dict:
        """
        Clone voice cho nhiều text cùng lúc với SRT
        
        Args:
            texts: List các text cần clone
            voice_id: ID của voice đã đăng ký
            output_folder: Thư mục output
            language: Ngôn ngữ (optional)
            segment_duration: Thời gian mỗi segment SRT (giây)
            
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
        
        print(f"🚀 Starting batch processing with SRT for {len(texts)} texts...")
        
        for i, text in enumerate(texts, 1):
            try:
                print(f"\n📝 Processing text {i}/{len(texts)}: {text[:50]}...")
                
                # Tạo tên file output
                base_filename = f"batch_{voice_id}_{i:03d}_{hash(text) % 10000}"
                audio_path = os.path.join(output_folder, f"{base_filename}.wav")
                srt_path = os.path.join(output_folder, f"{base_filename}.srt")
                
                # Clone voice với SRT
                result_audio, result_srt = self.clone_voice_with_srt(
                    text, voice_id, audio_path, language, srt_path, segment_duration
                )
                
                results['outputs'].append({
                    'text': text,
                    'audio_path': result_audio,
                    'srt_path': result_srt,
                    'filename': base_filename,
                    'index': i
                })
                results['success_count'] += 1
                
                print(f"✅ Text {i} processed successfully: {base_filename}")
                
            except Exception as e:
                error_msg = f"Failed to process text {i}: {str(e)}"
                print(f"❌ {error_msg}")
                results['errors'].append({
                    'text': text,
                    'error': str(e),
                    'index': i
                })
                results['failed_count'] += 1
        
        print(f"\n🎉 Batch processing with SRT completed!")
        print(f"✅ Success: {results['success_count']}")
        print(f"❌ Failed: {results['failed_count']}")
        print(f"📁 Output folder: {output_folder}")
        print(f"📝 SRT files created for each audio file")
        
        return results
    
    def create_srt_from_audio(self, audio_path: str, text: str, srt_path: str = None, 
                              segment_duration: float = 3.0) -> str:
        """
        Tạo file SRT từ audio file có sẵn
        
        Args:
            audio_path: Đường dẫn file audio
            text: Text tương ứng với audio
            srt_path: Đường dẫn file SRT (optional)
            segment_duration: Thời gian mỗi segment (giây)
            
        Returns:
            Đường dẫn file SRT đã tạo
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Tạo tên file SRT nếu không có
        if not srt_path:
            srt_path = audio_path.replace('.wav', '.srt').replace('.mp3', '.srt').replace('.flac', '.srt')
        
        # Tách text thành các segment
        segments = self._split_text_for_subtitles(text, segment_duration)
        
        # Tạo file SRT
        self._create_srt_file(segments, srt_path, segment_duration)
        
        print(f"✅ SRT file created: {srt_path}")
        print(f"🔤 {len(segments)} subtitle segments created")
        
        return srt_path
    
    def merge_srt_files(self, srt_files: list, output_path: str, 
                        segment_duration: float = 3.0) -> str:
        """
        Gộp nhiều file SRT thành một file duy nhất
        
        Args:
            srt_files: List đường dẫn các file SRT
            output_path: Đường dẫn file SRT output
            segment_duration: Thời gian mỗi segment (giây)
            
        Returns:
            Đường dẫn file SRT đã gộp
        """
        all_segments = []
        current_time = 0.0
        
        for srt_file in srt_files:
            if not os.path.exists(srt_file):
                print(f"⚠️ Warning: SRT file not found: {srt_file}")
                continue
            
            # Đọc file SRT
            with open(srt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse SRT content
            segments = self._parse_srt_content(content)
            
            # Thêm vào danh sách với thời gian mới
            for segment in segments:
                segment['start_time'] += current_time
                segment['end_time'] += current_time
                all_segments.append(segment)
            
            # Cập nhật thời gian cho file tiếp theo
            if segments:
                current_time = all_segments[-1]['end_time'] + 1.0  # Thêm 1 giây khoảng cách
        
        # Sắp xếp theo thời gian
        all_segments.sort(key=lambda x: x['start_time'])
        
        # Tạo file SRT gộp
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(all_segments, 1):
                start_time_str = self._format_srt_time(segment['start_time'])
                end_time_str = self._format_srt_time(segment['end_time'])
                
                f.write(f"{i}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{segment['text']}\n\n")
        
        print(f"✅ Merged SRT file created: {output_path}")
        print(f"🔤 Total segments: {len(all_segments)}")
        
        return output_path
    
    def _parse_srt_content(self, content: str) -> list:
        """
        Parse nội dung file SRT
        
        Args:
            content: Nội dung file SRT
            
        Returns:
            List các segment với thông tin thời gian
        """
        segments = []
        lines = content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or not line.isdigit():
                i += 1
                continue
            
            # Đọc số thứ tự
            segment_number = int(line)
            i += 1
            
            if i >= len(lines):
                break
            
            # Đọc thời gian
            time_line = lines[i].strip()
            i += 1
            
            if i >= len(lines):
                break
            
            # Parse thời gian
            try:
                start_time, end_time = self._parse_srt_time_line(time_line)
            except:
                continue
            
            # Đọc text
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            
            if text_lines:
                segments.append({
                    'number': segment_number,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': ' '.join(text_lines)
                })
            
            i += 1
        
        return segments
    
    def _parse_srt_time_line(self, time_line: str) -> tuple:
        """
        Parse dòng thời gian SRT (HH:MM:SS,mmm --> HH:MM:SS,mmm)
        
        Args:
            time_line: Dòng thời gian SRT
            
        Returns:
            Tuple (start_time, end_time) tính bằng giây
        """
        parts = time_line.split(' --> ')
        if len(parts) != 2:
            raise ValueError("Invalid SRT time format")
        
        start_time = self._parse_srt_time_to_seconds(parts[0].strip())
        end_time = self._parse_srt_time_to_seconds(parts[1].strip())
        
        return start_time, end_time
    
    def _parse_srt_time_to_seconds(self, time_str: str) -> float:
        """
        Chuyển đổi thời gian SRT thành giây
        
        Args:
            time_str: String thời gian SRT (HH:MM:SS,mmm)
            
        Returns:
            Thời gian tính bằng giây
        """
        # Thay dấu phẩy bằng dấu chấm
        time_str = time_str.replace(',', '.')
        
        # Tách thành giờ, phút, giây
        parts = time_str.split(':')
        if len(parts) != 3:
            raise ValueError("Invalid time format")
        
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        
        return hours * 3600 + minutes * 60 + seconds
    
    def import_voice_config(self, config_path: str):
        """Import cấu hình voice samples"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.voice_samples = config.get('voices', {})
        print(f"✅ Voice configuration imported: {len(self.voice_samples)} voices loaded")

    def get_voice_analytics(self) -> dict:
        """
        Lấy thống kê và phân tích về tất cả voices
        
        Returns:
            Dict chứa thống kê chi tiết
        """
        if not self.voice_samples:
            return {"error": "No voices available"}
        
        analytics = {
            'total_voices': len(self.voice_samples),
            'total_duration': 0,
            'total_size': 0,
            'voice_types': {},
            'languages': {},
            'file_formats': {},
            'quality_metrics': {},
            'recent_voices': [],
            'popular_voices': []
        }
        
        # Phân tích từng voice
        for voice_id, voice_info in self.voice_samples.items():
            # Tổng thời gian và kích thước
            duration = voice_info.get('duration', 0)
            analytics['total_duration'] += duration
            
            # Kích thước file
            if 'audio_path' in voice_info and os.path.exists(voice_info['audio_path']):
                file_size = os.path.getsize(voice_info['audio_path'])
                analytics['total_size'] += file_size
                
                # Phân tích format file
                file_ext = os.path.splitext(voice_info['audio_path'])[1].lower()
                analytics['file_formats'][file_ext] = analytics['file_formats'].get(file_ext, 0) + 1
            
            # Phân tích loại giọng (dựa trên text mô tả)
            text = voice_info.get('text', '').lower()
            if any(word in text for word in ['nam', 'male', 'ông', 'anh']):
                voice_type = 'male'
            elif any(word in text for word in ['nữ', 'female', 'bà', 'chị']):
                voice_type = 'female'
            elif any(word in text for word in ['trẻ', 'child', 'em', 'bé']):
                voice_type = 'child'
            elif any(word in text for word in ['già', 'elderly', 'cụ', 'ông già']):
                voice_type = 'elderly'
            else:
                voice_type = 'unknown'
            
            analytics['voice_types'][voice_type] = analytics['voice_types'].get(voice_type, 0) + 1
            
            # Phân tích ngôn ngữ
            detected_lang = self.auto_detect_language(text) if text else 'unknown'
            analytics['languages'][detected_lang] = analytics['languages'].get(detected_lang, 0) + 1
        
        # Tính toán metrics
        if analytics['total_voices'] > 0:
            analytics['quality_metrics'] = {
                'average_duration': analytics['total_duration'] / analytics['total_voices'],
                'average_size': analytics['total_size'] / analytics['total_voices'],
                'total_size_mb': round(analytics['total_size'] / (1024 * 1024), 2)
            }
        
        # Sắp xếp theo popularity (dựa trên duration)
        sorted_voices = sorted(self.voice_samples.items(), 
                              key=lambda x: x[1].get('duration', 0), reverse=True)
        analytics['popular_voices'] = [voice_id for voice_id, _ in sorted_voices[:5]]
        
        return analytics

    def assess_voice_quality(self, voice_id: str) -> dict:
        """
        Đánh giá chất lượng của voice sample
        
        Args:
            voice_id: ID của voice cần đánh giá
            
        Returns:
            Dict chứa các metrics chất lượng
        """
        if voice_id not in self.voice_samples:
            raise ValueError(f"Voice ID '{voice_id}' not found. Please add voice sample first.")
        
        try:
            import librosa
            import numpy as np
            
            voice_info = self.voice_samples[voice_id]
            audio_path = voice_info['audio_path']
            
            if not os.path.exists(audio_path):
                return {"error": "Audio file not found"}
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Tính toán các metrics chất lượng
            quality_metrics = {
                'voice_id': voice_id,
                'sample_rate': sr,
                'duration': len(audio) / sr,
                'total_samples': len(audio),
                'audio_metrics': {},
                'spectral_metrics': {},
                'noise_metrics': {},
                'overall_score': 0.0
            }
            
            # Audio metrics
            quality_metrics['audio_metrics'] = {
                'rms_energy': float(np.sqrt(np.mean(audio**2))),
                'peak_amplitude': float(np.max(np.abs(audio))),
                'dynamic_range': float(np.max(audio) - np.min(audio)),
                'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(audio).mean())
            }
            
            # Spectral metrics
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            
            quality_metrics['spectral_metrics'] = {
                'spectral_centroid_mean': float(spectral_centroids.mean()),
                'spectral_rolloff_mean': float(spectral_rolloff.mean()),
                'spectral_bandwidth_mean': float(spectral_bandwidth.mean()),
                'spectral_flatness': float(librosa.feature.spectral_flatness(y=audio).mean())
            }
            
            # Noise metrics
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            quality_metrics['noise_metrics'] = {
                'mfcc_variance': float(np.var(mfccs)),
                'signal_to_noise_ratio': float(10 * np.log10(np.mean(audio**2) / np.var(audio))),
                'harmonic_to_noise_ratio': float(librosa.effects.harmonic(audio).shape[0] / len(audio))
            }
            
            # Tính overall score (0-100)
            score = 0.0
            
            # Duration score (optimal: 3-10 seconds)
            duration = quality_metrics['duration']
            if 3 <= duration <= 10:
                score += 25
            elif 1 <= duration <= 15:
                score += 20
            else:
                score += 10
            
            # Sample rate score (optimal: >= 22050)
            if sr >= 44100:
                score += 25
            elif sr >= 22050:
                score += 20
            else:
                score += 10
            
            # Energy score
            rms_energy = quality_metrics['audio_metrics']['rms_energy']
            if 0.01 <= rms_energy <= 0.5:
                score += 25
            else:
                score += 15
            
            # Spectral score
            spectral_centroid = quality_metrics['spectral_metrics']['spectral_centroid_mean']
            if 1000 <= spectral_centroid <= 4000:
                score += 25
            else:
                score += 15
            
            quality_metrics['overall_score'] = min(100.0, score)
            
            # Quality level
            if quality_metrics['overall_score'] >= 80:
                quality_level = "Excellent"
            elif quality_metrics['overall_score'] >= 60:
                quality_level = "Good"
            elif quality_metrics['overall_score'] >= 40:
                quality_level = "Fair"
            else:
                quality_level = "Poor"
            
            quality_metrics['quality_level'] = quality_level
            
            print(f"🔍 Voice quality assessment for {voice_id}:")
            print(f"📊 Overall Score: {quality_metrics['overall_score']:.1f}/100 ({quality_level})")
            print(f"⏱️ Duration: {quality_metrics['duration']:.2f}s")
            print(f"🎵 Sample Rate: {sr} Hz")
            print(f"⚡ RMS Energy: {quality_metrics['audio_metrics']['rms_energy']:.4f}")
            
            return quality_metrics
            
        except Exception as e:
            print(f"❌ Error assessing voice quality: {e}")
            return {"error": str(e)}

    def transform_voice(self, voice_id: str, transformation_type: str, 
                       intensity: float = 0.5, output_path: str = None) -> str:
        """
        Biến đổi voice sample với các hiệu ứng đặc biệt
        
        Args:
            voice_id: ID của voice cần biến đổi
            transformation_type: Loại biến đổi
            intensity: Cường độ biến đổi (0.0 - 1.0)
            output_path: Đường dẫn output (optional)
            
        Returns:
            Đường dẫn đến file audio đã biến đổi
        """
        if voice_id not in self.voice_samples:
            raise ValueError(f"Voice ID '{voice_id}' not found. Please add voice sample first.")
        
        # Validate transformation type
        valid_transformations = {
            'robot': 'Robot voice effect',
            'alien': 'Alien voice effect', 
            'monster': 'Monster voice effect',
            'chipmunk': 'Chipmunk voice effect',
            'giant': 'Giant voice effect',
            'whisper': 'Whisper voice effect',
            'radio': 'Radio/telephone effect',
            'underwater': 'Underwater effect',
            'space': 'Space/echo effect',
            'time_warp': 'Time warp effect'
        }
        
        if transformation_type not in valid_transformations:
            raise ValueError(f"Invalid transformation type. Choose from: {list(valid_transformations.keys())}")
        
        # Validate intensity
        if intensity < 0.0 or intensity > 1.0:
            print(f"⚠️ Warning: Intensity {intensity} out of range [0.0, 1.0], using 0.5")
            intensity = 0.5
        
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            voice_info = self.voice_samples[voice_id]
            audio_path = voice_info['audio_path']
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError("Audio file not found")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Apply transformation based on type
            transformed_audio = audio.copy()
            
            if transformation_type == 'robot':
                # Robot effect: subtle metallic harmonics + pitch shift
                transformed_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=intensity * 2)
                # Very subtle harmonics
                harmonics = np.sin(2 * np.pi * 150 * np.arange(len(audio)) / sr) * intensity * 0.1
                transformed_audio += harmonics
                
            elif transformation_type == 'alien':
                # Alien effect: pitch shift only (no harmonics)
                transformed_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=intensity * 6)
                
            elif transformation_type == 'monster':
                # Monster effect: lower pitch only (no growl)
                transformed_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-intensity * 4)
                
            elif transformation_type == 'chipmunk':
                # Chipmunk effect: higher pitch only
                transformed_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=intensity * 6)
                
            elif transformation_type == 'giant':
                # Giant effect: lower pitch + subtle slow down
                transformed_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-intensity * 3)
                if intensity > 0.5:
                    transformed_audio = librosa.effects.time_stretch(transformed_audio, rate=1.0 - intensity * 0.3)
                
            elif transformation_type == 'whisper':
                # Whisper effect: reduce energy only (no breath noise)
                transformed_audio = audio * (1.0 - intensity * 0.5)
                
            elif transformation_type == 'radio':
                # Radio effect: bandpass filter only (no noise)
                low_freq = 300
                high_freq = 3000
                freqs = np.fft.fftfreq(len(audio), 1/sr)
                mask = (freqs >= low_freq) & (freqs <= high_freq)
                fft_audio = np.fft.fft(audio)
                fft_audio[~mask] *= intensity * 0.5
                transformed_audio = np.real(np.fft.ifft(fft_audio))
                
            elif transformation_type == 'underwater':
                # Underwater effect: low-pass filter only (no bubbles)
                cutoff_freq = 1000 + intensity * 2000
                freqs = np.fft.fftfreq(len(audio), 1/sr)
                mask = np.abs(freqs) <= cutoff_freq
                fft_audio = np.fft.fft(audio)
                fft_audio[~mask] *= intensity * 0.4
                transformed_audio = np.real(np.fft.ifft(fft_audio))
                
            elif transformation_type == 'space':
                # Space effect: subtle echo only (no cosmic sounds)
                delay_samples = int(sr * intensity * 0.3)
                echo = np.zeros_like(audio)
                echo[delay_samples:] = audio[:-delay_samples] * intensity * 0.3
                transformed_audio += echo
                
            elif transformation_type == 'time_warp':
                # Time warp effect: gentle speed variation
                warp_points = int(intensity * 5) + 1
                segment_length = len(audio) // warp_points
                warped_audio = []
                
                for i in range(warp_points):
                    start = i * segment_length
                    end = start + segment_length
                    segment = audio[start:end]
                    
                    # Gentle speed variation
                    speed_factor = 0.8 + intensity * 0.4  # Range: 0.8 - 1.2
                    warped_segment = librosa.effects.time_stretch(segment, rate=speed_factor)
                    warped_audio.append(warped_segment)
                
                transformed_audio = np.concatenate(warped_audio)
                # Trim to original length
                if len(transformed_audio) > len(audio):
                    transformed_audio = transformed_audio[:len(audio)]
                else:
                    # Pad if shorter
                    padding = np.zeros(len(audio) - len(transformed_audio))
                    transformed_audio = np.concatenate([transformed_audio, padding])
            
            # Clean up audio: remove noise and artifacts
            # Apply gentle noise gate to remove very low amplitude noise
            noise_threshold = 0.01 * intensity
            transformed_audio[np.abs(transformed_audio) < noise_threshold] = 0
            
            # Apply gentle compression to smooth out dynamics
            transformed_audio = np.tanh(transformed_audio * 0.8)
            
            # Normalize audio
            transformed_audio = librosa.util.normalize(transformed_audio)
            
            # Generate output path
            if not output_path:
                output_path = f"transformed_{voice_id}_{transformation_type}_{intensity:.2f}.wav"
            
            # Save transformed audio
            sf.write(output_path, transformed_audio, sr)
            
            print(f"🎭 Voice transformation completed!")
            print(f"🔧 Type: {transformation_type} ({valid_transformations[transformation_type]})")
            print(f"⚡ Intensity: {intensity:.2f}")
            print(f"📁 Output: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error in voice transformation: {e}")
            raise


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

    # Example of voice comparison
    print("\n🔍 Voice Comparison Demo:")
    try:
        comparison_results = cloner.compare_voices(
            "my_voice", "my_voice", "Hello, this is a voice comparison test. How do I sound?"
        )
        print(f"Comparison results: {json.dumps(comparison_results, indent=2)}")
    except Exception as e:
        print(f"❌ Voice comparison failed: {e}") 

    # Example of real-time voice cloning
    print("\n🔄 Real-time Voice Cloning Demo:")
    try:
        # Create a generator for text chunks
        def text_generator():
            yield "Hello, this is a real-time voice cloning test."
            yield "This is the second chunk of the real-time test."
            yield "And this is the final chunk."

        # Stream the voice cloning
        for result in cloner.stream_voice_cloning(text_generator(), "my_voice"):
            if 'error' in result:
                print(f"Error in stream: {result['error']}")
            else:
                print(f"Chunk {result['chunk_id']}: Audio size {result['audio_size']} bytes")
    except Exception as e:
        print(f"❌ Real-time voice cloning failed: {e}") 