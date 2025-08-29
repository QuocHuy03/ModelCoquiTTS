#!/usr/bin/env python3
"""
Voice Cloning Web Interface
Flask app để test voice cloning model
"""

import os
import json
import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from voice_cloner import VoiceCloner
import uuid

# ------------------------------------------------------------------------------
# App & CORS
# ------------------------------------------------------------------------------
app = Flask(__name__, template_folder='web/templates')
CORS(app)

# ------------------------------------------------------------------------------
# Globals & Config
# ------------------------------------------------------------------------------
voice_cloner = None            # instance toàn cục
USERS = {}                     # cache người dùng trong RAM
INITIALIZED = False            # cờ đã khởi tạo chưa

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
USERS_FILE = 'users.json'

# Thư mục cần thiết
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def load_users():
    """Load predefined users from users.json (tùy chọn)."""
    global USERS
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Hỗ trợ {"users":[{user_id,...}]} hoặc {"028":{...}}
                if isinstance(data, dict) and 'users' in data and isinstance(data['users'], list):
                    USERS = {u.get('user_id'): u for u in data['users'] if u.get('user_id')}
                elif isinstance(data, dict):
                    USERS = {k: v for k, v in data.items()}
                else:
                    USERS = {}
        else:
            USERS = {}
    except Exception as e:
        print(f"⚠️ Failed to load users: {e}")
        USERS = {}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_request_user_id():
    """Lấy user_id từ header/args/form/json."""
    # Header
    user_id = request.headers.get('X-User-Id')
    if user_id:
        return user_id.strip()
    # Query
    user_id = request.args.get('user_id')
    if user_id:
        return user_id.strip()
    # Form
    if request.form is not None:
        user_id = request.form.get('user_id')
        if user_id:
            return user_id.strip()
    # JSON
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')
        if user_id:
            return user_id.strip()
    except Exception:
        pass
    return None

def get_user_voice_id(user_id: str) -> str:
    return f"voice_{user_id}"

def init_voice_cloner() -> bool:
    """Khởi tạo voice cloner/model."""
    global voice_cloner
    try:
        # Nếu có config model custom: models/*_config.json
        config_files = list(Path('models').glob('*_config.json'))
        if config_files:
            config_path = str(config_files[0])
            voice_cloner = VoiceCloner(model_path=config_path)
            print(f"✅ Loaded custom model from: {config_path}")
        else:
            # Mặc định XTTS
            voice_cloner = VoiceCloner()
            print("✅ Loaded default XTTS model")
        return True
    except Exception as e:
        print(f"❌ Error initializing voice cloner: {e}")
        voice_cloner = None
        return False

def ensure_initialized():
    """Đảm bảo đã load users + init model (gọi 1 lần mỗi worker)."""
    global INITIALIZED
    if INITIALIZED:
        return
    print("🎵 Starting Voice Cloning Web Interface (import-time init)...")
    load_users()
    ok = init_voice_cloner()
    if ok:
        print("✅ Voice cloner initialized successfully")
    else:
        print("❌ Failed to initialize voice cloner")
        print("⚠️ Some features may not work properly")
    INITIALIZED = True

# --- INIT ON IMPORT (cho Gunicorn/PM2 & dev) ---
ensure_initialized()

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/app')
def app_page():
    return render_template('index.html')

@app.route('/api/login', methods=['POST'])
def login():
    """Đăng nhập đơn giản bằng users.json (nếu có)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        user_id = (data.get('user_id') or '').strip()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403
        user_info = USERS.get(user_id, {'user_id': user_id})
        return jsonify({'success': True, 'user': user_info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voices', methods=['GET'])
def get_voices():
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    try:
        user_id = get_request_user_id()
        voices = voice_cloner.get_available_voices()
        voice_info = {}
        
        print(f"🔍 Debug: Found {len(voices)} voices: {voices}")
        print(f"👤 User ID: {user_id or 'None (showing all)'}")
        
        # Debug: In ra tất cả voice samples để kiểm tra
        print(f"🔍 All voice samples: {list(voice_cloner.voice_samples.keys())}")
        for v_id, v_info in voice_cloner.voice_samples.items():
            print(f"  - {v_id}: original_id={v_info.get('original_id', 'N/A')}, filename={v_info.get('filename', 'N/A')}")
        
        # Hiển thị tất cả voice nếu không có user_id, hoặc hiển thị voice của user cụ thể
        if user_id:
            # Lấy voice của user cụ thể
            user_voices = voice_cloner.get_voices_by_user(user_id)
            print(f"👤 Found {len(user_voices)} voices for user {user_id}: {user_voices}")
            
            # Nếu không có voice nào, tự động scan uploads folder
            if len(user_voices) == 0:
                print(f"🔍 No voices found for user {user_id}, scanning uploads folder...")
                try:
                    user_dir = os.path.join(UPLOAD_FOLDER, user_id)
                    if os.path.exists(user_dir):
                        # Scan và đăng ký các file MP3 cũ
                        audio_files = []
                        for filename in os.listdir(user_dir):
                            file_path = os.path.join(user_dir, filename)
                            if os.path.isfile(file_path) and allowed_file(filename):
                                audio_files.append(file_path)
                        
                        if audio_files:
                            print(f"🔍 Found {len(audio_files)} audio files in uploads folder")
                            for file_path in audio_files:
                                try:
                                    voice_id = get_user_voice_id(user_id)
                                    text = f"{os.path.basename(file_path)}"
                                    actual_voice_id = voice_cloner.add_voice_sample(voice_id, file_path, text)
                                    print(f"{os.path.basename(file_path)} -> {actual_voice_id}")
                                except Exception as e:
                                    print(f"❌ Failed to auto-register {file_path}: {e}")
                            
                            # Lấy lại danh sách voice sau khi đăng ký
                            user_voices = voice_cloner.get_voices_by_user(user_id)
                            print(f"👤 After auto-registration: {len(user_voices)} voices for user {user_id}")
                        else:
                            print(f"🔍 No audio files found in uploads folder: {user_dir}")
                    else:
                        print(f"🔍 User uploads directory not found: {user_dir}")
                except Exception as e:
                    print(f"❌ Error during auto-scan: {e}")
            
            # Debug: Kiểm tra logic matching
            print(f"🔍 Checking voice matching for user {user_id}:")
            for v_id, v_info in voice_cloner.voice_samples.items():
                starts_with_user = v_id.startswith(f"voice_{user_id}")
                original_starts_with_user = v_info.get('original_id', '').startswith(f"voice_{user_id}")
                print(f"  - {v_id}: starts_with_user={starts_with_user}, original_starts_with_user={original_starts_with_user}")
                if starts_with_user or original_starts_with_user:
                    print(f"    ✅ Would match user {user_id}")
                else:
                    print(f"    ❌ Would NOT match user {user_id}")
            
            voices_to_process = user_voices
        else:
            # Hiển thị tất cả voice
            voices_to_process = voices
        
        for v_id in voices_to_process:
            try:
                info = voice_cloner.get_voice_info(v_id)
                if info:
                    voice_info[v_id] = info
                    print(f"✅ Added voice {v_id}: {info.get('text', 'N/A')}")
                else:
                    print(f"⚠️ No info for voice {v_id}")
            except Exception as e:
                print(f"❌ Error getting info for voice {v_id}: {e}")
        
        print(f"📊 Final result: {len(voice_info)} voices in response")
        # Trả về count thực tế của voice được hiển thị
        return jsonify({'voices': voice_info, 'count': len(voice_info)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/all_voices', methods=['GET'])
def get_all_voices():
    """Lấy tất cả voice đã upload, không cần user_id."""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    try:
        voices = voice_cloner.get_available_voices()
        voice_info = {}
        
        print(f"🔍 Debug: Found {len(voices)} voices: {voices}")
        
        # Hiển thị tất cả voice
        for v_id in voices:
            try:
                info = voice_cloner.get_voice_info(v_id)
                if info:
                    voice_info[v_id] = info
                    print(f"✅ Added voice {v_id}: {info.get('text', 'N/A')}")
                else:
                    print(f"⚠️ No info for voice {v_id}")
            except Exception as e:
                print(f"❌ Error getting info for voice {v_id}: {e}")
        
        print(f"📊 Final result: {len(voice_info)} voices in response")
        return jsonify({'voices': voice_info, 'count': len(voice_info)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_voice', methods=['POST'])
def upload_voice():
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    try:
        user_id = get_request_user_id()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Force single voice per user
        voice_id = get_user_voice_id(user_id)
        text = (request.form.get('text') or 'Voice sample').strip()

        user_dir = os.path.join(UPLOAD_FOLDER, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Tự động rename file nếu trùng tên
        original_filename = file.filename
        filename_without_ext = os.path.splitext(original_filename)[0]
        file_extension = os.path.splitext(original_filename)[1]
        
        # Tạo tên file mới nếu trùng
        counter = 1
        new_filename = original_filename
        file_path = os.path.join(user_dir, new_filename)
        
        while os.path.exists(file_path):
            new_filename = f"{filename_without_ext}_{counter:03d}{file_extension}"
            file_path = os.path.join(user_dir, new_filename)
            counter += 1
        
        # Lưu file với tên mới
        file.save(file_path)
        
        print(f"📁 File saved: {original_filename} -> {new_filename}")
        if counter > 1:
            print(f"🔄 File renamed to avoid conflict: {new_filename}")

        # Thêm voice sample và lấy voice_id thực tế (có thể đã thay đổi nếu trùng lặp)
        actual_voice_id = voice_cloner.add_voice_sample(voice_id, file_path, text)
        
        # Trả về thông tin chi tiết về file
        response_data = {
            'success': True,
            'voice_id': actual_voice_id,
            'message': f'Voice sample added successfully: {actual_voice_id}',
            'original_requested_id': voice_id,
            'file_info': {
                'original_filename': original_filename,
                'saved_filename': new_filename,
                'file_path': file_path,
                'was_renamed': counter > 1
            }
        }
        
        if counter > 1:
            response_data['message'] += f' (File renamed from {original_filename} to {new_filename})'
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        user_id = (data.get('user_id') or '').strip()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403

        text = (data.get('text') or '').strip()
        if not text:
            return jsonify({'error': 'Text is required'}), 400

        voice_id = get_user_voice_id(user_id)
        if voice_id not in voice_cloner.get_available_voices():
            return jsonify({'error': f'Voice ID "{voice_id}" not found'}), 404

        language = (data.get('language') or '').strip()
        voice_type = data.get('voice_type', 'normal')
        age_group = data.get('age_group', 'adult')
        speed = data.get('speed', 1.0)
        pitch_shift = data.get('pitch_shift', 0)

        user_out_dir = os.path.join(OUTPUT_FOLDER, user_id)
        os.makedirs(user_out_dir, exist_ok=True)
        
        # Tạo tên file với timestamp để tránh trùng lặp
        timestamp = int(time.time())
        base_filename = f"output_{user_id}_{timestamp}"
        output_filename = f"{base_filename}.wav"
        srt_filename = f"{base_filename}.srt"
        
        output_path = os.path.join(user_out_dir, output_filename)
        srt_path = os.path.join(user_out_dir, srt_filename)

        # Clone voice và tạo SRT tự động
        if any([voice_type != 'normal', age_group != 'adult', speed != 1.0, pitch_shift != 0]):
            result_path = voice_cloner.clone_voice_with_advanced_effects(
                text, voice_id, output_path,
                language=language if language else None,
                speed=speed, pitch_shift=pitch_shift,
                voice_type=voice_type, age_group=age_group
            )
        else:
            result_path = (voice_cloner.clone_voice(text, voice_id, output_path, language)
                           if language else
                           voice_cloner.clone_voice(text, voice_id, output_path))

        # Tạo SRT file tự động
        srt_created = False
        if os.path.exists(result_path):
            try:
                # Sử dụng segment_duration mặc định là 3.0 giây
                segment_duration = data.get('segment_duration', 3.0)
                voice_cloner.create_srt_from_audio(
                    audio_path=str(result_path),
                    text=text,
                    srt_path=str(srt_path),
                    segment_duration=segment_duration
                )
                srt_created = True
                print(f"✅ SRT file created: {srt_path}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to create SRT file: {e}")
                srt_created = False

        if os.path.exists(result_path):
            response_data = {
                'success': True,
                'audio_url': f'/api/audio/{user_id}/{output_filename}',
                'message': 'Voice cloning completed successfully'
            }
            
            # Thêm thông tin SRT nếu tạo thành công
            if srt_created and os.path.exists(srt_path):
                response_data['srt_url'] = f'/api/files/{user_id}/{srt_filename}'
                response_data['srt_filename'] = srt_filename
                response_data['message'] = 'Voice cloning completed successfully with SRT subtitle'
            
            return jsonify(response_data)
        return jsonify({'error': 'Failed to generate audio'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/<user_id>/<filename>')
def get_audio(user_id, filename):
    try:
        file_path = os.path.join(OUTPUT_FOLDER, user_id, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='audio/wav')
        return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transform_voice', methods=['POST'])
def transform_voice():
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        user_id = (data.get('user_id') or '').strip()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403

        voice_id = get_user_voice_id(user_id)
        transformation_type = (data.get('transformation_type') or '').strip()
        intensity = data.get('intensity', 0.5)

        if not transformation_type:
            return jsonify({'error': 'Transformation type is required'}), 400
        if voice_id not in voice_cloner.get_available_voices():
            return jsonify({'error': f'Voice ID "{voice_id}" not found'}), 404

        user_out_dir = os.path.join(OUTPUT_FOLDER, user_id)
        os.makedirs(user_out_dir, exist_ok=True)
        output_filename = f"transform_{transformation_type}_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(user_out_dir, output_filename)

        result_path = voice_cloner.transform_voice(
            voice_id, transformation_type, intensity, output_path
        )
        if os.path.exists(result_path):
            return jsonify({
                'success': True,
                'audio_url': f'/api/audio/{user_id}/{output_filename}',
                'message': f'Voice transformation completed successfully: {transformation_type}'
            })
        return jsonify({'error': 'Failed to transform voice'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/assess_quality/<voice_id>', methods=['GET'])
def assess_quality(voice_id):
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    try:
        user_id = get_request_user_id()
        if user_id:
            # Kiểm tra xem voice_id có thuộc về user này không
            user_voices = voice_cloner.get_voices_by_user(user_id)
            if voice_id not in user_voices:
                return jsonify({'error': f'Voice ID "{voice_id}" does not belong to user {user_id}'}), 403
        if voice_id not in voice_cloner.get_available_voices():
            return jsonify({'error': f'Voice ID "{voice_id}" not found'}), 404

        quality_results = voice_cloner.assess_voice_quality(voice_id)
        formatted = []
        for key, value in quality_results.items():
            if key == 'overall_score':
                formatted.append({'name': 'Overall Score', 'value': f"{value:.1f}/100",
                                  'description': f'Overall quality score: {value:.1f} out of 100'})
            elif key == 'quality_level':
                formatted.append({'name': 'Quality Level', 'value': value,
                                  'description': f'Quality assessment: {value}'})
            elif key == 'duration':
                formatted.append({'name': 'Duration', 'value': f"{value:.1f}s",
                                  'description': f'Audio duration: {value:.1f} seconds'})
            elif key == 'sample_rate':
                formatted.append({'name': 'Sample Rate', 'value': f"{value} Hz",
                                  'description': f'Audio sample rate: {value} Hz'})
            elif key == 'rms_energy':
                formatted.append({'name': 'RMS Energy', 'value': f"{value:.3f}",
                                  'description': f'Root Mean Square energy: {value:.3f}'})
            elif key == 'peak_amplitude':
                formatted.append({'name': 'Peak Amplitude', 'value': f"{value:.3f}",
                                  'description': f'Peak audio amplitude: {value:.3f}'})
            elif key == 'spectral_centroids':
                formatted.append({'name': 'Spectral Centroids', 'value': f"{value:.1f} Hz",
                                  'description': f'Average spectral centroid: {value:.1f} Hz'})
            elif key == 'mfcc_variance':
                formatted.append({'name': 'MFCC Variance', 'value': f"{value:.3f}",
                                  'description': f'MFCC variance: {value:.3f}'})
            elif key == 'snr':
                formatted.append({'name': 'Signal-to-Noise Ratio', 'value': f"{value:.1f} dB",
                                  'description': f'SNR: {value:.1f} dB'})
        return jsonify({'success': True, 'message': f'Quality assessment completed for voice "{voice_id}"',
                        'results': formatted})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/remove_voice', methods=['DELETE'])
def remove_voice():
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        user_id = (data.get('user_id') or '').strip()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403
        voice_id = get_user_voice_id(user_id)
        voice_cloner.remove_voice(voice_id)
        return jsonify({'success': True, 'message': f'Voice "{voice_id}" removed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user_files', methods=['GET'])
def get_user_files():
    """Lấy danh sách file đã upload của user"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    try:
        user_id = get_request_user_id()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403
        
        # Lấy voice của user
        user_voices = voice_cloner.get_voices_by_user(user_id)
        files_info = []
        
        for voice_id in user_voices:
            voice_info = voice_cloner.get_voice_info(voice_id)
            if voice_info:
                file_info = {
                    'voice_id': voice_id,
                    'filename': voice_info.get('filename', 'Unknown'),
                    'original_filename': voice_info.get('original_id', 'Unknown'),
                    'file_path': voice_info.get('audio_path', ''),
                    'file_size': voice_info.get('file_size', 0),
                    'duration': voice_info.get('duration', 0),
                    'sample_rate': voice_info.get('sample_rate', 0),
                    'upload_time': voice_info.get('upload_time', ''),
                    'text': voice_info.get('text', '')
                }
                files_info.append(file_info)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'files': files_info,
            'count': len(files_info)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/remove_user_voices', methods=['DELETE'])
def remove_user_voices():
    """Xóa tất cả voice của một user"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        user_id = (data.get('user_id') or '').strip()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403
        
        removed_count = voice_cloner.remove_user_voices(user_id)
        return jsonify({'success': True, 'message': f'Removed {removed_count} voices for user {user_id}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/<path:filepath>')
def serve_file(filepath):
    """Serve files from output directory"""
    try:
        # Chỉ cho phép truy cập file trong thư mục output
        if '..' in filepath or filepath.startswith('/'):
            return jsonify({'error': 'Invalid file path'}), 400
        
        file_path = os.path.join(OUTPUT_FOLDER, filepath)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Xác định MIME type dựa trên extension
        if filepath.endswith('.srt'):
            mimetype = 'text/plain'
        elif filepath.endswith('.wav'):
            mimetype = 'audio/wav'
        elif filepath.endswith('.mp3'):
            mimetype = 'audio/mpeg'
        else:
            mimetype = 'application/octet-stream'
        
        return send_file(file_path, mimetype=mimetype)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_config', methods=['GET'])
def export_config():
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    try:
        config_path = os.path.join(OUTPUT_FOLDER, 'voice_config.json')
        voice_cloner.export_voice_config(config_path)
        return jsonify({'success': True, 'config_url': '/api/config/voice_config.json',
                        'message': 'Configuration exported successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/<filename>')
def get_config(filename):
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='application/json')
        return jsonify({'error': 'Config file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'voice_cloner_ready': voice_cloner is not None,
        'available_voices': len(voice_cloner.get_available_voices()) if voice_cloner else 0
    })

@app.route('/api/synthesize_with_srt', methods=['POST'])
def synthesize_with_srt():
    """Clone voice và tạo audio + file SRT phụ đề"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text')
        user_id = data.get('user_id')
        voice_id = data.get('voice_id')
        language = data.get('language')
        segment_duration = data.get('segment_duration', 3.0)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Nếu không có voice_id, sử dụng voice đầu tiên của user
        if not voice_id:
            user_voices = voice_cloner.get_voices_by_user(user_id)
            if not user_voices:
                return jsonify({'error': f'No voices found for user {user_id}'}), 404
            voice_id = user_voices[0]
        else:
            # Kiểm tra quyền sở hữu voice
            user_voices = voice_cloner.get_voices_by_user(user_id)
            if voice_id not in user_voices:
                return jsonify({'error': f'Voice {voice_id} does not belong to user {user_id}'}), 403
        
        print(f"🎯 Starting voice cloning with SRT for user {user_id}")
        print(f"📝 Text: {text[:100]}...")
        print(f"🎤 Voice ID: {voice_id}")
        print(f"🌍 Language: {language or 'auto'}")
        print(f"⏱️ Segment duration: {segment_duration}s")
        
        # Tạo output paths
        timestamp = int(time.time())
        output_filename = f"voice_{user_id}_{timestamp}"
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        audio_path = output_dir / f"{output_filename}.wav"
        srt_path = output_dir / f"{output_filename}.srt"
        
        # Clone voice với SRT
        result_audio, result_srt = voice_cloner.clone_voice_with_srt(
            text=text,
            voice_id=voice_id,
            output_path=str(audio_path),
            language=language,
            srt_path=str(srt_path),
            segment_duration=segment_duration
        )
        
        # Lấy thông tin audio
        audio_info = voice_cloner.get_voice_info(voice_id)
        
        return jsonify({
            'success': True,
            'message': 'Voice cloned successfully with SRT',
            'audio_path': result_audio,
            'srt_path': result_srt,
            'voice_id': voice_id,
            'user_id': user_id,
            'text': text,
            'segment_duration': segment_duration,
            'audio_info': audio_info
        })
        
    except Exception as e:
        print(f"❌ Error in synthesize_with_srt: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/create_srt', methods=['POST'])
def create_srt():
    """Tạo file SRT từ audio file có sẵn"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    
    try:
        data = request.get_json()
        audio_path = data.get('audio_path')
        text = data.get('text')
        srt_path = data.get('srt_path')
        segment_duration = data.get('segment_duration', 3.0)
        
        if not audio_path:
            return jsonify({'error': 'Audio path is required'}), 400
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        print(f"📝 Creating SRT file for audio: {audio_path}")
        print(f"📝 Text: {text[:100]}...")
        print(f"⏱️ Segment duration: {segment_duration}s")
        
        # Tạo file SRT
        result_srt = voice_cloner.create_srt_from_audio(
            audio_path=audio_path,
            text=text,
            srt_path=srt_path,
            segment_duration=segment_duration
        )
        
        return jsonify({
            'success': True,
            'message': 'SRT file created successfully',
            'srt_path': result_srt,
            'audio_path': audio_path,
            'text': text,
            'segment_duration': segment_duration
        })
        
    except Exception as e:
        print(f"❌ Error creating SRT: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/merge_srt', methods=['POST'])
def merge_srt():
    """Gộp nhiều file SRT thành một file duy nhất"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    
    try:
        data = request.get_json()
        srt_files = data.get('srt_files')
        output_path = data.get('output_path')
        segment_duration = data.get('segment_duration', 3.0)
        
        if not srt_files or not isinstance(srt_files, list):
            return jsonify({'error': 'srt_files must be a list'}), 400
        
        if not output_path:
            return jsonify({'error': 'Output path is required'}), 400
        
        print(f"🔗 Merging {len(srt_files)} SRT files")
        print(f"📁 Output: {output_path}")
        print(f"⏱️ Segment duration: {segment_duration}s")
        
        # Gộp file SRT
        result_path = voice_cloner.merge_srt_files(
            srt_files=srt_files,
            output_path=output_path,
            segment_duration=segment_duration
        )
        
        return jsonify({
            'success': True,
            'message': 'SRT files merged successfully',
            'output_path': result_path,
            'input_files': srt_files,
            'segment_duration': segment_duration
        })
        
    except Exception as e:
        print(f"❌ Error merging SRT files: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Dev server (không dùng trong PM2/Gunicorn)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True, threaded=True)
