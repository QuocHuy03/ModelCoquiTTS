#!/usr/bin/env python3
"""
Voice Cloning Web Interface
Flask app để test voice cloning model
"""

import os
import json
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from voice_cloner import VoiceCloner
import uuid


app = Flask(__name__, template_folder='web/templates')
CORS(app)

# Global voice cloner instance
voice_cloner = None

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
USERS_FILE = 'users.json'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# In-memory users cache
USERS = {}


def load_users():
    """Load predefined users from users.json"""
    global USERS
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # support {"users": [{"user_id": "...", "name": "..."}, ...]} or {"028": {...}}
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


def get_request_user_id():
    """Extract user_id from request (header, args, form, or json)."""
    # Header
    user_id = request.headers.get('X-User-Id')
    if user_id:
        return user_id.strip()
    # Query args
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


@app.route('/api/login', methods=['POST'])
def login():
    """Simple login using predefined users.json. Expects JSON {"user_id": "..."}."""
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
        return jsonify({
            'success': True,
            'user': user_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_voice_cloner():
    """Initialize voice cloner"""
    global voice_cloner
    
    try:
        # Try to load custom model config if exists
        config_files = list(Path('models').glob('*_config.json'))
        if config_files:
            config_path = str(config_files[0])
            voice_cloner = VoiceCloner(model_path=config_path)
            print(f"✅ Loaded custom model from: {config_path}")
        else:
            # Load default XTTS model
            voice_cloner = VoiceCloner()
            print("✅ Loaded default XTTS model")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing voice cloner: {e}")
        return False


@app.route('/')
def index():
    """Login page"""
    return render_template('login.html')


@app.route('/app')
def app_page():
    """Main clone voice app page"""
    return render_template('index.html')


@app.route('/api/voices', methods=['GET'])
def get_voices():
    """Get available voices for current user (if provided)"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    
    try:
        user_id = get_request_user_id()
        voices = voice_cloner.get_available_voices()
        voice_info = {}
        
        for voice_id in voices:
            # If user_id provided, only expose that user's voice
            if user_id:
                if voice_id != get_user_voice_id(user_id):
                    continue
            info = voice_cloner.get_voice_info(voice_id)
            if info:
                voice_info[voice_id] = info
        
        return jsonify({
            'voices': voice_info,
            'count': len(voices)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_voice', methods=['POST'])
def upload_voice():
    """Upload voice sample"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    
    try:
        # Require user_id
        user_id = get_request_user_id()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403

        # Check if file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get voice ID and text
        voice_id = request.form.get('voice_id', '').strip()
        text = request.form.get('text', '').strip()
        
        # Force single voice per user
        voice_id = get_user_voice_id(user_id)
        
        if not text:
            text = "Voice sample"
        
        # Save uploaded file
        user_dir = os.path.join(UPLOAD_FOLDER, user_id)
        os.makedirs(user_dir, exist_ok=True)
        filename = f"{file.filename}"
        file_path = os.path.join(user_dir, filename)
        file.save(file_path)
        
        # Add voice sample
        voice_cloner.add_voice_sample(voice_id, file_path, text)
        
        return jsonify({
            'success': True,
            'voice_id': voice_id,
            'message': f'Voice sample added successfully: {voice_id}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    """Synthesize speech with voice cloning"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_id = data.get('user_id', '').strip()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403

        text = data.get('text', '').strip()
        voice_id = get_user_voice_id(user_id)
        language = data.get('language', '').strip()  # Optional language parameter
        voice_type = data.get('voice_type', 'normal')  # Voice type effects
        age_group = data.get('age_group', 'adult')  # Age group effects
        speed = data.get('speed', 1.0)  # Speed effect
        pitch_shift = data.get('pitch_shift', 0)  # Pitch effect
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Check if voice exists
        if voice_id not in voice_cloner.get_available_voices():
            return jsonify({'error': f'Voice ID "{voice_id}" not found'}), 404
        
        # Generate unique output filename
        user_out_dir = os.path.join(OUTPUT_FOLDER, user_id)
        os.makedirs(user_out_dir, exist_ok=True)
        output_filename = f"output_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(user_out_dir, output_filename)
        
        # Clone voice với advanced effects
        if any([voice_type != 'normal', age_group != 'adult', speed != 1.0, pitch_shift != 0]):
            # Sử dụng method với advanced effects
            result_path = voice_cloner.clone_voice_with_advanced_effects(
                text, voice_id, output_path, 
                language=language if language else None,
                speed=speed, pitch_shift=pitch_shift,
                voice_type=voice_type, age_group=age_group
            )
        else:
            # Sử dụng method cơ bản
            if language:
                result_path = voice_cloner.clone_voice(text, voice_id, output_path, language)
            else:
                result_path = voice_cloner.clone_voice(text, voice_id, output_path)
        
        if os.path.exists(result_path):
            return jsonify({
                'success': True,
                'audio_url': f'/api/audio/{user_id}/{output_filename}',
                'message': 'Voice cloning completed successfully'
            })
        else:
            return jsonify({'error': 'Failed to generate audio'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/audio/<user_id>/<filename>')
def get_audio(user_id, filename):
    """Serve audio files"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, user_id, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='audio/wav')
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/transform_voice', methods=['POST'])
def transform_voice():
    """Transform voice with special effects"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_id = data.get('user_id', '').strip()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403

        voice_id = get_user_voice_id(user_id)
        transformation_type = data.get('transformation_type', '').strip()
        intensity = data.get('intensity', 0.5)
        
        if not voice_id:
            return jsonify({'error': 'Voice ID is required'}), 400
        
        if not transformation_type:
            return jsonify({'error': 'Transformation type is required'}), 400
        
        # Check if voice exists
        if voice_id not in voice_cloner.get_available_voices():
            return jsonify({'error': f'Voice ID "{voice_id}" not found'}), 404
        
        # Generate unique output filename
        user_out_dir = os.path.join(OUTPUT_FOLDER, user_id)
        os.makedirs(user_out_dir, exist_ok=True)
        output_filename = f"transform_{transformation_type}_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(user_out_dir, output_filename)
        
        # Transform voice
        result_path = voice_cloner.transform_voice(
            voice_id, transformation_type, intensity, output_path
        )
        
        if os.path.exists(result_path):
            return jsonify({
                'success': True,
                'audio_url': f'/api/audio/{user_id}/{output_filename}',
                'message': f'Voice transformation completed successfully: {transformation_type}'
            })
        else:
            return jsonify({'error': 'Failed to transform voice'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/assess_quality/<voice_id>', methods=['GET'])
def assess_quality(voice_id):
    """Assess voice quality"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    
    try:
        # Optional user scoping: only allow if matches user's voice
        user_id = get_request_user_id()
        if user_id:
            expected_voice = get_user_voice_id(user_id)
            if voice_id != expected_voice:
                return jsonify({'error': 'Forbidden for this user_id'}), 403
        # Check if voice exists
        if voice_id not in voice_cloner.get_available_voices():
            return jsonify({'error': f'Voice ID "{voice_id}" not found'}), 404
        
        # Assess voice quality
        quality_results = voice_cloner.assess_voice_quality(voice_id)
        
        # Format results for display
        formatted_results = []
        for key, value in quality_results.items():
            if key == 'overall_score':
                formatted_results.append({
                    'name': 'Overall Score',
                    'value': f"{value:.1f}/100",
                    'description': f'Overall quality score: {value:.1f} out of 100'
                })
            elif key == 'quality_level':
                formatted_results.append({
                    'name': 'Quality Level',
                    'value': value,
                    'description': f'Quality assessment: {value}'
                })
            elif key == 'duration':
                formatted_results.append({
                    'name': 'Duration',
                    'value': f"{value:.1f}s",
                    'description': f'Audio duration: {value:.1f} seconds'
                })
            elif key == 'sample_rate':
                formatted_results.append({
                    'name': 'Sample Rate',
                    'value': f"{value} Hz",
                    'description': f'Audio sample rate: {value} Hz'
                })
            elif key == 'rms_energy':
                formatted_results.append({
                    'name': 'RMS Energy',
                    'value': f"{value:.3f}",
                    'description': f'Root Mean Square energy: {value:.3f}'
                })
            elif key == 'peak_amplitude':
                formatted_results.append({
                    'name': 'Peak Amplitude',
                    'value': f"{value:.3f}",
                    'description': f'Peak audio amplitude: {value:.3f}'
                })
            elif key == 'spectral_centroids':
                formatted_results.append({
                    'name': 'Spectral Centroids',
                    'value': f"{value:.1f} Hz",
                    'description': f'Average spectral centroid: {value:.1f} Hz'
                })
            elif key == 'mfcc_variance':
                formatted_results.append({
                    'name': 'MFCC Variance',
                    'value': f"{value:.3f}",
                    'description': f'Mel-frequency cepstral coefficient variance: {value:.3f}'
                })
            elif key == 'snr':
                formatted_results.append({
                    'name': 'Signal-to-Noise Ratio',
                    'value': f"{value:.1f} dB",
                    'description': f'Signal-to-noise ratio: {value:.1f} dB'
                })
        
        return jsonify({
            'success': True,
            'message': f'Quality assessment completed for voice "{voice_id}"',
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/remove_voice', methods=['DELETE'])
def remove_voice():
    """Remove voice sample"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_id = data.get('user_id', '').strip()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        if USERS and user_id not in USERS:
            return jsonify({'error': 'Invalid user_id'}), 403

        voice_id = get_user_voice_id(user_id)
        
        voice_cloner.remove_voice(voice_id)
        
        return jsonify({
            'success': True,
            'message': f'Voice "{voice_id}" removed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export_config', methods=['GET'])
def export_config():
    """Export voice configuration"""
    if not voice_cloner:
        return jsonify({'error': 'Voice cloner not initialized'}), 500
    
    try:
        config_path = os.path.join(OUTPUT_FOLDER, 'voice_config.json')
        voice_cloner.export_voice_config(config_path)
        
        return jsonify({
            'success': True,
            'config_url': '/api/config/voice_config.json',
            'message': 'Configuration exported successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/<filename>')
def get_config(filename):
    """Serve configuration files"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='application/json')
        else:
            return jsonify({'error': 'Config file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'voice_cloner_ready': voice_cloner is not None,
        'available_voices': len(voice_cloner.get_available_voices()) if voice_cloner else 0
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🎵 Starting Voice Cloning Web Interface...")
    
    # Load users
    load_users()

    # Initialize voice cloner
    if init_voice_cloner():
        print("✅ Voice cloner initialized successfully")
    else:
        print("❌ Failed to initialize voice cloner")
        print("⚠️ Some features may not work properly")
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    ) 