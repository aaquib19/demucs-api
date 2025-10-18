from flask import Flask, request, jsonify, send_file
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import torch
import os
import tempfile
import io
import zipfile
import librosa
import numpy as np
import soundfile as sf
import traceback
import threading
import uuid
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Global model variable
model = None
device = None

# Store processing results
results_dir = tempfile.mkdtemp()
processing_jobs = {}

def load_model():
    """Lazy load model on first request"""
    global model, device
    if model is None:
        try:
            print("Loading model...")
            model = get_model('hdemucs_mmi')
            device = 'cpu'
            model = model.to(device)
            print(f"✓ Model loaded on {device}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            traceback.print_exc()
            return False
    return True

def process_audio(job_id, temp_input, instruments):
    """Background task to process audio"""
    try:
        processing_jobs[job_id] = {'status': 'processing', 'progress': 0}
        
        # Load audio
        print(f"[{job_id}] Loading audio...")
        audio, sr = librosa.load(temp_input, sr=16000, mono=False)
        
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        waveform = torch.from_numpy(audio).float()
        waveform = waveform.to(device)
        
        # Separate
        print(f"[{job_id}] Separating audio...")
        with torch.no_grad():
            sources = apply_model(model, waveform[None])
        
        # Save stems
        model_sources = ['vocals', 'drums', 'bass', 'other']
        zip_path = os.path.join(results_dir, f'{job_id}.zip')
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for idx, source_name in enumerate(model_sources):
                if source_name in instruments:
                    audio_np = sources[0, idx:idx+1].cpu().numpy().squeeze()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_tmp:
                        sf.write(wav_tmp.name, audio_np, 16000)
                        with open(wav_tmp.name, 'rb') as f:
                            zip_file.writestr(f'{source_name}.wav', f.read())
                        os.unlink(wav_tmp.name)
                        print(f"[{job_id}] Added {source_name}")
        
        processing_jobs[job_id] = {'status': 'completed', 'progress': 100}
        print(f"[{job_id}] ✓ Complete")
        
    except Exception as e:
        print(f"[{job_id}] ✗ Error: {e}")
        traceback.print_exc()
        processing_jobs[job_id] = {'status': 'failed', 'error': str(e)}
    finally:
        if os.path.exists(temp_input):
            os.unlink(temp_input)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None}), 200

@app.route('/separate', methods=['POST'])
def separate():
    """
    Start background audio separation job
    Returns job ID for polling
    """
    try:
        if not load_model():
            return jsonify({'error': 'Model failed to load'}), 503
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'Invalid file'}), 400
        
        instruments = request.form.get('instruments', 'vocals,drums,bass,other').split(',')
        instruments = [i.strip().lower() for i in instruments]
        
        valid_instruments = {'vocals', 'drums', 'bass', 'other'}
        instruments = [i for i in instruments if i in valid_instruments]
        
        if not instruments:
            return jsonify({'error': 'No valid instruments requested'}), 400
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            file.save(tmp.name)
            temp_input = tmp.name
        
        # Create job
        job_id = str(uuid.uuid4())
        processing_jobs[job_id] = {'status': 'queued', 'progress': 0}
        
        # Start background thread
        thread = threading.Thread(
            target=process_audio,
            args=(job_id, temp_input, instruments),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Processing started. Poll /status/<job_id> to check progress'
        }), 202
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    """Check job status"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_jobs[job_id]), 200

@app.route('/download/<job_id>', methods=['GET'])
def download(job_id):
    """Download completed job"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': f"Job status: {job['status']}"}), 400
    
    zip_path = os.path.join(results_dir, f'{job_id}.zip')
    if not os.path.exists(zip_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(
        zip_path,
        mimetype='application/zip',
        as_attachment=True,
        download_name='separated_stems.zip'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)