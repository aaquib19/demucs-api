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
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE', 100 * 1024 * 1024))

# Global model variable
model = None
device = None
model_sample_rate = None

# Store processing results
results_dir = tempfile.mkdtemp()
processing_jobs = {}

# Configuration
JOB_RETENTION_HOURS = int(os.getenv('JOB_RETENTION_HOURS', 24))
CLEANUP_INTERVAL_SECONDS = int(os.getenv('CLEANUP_INTERVAL_SECONDS', 3600))
USE_GPU = os.getenv('USE_GPU', 'auto').lower()
MODEL_NAME = os.getenv('MODEL_NAME', 'htdemucs')
AUDIO_SAMPLE_RATE = int(os.getenv('AUDIO_SAMPLE_RATE', 44100))

def get_device():
    """Determine best available device"""
    if USE_GPU == 'false':
        return 'cpu'
    elif USE_GPU == 'true':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            print("Warning: GPU requested but not available, falling back to CPU")
            return 'cpu'
    else:  # auto
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

def load_model():
    """Lazy load model on first request"""
    global model, device, model_sample_rate
    if model is None:
        try:
            print(f"Loading model '{MODEL_NAME}'...")
            device = get_device()
            print(f"Using device: {device}")

            model = get_model(MODEL_NAME)
            model = model.to(device)
            model.eval()  # Set to evaluation mode

            # Get model's native sample rate
            model_sample_rate = model.samplerate
            print(f"✓ Model loaded on {device} (sample rate: {model_sample_rate}Hz)")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            traceback.print_exc()
            return False
    return True

def cleanup_old_jobs():
    """Clean up old completed/failed jobs and their files"""
    while True:
        try:
            time.sleep(CLEANUP_INTERVAL_SECONDS)
            cutoff_time = datetime.now() - timedelta(hours=JOB_RETENTION_HOURS)

            jobs_to_remove = []
            for job_id, job_info in processing_jobs.items():
                job_time = job_info.get('created_at')
                if job_time and datetime.fromisoformat(job_time) < cutoff_time:
                    # Remove zip file if it exists
                    zip_path = os.path.join(results_dir, f'{job_id}.zip')
                    if os.path.exists(zip_path):
                        os.unlink(zip_path)
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del processing_jobs[job_id]
                print(f"Cleaned up old job: {job_id}")

        except Exception as e:
            print(f"Cleanup error: {e}")

def process_audio(job_id, temp_input, instruments):
    """Background task to process audio"""
    try:
        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['progress'] = 10
        
        # Verify file exists and is accessible
        if not os.path.exists(temp_input):
            raise FileNotFoundError(f"Temporary file not found: {temp_input}")
        
        if not os.path.isfile(temp_input):
            raise ValueError(f"Path is not a regular file: {temp_input}")

        # Load audio using librosa for better format compatibility
        print(f"[{job_id}] Loading audio...")
        use_sr = model_sample_rate if model_sample_rate else AUDIO_SAMPLE_RATE
        
        # Use librosa for loading which has better format support
        try:
            audio, sr = librosa.load(temp_input, sr=use_sr, mono=False)
            # Convert to tensor
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)  # Mono to (1, samples)
            else:
                audio = audio  # Already (channels, samples)
            waveform = torch.from_numpy(audio).float()
        except Exception as e:
            print(f"[{job_id}] Librosa failed, trying torchaudio: {e}")
            # Fallback to torchaudio
            waveform, sr = torchaudio.load(temp_input)
            
            # Resample if needed
            if sr != use_sr:
                print(f"[{job_id}] Resampling from {sr}Hz to {use_sr}Hz...")
                resampler = torchaudio.transforms.Resample(sr, use_sr)
                waveform = resampler(waveform)
                sr = use_sr

        processing_jobs[job_id]['progress'] = 30

        # Move to device (avoid unnecessary CPU->GPU copies)
        if device != 'cpu':
            waveform = waveform.to(device, non_blocking=True)
        else:
            waveform = waveform.to(device)

        # Separate with optimizations
        print(f"[{job_id}] Separating audio (device: {device})...")
        with torch.no_grad(), torch.inference_mode():
            sources = apply_model(model, waveform[None], device=device)

        processing_jobs[job_id]['progress'] = 70

        # Save stems with optimized file handling
        model_sources = ['drums', 'bass', 'other', 'vocals']
        zip_path = os.path.join(results_dir, f'{job_id}.zip')

        print(f"[{job_id}] Saving {len(instruments)} stems...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
            for idx, source_name in enumerate(model_sources):
                if source_name in instruments:
                    # Move to CPU only when needed and convert efficiently
                    audio_np = sources[0, idx].cpu().numpy()

                    # Transpose if stereo for correct format
                    if audio_np.ndim == 2:
                        audio_np = audio_np.T

                    # Write directly to zip buffer (avoid temp file)
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_np, sr, format='WAV')
                    zip_file.writestr(f'{source_name}.wav', buffer.getvalue())
                    print(f"[{job_id}] Added {source_name}")

        processing_jobs[job_id]['status'] = 'completed'
        processing_jobs[job_id]['progress'] = 100
        processing_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        print(f"[{job_id}] ✓ Complete")

    except Exception as e:
        print(f"[{job_id}] ✗ Error: {e}")
        traceback.print_exc()
        processing_jobs[job_id]['status'] = 'failed'
        processing_jobs[job_id]['error'] = str(e)
    finally:
        # Clean up the temp file
        if os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
            except Exception as e:
                print(f"Error deleting temp file: {e}")
        
        # Clear GPU cache if using CUDA
        if device == 'cuda':
            torch.cuda.empty_cache()

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

        # Create a more reliable temporary file
        file_ext = os.path.splitext(file.filename)[1] or '.mp3'
        temp_dir = tempfile.mkdtemp()
        temp_input = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
        
        # Save the file with proper error handling
        try:
            file.save(temp_input)
            # Verify the file was saved correctly
            if not os.path.exists(temp_input) or os.path.getsize(temp_input) == 0:
                raise ValueError("Failed to save uploaded file")
        except Exception as e:
            # Clean up if save failed
            if os.path.exists(temp_input):
                os.unlink(temp_input)
            os.rmdir(temp_dir)
            return jsonify({'error': f'Failed to save uploaded file: {str(e)}'}), 500

        # Create job
        job_id = str(uuid.uuid4())
        processing_jobs[job_id] = {
            'status': 'queued',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'instruments': instruments,
            'temp_dir': temp_dir  # Store temp dir for cleanup
        }

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

@app.route('/batch-separate', methods=['POST'])
def batch_separate():
    """
    Start batch audio separation for multiple files
    Returns list of job IDs for polling
    """
    try:
        if not load_model():
            return jsonify({'error': 'Model failed to load'}), 503

        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400

        max_batch_size = int(os.getenv('MAX_BATCH_SIZE', 10))
        if len(files) > max_batch_size:
            return jsonify({'error': f'Maximum {max_batch_size} files allowed per batch'}), 400

        instruments = request.form.get('instruments', 'vocals,drums,bass,other').split(',')
        instruments = [i.strip().lower() for i in instruments]

        valid_instruments = {'vocals', 'drums', 'bass', 'other'}
        instruments = [i for i in instruments if i in valid_instruments]

        if not instruments:
            return jsonify({'error': 'No valid instruments requested'}), 400

        job_ids = []
        for file in files:
            if file and file.filename != '':
                # Create a more reliable temporary file
                file_ext = os.path.splitext(file.filename)[1] or '.mp3'
                temp_dir = tempfile.mkdtemp()
                temp_input = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
                
                # Save the file with proper error handling
                try:
                    file.save(temp_input)
                    # Verify the file was saved correctly
                    if not os.path.exists(temp_input) or os.path.getsize(temp_input) == 0:
                        raise ValueError("Failed to save uploaded file")
                except Exception as e:
                    # Clean up if save failed
                    if os.path.exists(temp_input):
                        os.unlink(temp_input)
                    os.rmdir(temp_dir)
                    return jsonify({'error': f'Failed to save uploaded file: {str(e)}'}), 500

                # Create job
                job_id = str(uuid.uuid4())
                processing_jobs[job_id] = {
                    'status': 'queued',
                    'progress': 0,
                    'created_at': datetime.now().isoformat(),
                    'instruments': instruments,
                    'filename': file.filename,
                    'temp_dir': temp_dir  # Store temp dir for cleanup
                }

                # Start background thread
                thread = threading.Thread(
                    target=process_audio,
                    args=(job_id, temp_input, instruments),
                    daemon=True
                )
                thread.start()

                job_ids.append({
                    'job_id': job_id,
                    'filename': file.filename,
                    'status': 'queued'
                })

        return jsonify({
            'jobs': job_ids,
            'message': f'Batch processing started for {len(job_ids)} files'
        }), 202

    except Exception as e:
        print(f"Batch error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_jobs, daemon=True)
    cleanup_thread.start()
    print(f"Cleanup thread started (retention: {JOB_RETENTION_HOURS}h)")

    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    print(f"Configuration: GPU={USE_GPU}, Model={MODEL_NAME}, Sample Rate={AUDIO_SAMPLE_RATE}Hz")
    app.run(host='0.0.0.0', port=port, debug=False)