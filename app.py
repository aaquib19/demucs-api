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
import traceback
import soundfile as sf


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Global model variable - lazy load
model = None
device = None

def load_model():
    """Lazy load model on first request"""
    global model, device
    if model is None:
        try:
            print("Loading model...")
            model = get_model('hdemucs_mmi')
            device = 'cpu'  # Force CPU for Render compatibility
            model = model.to(device)
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None}), 200

@app.route('/separate', methods=['POST'])
def separate():
    """
    Expects multipart form data with:
    - file: audio file (mp3, wav, etc.)
    - instruments: comma-separated list (vocals,drums,bass,other)
    """
    try:
        # Load model on first request
        load_model()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'Invalid file'}), 400
        
        instruments = request.form.get('instruments', 'vocals,drums,bass,other').split(',')
        instruments = [i.strip().lower() for i in instruments]
        
        # Validate requested instruments
        valid_instruments = {'vocals', 'drums', 'bass', 'other'}
        instruments = [i for i in instruments if i in valid_instruments]
        
        if not instruments:
            return jsonify({'error': 'No valid instruments requested'}), 400
        
        # Save uploaded file temporarily
        temp_input = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                file.save(tmp.name)
                temp_input = tmp.name
            
            # Load audio using librosa
            print(f"Loading audio from {temp_input}")
            audio, sr = librosa.load(temp_input, sr=16000, mono=False)
            
            # Convert to torch tensor
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=0)
            waveform = torch.from_numpy(audio).float()
            
            # Move to device
            waveform = waveform.to(device)
            
            # Apply model
            print("Separating audio...")
            with torch.no_grad():
                sources = apply_model(model, waveform[None])
            
            # Get model sources
            model_sources = ['vocals', 'drums', 'bass', 'other']
            
            # Create zip file with requested stems
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for idx, source_name in enumerate(model_sources):
                    if source_name in instruments:
                        # Save to buffer instead of disk
                        audio_np = sources[0, idx:idx+1].cpu().numpy()
                            # Remove the extra dimension and transpose properly
                        audio_np = audio_np.squeeze()
                        if audio_np.ndim == 1:
                            audio_np = audio_np
                        else:
                            audio_np = audio_np.T
                            
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_tmp:
                            sf.write(wav_tmp.name, audio_np, 16000)
                            with open(wav_tmp.name, 'rb') as f:
                                zip_file.writestr(f'{source_name}.wav', f.read())
                            os.unlink(wav_tmp.name)
                            print(f"Added {source_name}")
                            print(f"Added {source_name}")
            
            zip_buffer.seek(0)
            print("Returning zip file")
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name='separated_stems.zip'
            )
        
        finally:
            if temp_input and os.path.exists(temp_input):
                os.unlink(temp_input)
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)