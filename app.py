from flask import Flask, request, jsonify, send_file
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import torch
import os
import tempfile
import io
import zipfile
from werkzeug.utils import secure_filename

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
            model = get_model('htdemucs')
            device = 'cpu'  # Force CPU for Render compatibility
            model = model.to(device)
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
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
            
            # Load audio
            print(f"Loading audio from {temp_input}")
            waveform, sr = torchaudio.load(temp_input)
            
            # Resample if needed
            if sr != 16000:
                print(f"Resampling from {sr} to 16000")
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
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
                        wav_buffer = io.BytesIO()
                        torchaudio.save(wav_buffer, sources[0, idx:idx+1], 16000, format='wav')
                        wav_buffer.seek(0)
                        zip_file.writestr(f'{source_name}.wav', wav_buffer.read())
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
        print(f"Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

**Key changes:**

1. **Lazy loading** - Model loads on first request, not at startup (prevents timeout)
2. **Forced CPU** - `device = 'cpu'` explicitly, avoiding GPU detection issues
3. **Better error handling** - Added logging and proper cleanup
4. **In-memory processing** - Avoid multiple disk writes for better performance
5. **Input validation** - Validate instruments early
6. **Reduced file size limit** - 50MB is safer for Render's constraints
7. **Better logging** - Added print statements for debugging

**Also update your requirements.txt to CPU-only:**
```
Flask==3.0.0
torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
torchaudio==2.9.0
demucs==4.0.0
numpy==2.0.0
gunicorn==22.0.0
python-dotenv==1.0.0