from flask import Flask, request, jsonify, send_file
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import torch
import os
import tempfile
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Load model on startup
try:
    model = get_model('htdemucs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/separate', methods=['POST'])
def separate():
    """
    Expects multipart form data with:
    - file: audio file (mp3, wav, etc.)
    - instruments: comma-separated list (vocals,drums,bass,other)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        instruments = request.form.get('instruments', 'vocals,drums,bass,other').split(',')
        instruments = [i.strip() for i in instruments]
        
        if not file or file.filename == '':
            return jsonify({'error': 'Invalid file'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            file.save(tmp.name)
            temp_input = tmp.name
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(temp_input)
            
            # Resample if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Move to device
            waveform = waveform.to(model.device)
            
            # Apply model
            with torch.no_grad():
                sources = apply_model(model, waveform[None])
            
            # Get model sources (vocals, drums, bass, other)
            model_sources = ['vocals', 'drums', 'bass', 'other']
            
            # Save requested instruments
            result_files = {}
            with tempfile.TemporaryDirectory() as tmpdir:
                for idx, source_name in enumerate(model_sources):
                    if source_name in instruments:
                        output_path = os.path.join(tmpdir, f'{source_name}.wav')
                        torchaudio.save(output_path, sources[0, idx:idx+1], 16000)
                        
                        with open(output_path, 'rb') as f:
                            result_files[source_name] = io.BytesIO(f.read())
                
                # Create zip file with all stems
                import zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for instrument, file_data in result_files.items():
                        file_data.seek(0)
                        zip_file.writestr(f'{instrument}.wav', file_data.read())
                
                zip_buffer.seek(0)
                return send_file(
                    zip_buffer,
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name='separated_stems.zip'
                )
        
        finally:
            os.unlink(temp_input)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)