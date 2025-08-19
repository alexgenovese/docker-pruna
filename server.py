#!/usr/bin/env python3
"""
Flask server API for inference with Flux Krea models.
Exposes REST endpoints for image generation using models compiled with Pruna.
"""


import os
import sys
import shutil
import torch
import traceback
from pathlib import Path
from flask import send_from_directory
from typing import Dict, Any, Optional
from lib.utils import load_model, get_best_available_device, validate_configuration, image_to_base64
from lib.const import DEFAULT_CONFIG, MODEL_CACHE

import importlib.util
spec = importlib.util.spec_from_file_location("download_model_and_compile", os.path.join(os.path.dirname(__file__), "download_model_and_compile.py"))
if spec and spec.loader:
    dl_mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(dl_mod)
    except Exception as e:
        # Don't fail import of server if the download/compile helper has heavy deps
        print(f"⚠️  Warning: failed to load download_model_and_compile.py: {e}")
        dl_mod = None
else:
    dl_mod = None
from flask import Flask, request, jsonify, url_for
import threading
import uuid
import time

# Flask application configuration
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# In-memory download task registry (simple, non-persistent)
DOWNLOAD_TASKS = {}
DOWNLOAD_TASKS_LOCK = threading.Lock()

#
# Update the status of a download task
#
def _update_task(task_id, **kwargs):
    with DOWNLOAD_TASKS_LOCK:
        if task_id in DOWNLOAD_TASKS:
            DOWNLOAD_TASKS[task_id].update(kwargs)


#
# Background worker for downloading models
#
# This runs in a separate thread to avoid blocking the main Flask thread
# and to handle long-running downloads without HTTP timeouts.
def _download_worker(task_id, model_id, download_dir, hf_token):
    _update_task(task_id, status='running', started_at=time.time(), message='Starting download')
    try:
        path = dl_mod.download_model(model_id, download_dir, hf_token)
        _update_task(task_id, status='finished', finished_at=time.time(), result=path, message='Download completed')
    except Exception as e:
        _update_task(task_id, status='error', finished_at=time.time(), error=str(e), message='Download failed')


#
# Serve generated static files (images)
#
@app.route('/generated_images/<path:filename>')
def serve_generated_image(filename):
    return send_from_directory('generated_images', filename)

#
# Endpoint to delete a downloaded or compiled model
#
@app.route('/delete-model', methods=['POST'])
def delete_model():
    """Delete the downloaded or compiled model folder given a model_id and a type ('downloaded' or 'compiled')."""
    try:
        data = request.get_json() or {}
        model_id = data.get('model_id') or data.get('modelId')
        model_type = data.get('type', 'all')  # 'downloaded', 'compiled', 'all'
        if not model_id:
            return jsonify({'status': 'error', 'message': 'Parameter "model_id" required'}), 400

        # derive paths
        model_name = model_id.replace('/', '--')
        download_path = os.path.join(DEFAULT_CONFIG['download_dir'], model_name)
        compiled_path = os.path.join(DEFAULT_CONFIG['compiled_dir'], model_name)
        deleted = []
        errors = []

        if model_type in ('downloaded', 'all'):
            if os.path.exists(download_path):
                try:
                    shutil.rmtree(download_path)
                    deleted.append({'type': 'downloaded', 'path': download_path})
                except Exception as e:
                    errors.append({'type': 'downloaded', 'path': download_path, 'error': str(e)})
        if model_type in ('compiled', 'all'):
            if os.path.exists(compiled_path):
                try:
                    shutil.rmtree(compiled_path)
                    deleted.append({'type': 'compiled', 'path': compiled_path})
                except Exception as e:
                    errors.append({'type': 'compiled', 'path': compiled_path, 'error': str(e)})

        if not deleted and not errors:
            return jsonify({'status': 'not_found', 'message': 'No folder found for the specified model_id.'}), 404

        return jsonify({'status': 'success' if not errors else 'partial', 'deleted': deleted, 'errors': errors})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Internal error: {str(e)}'}), 500

#
# Endpoint to download a model (download only, no compilation)
#
@app.route('/download', methods=['POST'])
def download_model():
    """Download a HuggingFace model into the download directory."""
    data = request.get_json() or {}
    model_id = data.get('model_id') or data.get('modelId')
    if not model_id:
        return jsonify({'status': 'error', 'message': 'Parameter "model_id" required'}), 400

    # Ensure dynamic module exposes download function
    if dl_mod is None or not hasattr(dl_mod, 'download_model'):
        return jsonify({'status': 'error', 'message': 'download_model function not available'}), 500

    download_dir = DEFAULT_CONFIG['download_dir']
    hf_token = DEFAULT_CONFIG['hf_token']

    # Create a background task for the download to avoid long HTTP timeouts (e.g., 524)
    task_id = str(uuid.uuid4())
    with DOWNLOAD_TASKS_LOCK:
        DOWNLOAD_TASKS[task_id] = {
            'task_id': task_id,
            'model_id': model_id,
            'status': 'queued',
            'created_at': time.time(),
            'message': 'Queued for download'
        }

    t = threading.Thread(target=_download_worker, args=(task_id, model_id, download_dir, hf_token), daemon=True)
    t.start()

    status_url = url_for('get_task', task_id=task_id, _external=True)
    return jsonify({'status': 'accepted', 'task_id': task_id, 'status_url': status_url}), 202

#
# Endpoint to get the status of a download task
#
@app.route('/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    """Return status for an asynchronous download task."""
    with DOWNLOAD_TASKS_LOCK:
        task = DOWNLOAD_TASKS.get(task_id)
    if not task:
        return jsonify({'status': 'error', 'message': 'Task not found'}), 404
    return jsonify(task)

#
# Endpoint to compile a model that has already been downloaded
#
@app.route('/compile', methods=['POST'])
def compile_model():
    """Compile a model already downloaded into the compiled_dir."""
    try:
        data = request.get_json() or {}
        model_id = data.get('model_id') or data.get('modelId')
        if not model_id:
            return jsonify({'status': 'error', 'message': 'Parameter "model_id" required'}), 400

        # Use the compile function from the dynamic module
        if not hasattr(dl_mod, 'compile_model'):
            return jsonify({'status': 'error', 'message': 'compile_model function not found in download_model_and_compile.py'}), 500

        download_dir = DEFAULT_CONFIG['download_dir']
        compiled_dir = DEFAULT_CONFIG['compiled_dir']
        try:
            dl_mod.compile_model(model_id, download_dir, compiled_dir)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error during compilation: {str(e)}'}), 500

        return jsonify({'status': 'success', 'message': f'Model {model_id} compiled to {compiled_dir}'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Internal error: {str(e)}'}), 500

#
# Ping endpoint
#
@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'timestamp': str(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
    })

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint that validates configuration"""
    try:
        status = validate_configuration()
        
        http_status = 200 if status['valid'] else 500
        
        return jsonify({
            'status': 'healthy' if status['valid'] else 'unhealthy',
            'configuration': status['config'],
            'system_info': status['system_info'],
            'errors': status['errors'],
            'warnings': status['warnings'],
            'timestamp': str(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        }), http_status
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Health check failed: {str(e)}',
            'timestamp': str(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        }), 500

# Image generation endpoint
@app.route('/generate', methods=['POST'])
def generate():
    """Image generation endpoint for Flux Krea"""
    try:
        # Parse request parameters
        data = request.get_json() or {}

        # Generation parameters
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', 'lowres, bad quality, low quality')
        model_id = data.get('model_id', DEFAULT_CONFIG['model_id'])
        num_inference_steps = data.get('num_inference_steps', 20)
        guidance_scale = data.get('guidance_scale', 7.5)
        width = data.get('width', 512)
        height = data.get('height', 512)
        seed = data.get('seed', None)
        num_images = data.get('num_images', 1)
        debug = data.get('debug', False)

        # Validate parameters
        if not prompt:
            return jsonify({
                'status': 'error',
                'message': 'Parameter "prompt" required'
            }), 400

        if num_images > 4:
            return jsonify({
                'status': 'error',
                'message': 'Maximum number of images: 4'
            }), 400

        # Load the model
        try:
            pipeline = load_model(model_id)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error loading model: {str(e)}'
            }), 500

        # Configure generator with seed
        generator = None
        if seed is not None:
            device = get_best_available_device(DEFAULT_CONFIG['force_cpu'], DEFAULT_CONFIG['device'])
            generator = torch.Generator(device=device).manual_seed(int(seed))

        # Prepare parameters for the pipeline
        generation_params = {
            'prompt': prompt,
            'num_inference_steps': num_inference_steps,
            'width': width,
            'height': height,
            'num_images_per_prompt': num_images,
            'generator': generator
        }

        # Add model-specific parameters
        model_type = MODEL_CACHE.get('model_type', 'generic')
        is_pruna_compiled = MODEL_CACHE.get('is_pruna_compiled', False)

        if is_pruna_compiled:
            # Parameters for models compiled with Pruna
            # Pruna keeps an interface similar to diffusers
            generation_params['guidance_scale'] = guidance_scale
            if negative_prompt:
                generation_params['negative_prompt'] = negative_prompt
        elif model_type in ['stable-diffusion', 'generic']:
            # Parameters for Stable Diffusion
            generation_params['guidance_scale'] = guidance_scale
            if negative_prompt:
                generation_params['negative_prompt'] = negative_prompt
        elif model_type == 'flux':
            # Parameters for FLUX (may have different params)
            generation_params['guidance_scale'] = guidance_scale
            if negative_prompt:
                generation_params['negative_prompt'] = negative_prompt

        print(f"Generating image with params: {generation_params}")

        # Generate images
        try:
            with torch.no_grad():
                result = pipeline(**generation_params)
                images = result.images
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error during generation: {str(e)}',
                'traceback': traceback.format_exc()
            }), 500

        # Convert images to base64
        images_b64 = []
        saved_files = []

        for i, image in enumerate(images):
            img_b64 = image_to_base64(image)
            images_b64.append({
                'index': i,
                'base64': img_b64,
                'format': 'png'
            })
            # Save image locally if debug = true
            if debug:
                try:
                    output_dir = "./generated_images"
                    os.makedirs(output_dir, exist_ok=True)
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"generated_{timestamp}_img_{i:02d}.png"
                    file_path = os.path.join(output_dir, filename)
                    image.save(file_path)
                    # Absolute URL for browser
                    relative_url = url_for('serve_generated_image', filename=filename)
                    absolute_url = request.host_url.rstrip('/') + relative_url
                    saved_files.append({
                        'index': i,
                        'filename': filename,
                        'url': absolute_url
                    })
                    print(f"💾 Debug: Image saved to {file_path}")
                except Exception as e:
                    print(f"⚠️  Error saving image {i}: {str(e)}")
                    saved_files.append({
                        'index': i,
                        'error': str(e)
                    })

        response_data = {
            'status': 'success',
            'message': f'Generated {len(images)} images',
            'parameters': {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'model_id': model_id,
                'model_type': model_type,
                'is_pruna_compiled': is_pruna_compiled,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'width': width,
                'height': height,
                'seed': seed,
                'num_images': num_images,
                'debug': debug
            },
            'images': images_b64,
            'timestamp': str(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        }

        # Add saved files info if debug = true
        if debug and saved_files:
            response_data['saved_files'] = saved_files

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

# 404 error handler
@app.errorhandler(404)
def not_found(error):
    """404 handler"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'available_endpoints': ['/ping', '/health', '/generate']
    }), 404

# 500 error handler
@app.errorhandler(500)
def internal_error(error):
    """Internal error handler"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'details': str(error)
    }), 500

# Funzione principale per avviare il server
def main():
    """Funzione principale per avviare il server"""
    import argparse

    parser = argparse.ArgumentParser(description='Flask Server API')
    parser.add_argument('--host', default='0.0.0.0', help='Host su cui avviare il server')
    parser.add_argument('--port', type=int, default=8000, help='Porta su cui avviare il server')
    parser.add_argument('--debug', action='store_true', help='Attiva modalità debug')
    parser.add_argument('--preload-model', help='Pre-carica un modello all\'avvio')
    
    args = parser.parse_args()

    print("🚀 Starting Flask Server...")
    print("=" * 50)
    
    # Stampa configurazione
    status = validate_configuration()
    print("📋 Configuration:")
    for key, value in status['config'].items():
        if 'token' in key.lower() and value:
            value = '***SET***'
        print(f"   - {key}: {value}")
    
    print(f"\n💻 System:")
    for key, value in status['system_info'].items():
        print(f"   - {key}: {value}")
    
    if status['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in status['warnings']:
            print(f"   - {warning}")
    
    if status['errors']:
        print(f"\n❌ Errors:")
        for error in status['errors']:
            print(f"   - {error}")
        print("\n🛑 Unable to start the server due to configuration errors")
        sys.exit(1)

    # Pre-carica modello se richiesto
    # Esempio: load_model("stable-diffusion-v1-5")
    if args.preload_model:
        print(f"\n🔄 Preloading model: {args.preload_model}")
        try:
            load_model(args.preload_model)
            print("✅ Model preloaded successfully")
        except Exception as e:
            print(f"⚠️  Error during preloading: {e}")
    
    print(f"\n🌐 Server starting at http://{args.host}:{args.port}")
    print("📝 Available endpoints:")
    # Build dynamic endpoints list
    endpoints = [
        ("GET", "/ping", "Connection test"),
        ("GET", "/health", "System health check"),
        ("POST", "/generate", "Image generation"),
        ("GET", "/generated_images/<filename>", "Serve generated images"),
        ("POST", "/delete-model", "Delete downloaded or compiled model")
    ]

    # Conditionally add download/compile if functions exist in the dynamic module
    if dl_mod is not None and hasattr(dl_mod, 'download_model'):
        endpoints.append(("POST", "/download", "Download model (HuggingFace)"))
    if dl_mod is not None and hasattr(dl_mod, 'compile_model'):
        endpoints.append(("POST", "/compile", "Compile downloaded model"))

    for method, path, desc in endpoints:
        print(f"   - {method:<4} {path:<30} - {desc}")
    print("=" * 50)
    
    # Avvia il server Flask
    app.run(
        host=args.host,
        port=args.port,
        debug=True,
        threaded=True
    )


if __name__ == '__main__':
    main()
