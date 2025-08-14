#!/usr/bin/env python3
"""
FastAPI Server API per inferenza con modelli Flux Krea.
Espone endpoint REST per generazione di immagini utilizzando i modelli compilati con Pruna.
"""

import os
import sys
import base64
import io
import json
import traceback
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from PIL import Image
from flask import Flask, request, jsonify

# Import dinamico delle funzioni di download/compilazione
import importlib.util
spec = importlib.util.spec_from_file_location("download_model_and_compile", os.path.join(os.path.dirname(__file__), "download_model_and_compile.py"))
if spec and spec.loader:
    dl_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl_mod)
else:
    raise ImportError("Impossibile importare download_model_and_compile.py")
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
try:
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    FluxPipeline = None

try:
    from pruna import PrunaModel
    PRUNA_AVAILABLE = True
except ImportError:
    PRUNA_AVAILABLE = False
    PrunaModel = None


# Configurazione dell'applicazione Flask
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Cache per il modello caricato
MODEL_CACHE: Dict[str, Any] = {
    'pipeline': None,
    'model_path': None,
    'model_type': None
}

# Configurazione di default (come in main.py)
DEFAULT_CONFIG = {
    'model_id': os.environ.get('MODEL_DIFF', 'stable-diffusion-v1-5'),
    'download_dir': os.environ.get('DOWNLOAD_DIR', './models'),
    'compiled_dir': os.environ.get('PRUNA_COMPILED_DIR', './compiled_models'),
    'hf_token': os.environ.get('HF_TOKEN'),
    'torch_dtype': 'float16',
    'device': None,
    'force_cpu': False
}


def detect_model_type(model_id_or_path):
    """Rileva il tipo di modello basandosi sull'ID o path"""
    model_str = str(model_id_or_path).lower()
    
    if 'flux' in model_str:
        return 'flux'
    elif any(keyword in model_str for keyword in ['stable-diffusion', 'sd-', 'compvis']):
        return 'stable-diffusion'
    else:
        return 'generic'


def get_best_available_device(force_cpu=False, device_override=None):
    """Determina il miglior dispositivo disponibile"""
    if force_cpu or device_override == 'cpu':
        return "cpu"
    
    if device_override:
        if device_override == 'cuda' and torch.cuda.is_available():
            return "cuda"
        elif device_override == 'mps' and torch.backends.mps.is_available():
            return "mps"
    
    # Auto-detection
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def validate_configuration() -> Dict[str, Any]:
    """Valida la configurazione del sistema e ritorna lo stato"""
    status = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'config': {},
        'system_info': {}
    }
    
    # Verifica parametri di configurazione
    for key, value in DEFAULT_CONFIG.items():
        status['config'][key] = value
        
        if key in ['model_id'] and not value:
            status['errors'].append(f"Parametro obbligatorio '{key}' non configurato")
            status['valid'] = False
    
    # Verifica disponibilità dispositivi
    status['system_info']['torch_version'] = torch.__version__
    status['system_info']['cuda_available'] = torch.cuda.is_available()
    status['system_info']['mps_available'] = torch.backends.mps.is_available()
    status['system_info']['flux_available'] = FLUX_AVAILABLE
    status['system_info']['pruna_available'] = PRUNA_AVAILABLE
    
    if torch.cuda.is_available():
        status['system_info']['cuda_device_count'] = torch.cuda.device_count()
        status['system_info']['cuda_device_name'] = torch.cuda.get_device_name(0)
    
    # Verifica directory modelli
    if DEFAULT_CONFIG['compiled_dir']:
        if not os.path.exists(DEFAULT_CONFIG['compiled_dir']):
            status['warnings'].append(f"Directory modelli compilati non trovata: {DEFAULT_CONFIG['compiled_dir']}")
    else:
        status['warnings'].append("PRUNA_COMPILED_DIR non configurata")
    
    if DEFAULT_CONFIG['download_dir']:
        if not os.path.exists(DEFAULT_CONFIG['download_dir']):
            status['warnings'].append(f"Directory download non trovata: {DEFAULT_CONFIG['download_dir']}")
    else:
        status['warnings'].append("DOWNLOAD_DIR non configurata")
    
    # Verifica token HuggingFace
    if not DEFAULT_CONFIG['hf_token']:
        status['warnings'].append("HF_TOKEN non configurato - alcuni modelli potrebbero non essere accessibili")
    
    # Determina dispositivo di default
    device = get_best_available_device(DEFAULT_CONFIG['force_cpu'], DEFAULT_CONFIG['device'])
    status['system_info']['selected_device'] = device
    
    return status


def find_model_path(model_id: str) -> tuple[Optional[str], bool]:
    """Trova il path del modello, prioritizzando i modelli compilati
    Ritorna (path, is_pruna_compiled)"""
    
    # Converti l'ID del modello in nome file
    model_name = model_id.replace('/', '--')
    
    # Priorità 1: Modello compilato con Pruna
    if DEFAULT_CONFIG['compiled_dir']:
        # Cerca prima la directory con il nome diretto del modello
        compiled_path = os.path.join(DEFAULT_CONFIG['compiled_dir'], model_name)
        if os.path.exists(compiled_path):
            # Verifica se contiene direttamente il smash_config.json
            if os.path.exists(os.path.join(compiled_path, 'smash_config.json')):
                return compiled_path, True
            
            # Altrimenti cerca nelle sottodirectory
            if os.path.isdir(compiled_path):
                for subdir in os.listdir(compiled_path):
                    subdir_path = os.path.join(compiled_path, subdir)
                    if os.path.isdir(subdir_path) and os.path.exists(os.path.join(subdir_path, 'smash_config.json')):
                        return subdir_path, True
        
        # Cerca anche nella sottodirectory con nome duplicato (struttura legacy)
        compiled_path_nested = os.path.join(DEFAULT_CONFIG['compiled_dir'], model_name, model_name)
        if os.path.exists(compiled_path_nested) and os.path.exists(os.path.join(compiled_path_nested, 'smash_config.json')):
            return compiled_path_nested, True
    
    # Priorità 2: Modello scaricato (non compilato)
    if DEFAULT_CONFIG['download_dir']:
        download_path = os.path.join(DEFAULT_CONFIG['download_dir'], model_name)
        if os.path.exists(download_path):
            # Verifica se contiene direttamente il model_index.json (standard per diffusers)
            if os.path.exists(os.path.join(download_path, 'model_index.json')):
                return download_path, False
            
            # Altrimenti cerca nelle sottodirectory
            if os.path.isdir(download_path):
                for subdir in os.listdir(download_path):
                    subdir_path = os.path.join(download_path, subdir)
                    if os.path.isdir(subdir_path) and os.path.exists(os.path.join(subdir_path, 'model_index.json')):
                        return subdir_path, False
        
        # Cerca anche nella sottodirectory con nome duplicato
        download_path_nested = os.path.join(DEFAULT_CONFIG['download_dir'], model_name, model_name)
        if os.path.exists(download_path_nested) and os.path.exists(os.path.join(download_path_nested, 'model_index.json')):
            return download_path_nested, False
    
    # Priorità 3: Path diretto se esiste
    if os.path.exists(model_id):
        # Verifica se è un modello Pruna o diffusers standard
        is_pruna = os.path.exists(os.path.join(model_id, 'smash_config.json'))
        return model_id, is_pruna
    
    return None, False


def load_model(model_id: str, force_reload: bool = False) -> Any:
    """Carica il modello in cache"""
    
    # Verifica se il modello è già caricato
    if not force_reload and MODEL_CACHE['pipeline'] is not None and MODEL_CACHE['model_path'] == model_id:
        return MODEL_CACHE['pipeline']
    
    # Trova il path del modello
    model_path, is_pruna_compiled = find_model_path(model_id)
    if not model_path:
        raise FileNotFoundError(f"Modello non trovato: {model_id}")
    
    # Rileva il tipo di modello
    model_type = detect_model_type(model_path)
    
    # Configura torch dtype
    torch_dtype = torch.float16 if DEFAULT_CONFIG['torch_dtype'] == 'float16' else torch.float32
    
    # Determina dispositivo
    device = get_best_available_device(DEFAULT_CONFIG['force_cpu'], DEFAULT_CONFIG['device'])
    
    print(f"Caricamento modello: {model_path}")
    print(f"Tipo modello: {model_type}")
    print(f"Modello compilato Pruna: {is_pruna_compiled}")
    print(f"Dispositivo: {device}")
    print(f"Dtype: {torch_dtype}")
    
    # Carica il modello basandosi sul tipo
    pipeline = None
    
    try:
        # Se il modello è compilato con Pruna, usa PrunaModel
        if is_pruna_compiled and PRUNA_AVAILABLE and PrunaModel is not None:
            print("Caricamento con PrunaModel...")
            pipeline = PrunaModel.from_pretrained(model_path)
            
            # Per i modelli Pruna, non è necessario spostare esplicitamente sul dispositivo
            # dato che Pruna gestisce automaticamente l'ottimizzazione
            
        elif model_type == 'flux' and FLUX_AVAILABLE and FluxPipeline is not None:
            print("Caricamento con FluxPipeline...")
            pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
        elif model_type == 'stable-diffusion':
            print("Caricamento con StableDiffusionPipeline...")
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                safety_checker=None
            )
        else:
            print("Caricamento con DiffusionPipeline generico...")
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
        
        # Sposta il modello sul dispositivo appropriato (solo per modelli non-Pruna)
        if not is_pruna_compiled and device != "cpu":
            pipeline = pipeline.to(device)
        
        # Salva in cache
        MODEL_CACHE['pipeline'] = pipeline
        MODEL_CACHE['model_path'] = model_id
        MODEL_CACHE['model_type'] = model_type
        MODEL_CACHE['is_pruna_compiled'] = is_pruna_compiled
        
        print(f"Modello caricato con successo in cache")
        return pipeline
        
    except Exception as e:
        raise RuntimeError(f"Errore nel caricamento del modello: {str(e)}")


def image_to_base64(image: Image.Image) -> str:
    """Converte un'immagine PIL in stringa base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


@app.route('/ping', methods=['GET'])
def ping():
    """Endpoint di ping semplice"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'timestamp': str(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
    })


@app.route('/health', methods=['GET'])
def health():
    """Endpoint di health check che valida la configurazione"""
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


@app.route('/download', methods=['POST'])
def download():
    """Scarica i pesi di un modello HuggingFace se non già presenti, elimina .safetensors dalla root prima del download."""
    try:
        data = request.get_json() or {}
        model_id = data.get('model_id') or data.get('modelId')
        if not model_id:
            return jsonify({'status': 'error', 'message': 'Parametro "model_id" obbligatorio'}), 400

        # Controlla se il modello esiste già
        model_exists, _, model_path, _ = dl_mod.check_model_exists(
            model_id,
            DEFAULT_CONFIG['download_dir'],
            DEFAULT_CONFIG['compiled_dir']
        )
        if model_exists:
            return jsonify({'status': 'ok', 'message': f'Modello già presente: {model_path}', 'model_path': model_path})

        # Elimina tutti i .safetensors dalla root della repo
        root_dir = os.path.dirname(os.path.abspath(__file__))
        deleted = []
        for fname in os.listdir(root_dir):
            if fname.endswith('.safetensors'):
                try:
                    os.remove(os.path.join(root_dir, fname))
                    deleted.append(fname)
                except Exception:
                    pass

        # Scarica il modello
        try:
            model_path = dl_mod.download_model(
                model_id,
                DEFAULT_CONFIG['download_dir'],
                torch_dtype=torch.float16 if DEFAULT_CONFIG['torch_dtype'] == 'float16' else torch.float32,
                hf_token=DEFAULT_CONFIG['hf_token']
            )
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Errore nel download: {str(e)}'}), 500

        return jsonify({'status': 'success', 'message': f'Modello scaricato in {model_path}', 'deleted_files': deleted, 'model_path': model_path})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Errore interno: {str(e)}'}), 500


@app.route('/compile', methods=['POST'])
def compile():
    """Compila i pesi di un modello già scaricato, se non presente restituisce errore."""
    try:
        data = request.get_json() or {}
        model_id = data.get('model_id') or data.get('modelId')
        compilation_mode = data.get('compilation_mode', 'moderate')
        if not model_id:
            return jsonify({'status': 'error', 'message': 'Parametro "model_id" obbligatorio'}), 400

        # Controlla se il modello base esiste
        model_exists, compiled_exists, model_path, compiled_path = dl_mod.check_model_exists(
            model_id,
            DEFAULT_CONFIG['download_dir'],
            DEFAULT_CONFIG['compiled_dir']
        )
        if not model_exists:
            return jsonify({'status': 'error', 'message': 'Modello non trovato. Scaricalo prima con /download-weights.'}), 404

        # Se già compilato, restituisci info
        if compiled_exists:
            return jsonify({'status': 'ok', 'message': f'Modello già compilato: {compiled_path}', 'compiled_path': compiled_path})

        # Compila il modello
        try:
            compiled_path = dl_mod.compile_model_with_pruna(
                model_path,
                DEFAULT_CONFIG['compiled_dir'],
                torch.float16 if DEFAULT_CONFIG['torch_dtype'] == 'float16' else torch.float32,
                compilation_mode,
                DEFAULT_CONFIG['force_cpu'],
                DEFAULT_CONFIG['device']
            )
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Errore nella compilazione: {str(e)}'}), 500

        return jsonify({'status': 'success', 'message': f'Modello compilato e salvato in {compiled_path}', 'compiled_path': compiled_path})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Errore interno: {str(e)}'}), 500


@app.route('/generate', methods=['POST'])
def generate():
    """Endpoint per generazione di immagini con Flux Krea"""
    try:
        # Parse dei parametri della richiesta
        data = request.get_json() or {}
        
        # Parametri di generazione
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
        
        # Validazione parametri
        if not prompt:
            return jsonify({
                'status': 'error',
                'message': 'Parametro "prompt" obbligatorio'
            }), 400
        
        if num_images > 4:
            return jsonify({
                'status': 'error',
                'message': 'Numero massimo di immagini: 4'
            }), 400
        
        # Carica il modello
        try:
            pipeline = load_model(model_id)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Errore nel caricamento del modello: {str(e)}'
            }), 500
        
        # Configurazione del generatore con seed
        generator = None
        if seed is not None:
            device = get_best_available_device(DEFAULT_CONFIG['force_cpu'], DEFAULT_CONFIG['device'])
            generator = torch.Generator(device=device).manual_seed(int(seed))
        
        # Preparazione parametri per la pipeline
        generation_params = {
            'prompt': prompt,
            'num_inference_steps': num_inference_steps,
            'width': width,
            'height': height,
            'num_images_per_prompt': num_images,
            'generator': generator
        }
        
        # Aggiungi parametri specifici per tipo di modello
        model_type = MODEL_CACHE.get('model_type', 'generic')
        is_pruna_compiled = MODEL_CACHE.get('is_pruna_compiled', False)
        
        if is_pruna_compiled:
            # Parametri per modelli compilati con Pruna
            # Pruna mantiene l'interfaccia simile a diffusers
            generation_params['guidance_scale'] = guidance_scale
            if negative_prompt:
                generation_params['negative_prompt'] = negative_prompt
        elif model_type in ['stable-diffusion', 'generic']:
            # Parametri per Stable Diffusion
            generation_params['guidance_scale'] = guidance_scale
            if negative_prompt:
                generation_params['negative_prompt'] = negative_prompt
        elif model_type == 'flux':
            # Parametri per FLUX (potrebbe avere parametri diversi)
            generation_params['guidance_scale'] = guidance_scale
            if negative_prompt:
                generation_params['negative_prompt'] = negative_prompt
        
        print(f"Generazione immagine con parametri: {generation_params}")
        
        # Genera le immagini
        try:
            with torch.no_grad():
                result = pipeline(**generation_params)
                images = result.images
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Errore durante la generazione: {str(e)}',
                'traceback': traceback.format_exc()
            }), 500
        
        # Converti le immagini in base64
        images_b64 = []
        saved_files = []
        
        for i, image in enumerate(images):
            img_b64 = image_to_base64(image)
            images_b64.append({
                'index': i,
                'base64': img_b64,
                'format': 'png'
            })
            
            # Salva l'immagine in locale se debug = true
            if debug:
                try:
                    # Crea directory di output se non esiste
                    output_dir = "./generated_images"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Genera nome file con timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"generated_{timestamp}_img_{i:02d}.png"
                    file_path = os.path.join(output_dir, filename)
                    
                    # Salva l'immagine
                    image.save(file_path)
                    saved_files.append({
                        'index': i,
                        'filename': filename,
                        'path': file_path
                    })
                    print(f"💾 Debug: Immagine salvata in {file_path}")
                    
                except Exception as e:
                    print(f"⚠️  Errore nel salvare l'immagine {i}: {str(e)}")
                    saved_files.append({
                        'index': i,
                        'error': str(e)
                    })
        
        response_data = {
            'status': 'success',
            'message': f'Generati {len(images)} immagini',
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
        
        # Aggiungi informazioni sui file salvati se debug = true
        if debug and saved_files:
            response_data['saved_files'] = saved_files
            
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Errore interno del server: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handler per 404"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint non trovato',
        'available_endpoints': ['/ping', '/health', '/generate']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handler per errori interni"""
    return jsonify({
        'status': 'error',
        'message': 'Errore interno del server',
        'details': str(error)
    }), 500


def main():
    """Funzione principale per avviare il server"""
    import argparse

    parser = argparse.ArgumentParser(description='FastAPI Server API')
    parser.add_argument('--host', default='0.0.0.0', help='Host su cui avviare il server')
    parser.add_argument('--port', type=int, default=8000, help='Porta su cui avviare il server')
    parser.add_argument('--debug', action='store_true', help='Attiva modalità debug')
    parser.add_argument('--preload-model', help='Pre-carica un modello all\'avvio')
    
    args = parser.parse_args()

    print("🚀 Avvio FastAPI Server...")
    print("=" * 50)
    
    # Stampa configurazione
    status = validate_configuration()
    print("📋 Configurazione:")
    for key, value in status['config'].items():
        if 'token' in key.lower() and value:
            value = '***IMPOSTATO***'
        print(f"   - {key}: {value}")
    
    print(f"\n💻 Sistema:")
    for key, value in status['system_info'].items():
        print(f"   - {key}: {value}")
    
    if status['warnings']:
        print(f"\n⚠️  Avvisi:")
        for warning in status['warnings']:
            print(f"   - {warning}")
    
    if status['errors']:
        print(f"\n❌ Errori:")
        for error in status['errors']:
            print(f"   - {error}")
        print("\n🛑 Impossibile avviare il server a causa di errori di configurazione")
        sys.exit(1)

    # Pre-carica modello se richiesto
    # Esempio: load_model("stable-diffusion-v1-5")
    if args.preload_model:
        print(f"\n🔄 Pre-caricamento modello: {args.preload_model}")
        try:
            load_model(args.preload_model)
            print("✅ Modello pre-caricato con successo")
        except Exception as e:
            print(f"⚠️  Errore nel pre-caricamento: {e}")
    
    print(f"\n🌐 Server in avvio su http://{args.host}:{args.port}")
    print("📝 Endpoints disponibili:")
    print(f"   - GET  /ping      - Test di connessione")
    print(f"   - GET  /health    - Controllo stato sistema")
    print(f"   - POST /generate  - Generazione immagini")
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
