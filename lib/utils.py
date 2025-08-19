import os
import io
import torch
from typing import Dict, Any, Optional
from PIL import Image
import io, base64
from lib.const import DEFAULT_CONFIG, MODEL_CACHE

# Import dinamico delle funzioni di download/compilazione
import importlib.util
spec = importlib.util.spec_from_file_location("download_model_and_compile", os.path.join(os.path.dirname(__file__), "../download_model_and_compile.py"))
if spec and spec.loader:
    dl_mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(dl_mod)
    except Exception as e:
        # Do not fail import if download/compile helper has heavy deps missing
        print(f"⚠️  Warning: failed to load download_model_and_compile.py in lib.utils: {e}")
        dl_mod = None
else:
    dl_mod = None
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

# Funzione per caricare un modello
def load_model(model_id: str):
    """
    Carica un modello compilato Pruna dalla cartella compiled_dir.
    Aggiorna MODEL_CACHE e restituisce la pipeline.
    Se il modello non esiste, solleva un'eccezione.
    """
    compiled_dir = DEFAULT_CONFIG['compiled_dir']
    if not compiled_dir or not os.path.exists(compiled_dir):
        raise FileNotFoundError(f"La cartella dei modelli compilati non esiste: {compiled_dir}")

    model_name = model_id.replace('/', '--')
    model_path = os.path.join(compiled_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello compilato non trovato: {model_path}")

    # Carica il modello compilato con Pruna
    if not PRUNA_AVAILABLE:
        raise ImportError("Il supporto Pruna non è disponibile. Assicurati che il pacchetto 'pruna' sia installato.")

    if PrunaModel is None:
        raise ImportError("Il modello Pruna non è disponibile. Assicurati che il pacchetto 'pruna' sia installato.")

    pipeline = PrunaModel.from_pretrained(model_path)
    MODEL_CACHE['pipeline'] = pipeline
    MODEL_CACHE['model_path'] = model_path
    MODEL_CACHE['model_type'] = 'pruna_compiled'
    MODEL_CACHE['is_pruna_compiled'] = True
    return pipeline

# Funzione per rilevare il tipo di modello basandosi sull'ID o path
def detect_model_type(model_id_or_path):
    """Rileva il tipo di modello basandosi sull'ID o path"""
    model_str = str(model_id_or_path).lower()
    
    if 'flux' in model_str:
        return 'flux'
    elif any(keyword in model_str for keyword in ['stable-diffusion', 'sd-', 'compvis']):
        return 'stable-diffusion'
    else:
        return 'generic'

# Funzione per determinare il miglior dispositivo disponibile
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

# Validazione della configurazione
def validate_configuration() -> Dict[str, Any]:
    status = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'config': {},
        'system_info': {}
    }
    for key, value in DEFAULT_CONFIG.items():
        status['config'][key] = value
        if key in ['model_id'] and not value:
            status['errors'].append(f"Parametro obbligatorio '{key}' non configurato")
            status['valid'] = False
    status['system_info']['torch_version'] = torch.__version__
    status['system_info']['cuda_available'] = torch.cuda.is_available()
    status['system_info']['mps_available'] = torch.backends.mps.is_available()
    status['system_info']['flux_available'] = FLUX_AVAILABLE
    status['system_info']['pruna_available'] = PRUNA_AVAILABLE
    if torch.cuda.is_available():
        status['system_info']['cuda_device_count'] = torch.cuda.device_count()
        status['system_info']['cuda_device_name'] = torch.cuda.get_device_name(0)
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
    if not DEFAULT_CONFIG['hf_token']:
        status['warnings'].append("HF_TOKEN non configurato - alcuni modelli potrebbero non essere accessibili")
    device = get_best_available_device(DEFAULT_CONFIG['force_cpu'], DEFAULT_CONFIG['device'])
    status['system_info']['selected_device'] = device
    return status

# Funzione per convertire un'immagine in base64
def image_to_base64(image: Image.Image) -> str:
    """Converte un'immagine PIL in stringa base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str
