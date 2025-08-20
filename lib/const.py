import os 
from typing import Dict, Any, Optional

DEFAULT_CONFIG = {
    # Use "or" fallback so an empty env var won't override the default value.
    'model_id': os.environ.get('MODEL_DIFF') or 'stable-diffusion-v1-5',
    'download_dir': os.environ.get('DOWNLOAD_DIR') or './models',
    'compiled_dir': os.environ.get('PRUNA_COMPILED_DIR') or './compiled_models',
    'hf_token': os.environ.get('HF_TOKEN') or None,
    'torch_dtype': os.environ.get('TORCH_DTYPE') or 'float16',
    # Allow overriding device via env, or keep None for auto-detection
    'device': os.environ.get('DEVICE') or None,
    # Treat common truthy env values as True
    'force_cpu': str(os.environ.get('FORCE_CPU', 'False')).lower() in ('1', 'true', 'yes')
}

# Cache per il modello caricato
MODEL_CACHE: Dict[str, Any] = {
    'pipeline': None,
    'model_path': None,
    'model_type': None
}
