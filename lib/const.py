import os 
from typing import Dict, Any, Optional

DEFAULT_CONFIG = {
    'model_id': os.environ.get('MODEL_DIFF', 'stable-diffusion-v1-5'),
    'download_dir': os.environ.get('DOWNLOAD_DIR', './models'),
    'compiled_dir': os.environ.get('PRUNA_COMPILED_DIR', './compiled_models'),
    'hf_token': os.environ.get('HF_TOKEN'),
    'torch_dtype': 'float16',
    'device': None,
    'force_cpu': False
}

# Cache per il modello caricato
MODEL_CACHE: Dict[str, Any] = {
    'pipeline': None,
    'model_path': None,
    'model_type': None
}
