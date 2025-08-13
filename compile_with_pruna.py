from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from pruna import SmashConfig, smash

import torch
import os
import sys

# Permetti override tramite variabile d'ambiente o argomento CLI
MODEL_PATH = './models/stable-diffusion-v1-4'
print(f"Uso MODEL_PATH: {MODEL_PATH}")
SAVE_PATH = "./compiled_models/stable-diffusion-v1-4"

os.makedirs(SAVE_PATH, exist_ok=True)

# Carica modello: verifica se è una directory o un file
if os.path.isdir(MODEL_PATH):
    # Carica da directory
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None  # Opzionale: disabilita il safety checker per velocizzare il processo
    )
else:
    # Carica da singolo file
    try:
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
        )
    except AttributeError:
        # Fallback al caricamento standard
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
        )

cfg = SmashConfig()
compiled = smash(pipe, smash_config=cfg)
compiled.save_pretrained(SAVE_PATH)
print(f"Modello ottimizzato salvato in {SAVE_PATH}")
