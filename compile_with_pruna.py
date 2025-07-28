from diffusers import DiffusionPipeline
from pruna import SmashConfig, smash

import torch
import os

MODEL_PATH = "/models/flux_dev_fp8_scaled_diffusion_model.safetensors"
SAVE_PATH = "/compiled_models/flux_pruna"

os.makedirs(SAVE_PATH, exist_ok=True)

# Carica modello: personalizza in base a come la tua pipeline gestisce i 'safetensors'
pipe = DiffusionPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float16,
)

cfg = SmashConfig()
compiled = smash(pipe, smash_config=cfg)
compiled.save_pretrained(SAVE_PATH)
print(f"Modello ottimizzato salvato in {SAVE_PATH}")
