#!/usr/bin/env python3
"""
Script per scaricare il modello Stable Diffusion v1.4 da Hugging Face
"""
import os
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import torch

# Assicurati che esista la directory di destinazione
os.makedirs("./models", exist_ok=True)

print("Scaricando CompVis/stable-diffusion-v1-4 da Hugging Face...")
print("Questo può richiedere alcuni minuti in base alla tua connessione internet.")

# Scarica il modello usando diffusers
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    use_safetensors=True,
    cache_dir="./models/stable-diffusion-v1-4"
)

# Salva il modello localmente
save_path = "./models/stable-diffusion-v1-4"
os.makedirs(save_path, exist_ok=True)
pipeline.save_pretrained(save_path)

print(f"Modello scaricato e salvato in {save_path}")
print("Ora puoi usare questo modello con compile_with_pruna.py utilizzando:")
print(f"python compile_with_pruna.py {save_path}")
