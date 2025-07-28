FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# -- Dipendenze di base per Python, git, aria2, pip moderni, ecc.
RUN apt-get update && \
    apt-get install -y python3-pip aria2 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -- Installa dipendenze Python: ComfyUI, Pruna, torch, ecc.
COPY requirements.txt .
RUN pip install --upgrade pip

# -- Installa prima torch e numpy separatamente (richiesti da auto-gptq e altri pacchetti)
RUN pip install torch>=2.1 numpy

# -- Imposta CUDA_HOME per i pacchetti che compilano estensioni CUDA
ENV CUDA_HOME=/usr/local/cuda

# -- Installa le altre dipendenze
RUN pip install -r requirements.txt

# -- Scarica il modello safetensors
RUN mkdir -p /models
RUN aria2c -o /models/flux_dev_fp8_scaled_diffusion_model.safetensors \
    https://huggingface.co/alexgenovese/checkpoint/resolve/main/FLUX/flux_dev_fp8_scaled_diffusion_model.safetensors

# -- Script di compilazione Pruna
COPY compile_with_pruna.py .

# -- Compila e salva modello ottimizzato (eseguito in fase build!)
RUN python3 compile_with_pruna.py

# -- (Facoltativo) Aggiungi ComfyUI e tuoi script custom
# (inserisci qui eventuale COPY degli script comfyui/start.sh ecc.)

# -- Directory cache/compiled per Pruna node di ComfyUI
ENV PRUNA_COMPILED_DIR=/compiled_models

# ONLY for TESTING: Esegui un test di inferenza con Pruna
COPY test_pruna_infer.py /test_pruna_infer.py
CMD ["python3", "/test_pruna_infer.py"]

# -- Avvio backend ComfyUI (modifica path/script come necessario!)
# CMD ["python3", "comfyui_server.py"]

