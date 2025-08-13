# Esempio di Dockerfile personalizzato con ARG per parametri build-time
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# -- Argomenti build-time per personalizzazione
ARG MODEL_DIFF=CompVis/stable-diffusion-v1-4
ARG DOWNLOAD_DIR=/app/models
ARG PRUNA_COMPILED_DIR=/app/compiled_models
ARG TORCH_DTYPE=float16
ARG HF_TOKEN

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

# -- Script principale parametrizzato
COPY main.py .

# -- Imposta variabili d'ambiente dai parametri build
ENV MODEL_DIFF=${MODEL_DIFF}
ENV DOWNLOAD_DIR=${DOWNLOAD_DIR}
ENV PRUNA_COMPILED_DIR=${PRUNA_COMPILED_DIR}
ENV HF_TOKEN=${HF_TOKEN}

# -- Scarica e compila modello con Pruna (usando parametri)
RUN python3 main.py --torch-dtype ${TORCH_DTYPE} --model-id ${MODEL_DIFF} --download-dir ${DOWNLOAD_DIR} --compiled-dir ${PRUNA_COMPILED_DIR} --hf-token ${HF_TOKEN}

# -- (Facoltativo) Aggiungi ComfyUI e tuoi script custom
# (inserisci qui eventuale COPY degli script comfyui/start.sh ecc.)

# ONLY for TESTING: Esegui un test di inferenza con Pruna
COPY test_pruna_infer.py /test_pruna_infer.py
CMD ["python3", "/test_pruna_infer.py"]

# -- Avvio backend ComfyUI (modifica path/script come necessario!)
# CMD ["python3", "comfyui_server.py"]
