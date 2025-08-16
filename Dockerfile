# Esempio di Dockerfile personalizzato con ARG per parametri build-time
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

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

FROM base AS pip-env 

WORKDIR /app

# -- Installa dipendenze Python: Pruna, torch, ecc.
COPY requirements.txt .
RUN pip install --upgrade pip

# -- Installa prima torch e numpy separatamente (richiesti da auto-gptq e altri pacchetti)
# RUN pip install torch>=2.1 numpy

# -- Imposta CUDA_HOME per i pacchetti che compilano estensioni CUDA
ENV CUDA_HOME=/usr/local/cuda

# -- Installa le altre dipendenze
RUN pip install -r requirements.txt

FROM pip-env AS setup-server

# -- Script principali
COPY download_model_and_compile.py .
COPY server.py .
COPY test_pruna_infer.py .

# -- Copia la directory lib con tutti i moduli necessari
COPY lib/ ./lib/

# -- Imposta variabili d'ambiente dai parametri build
ENV MODEL_DIFF=${MODEL_DIFF}
ENV DOWNLOAD_DIR=${DOWNLOAD_DIR}
ENV PRUNA_COMPILED_DIR=${PRUNA_COMPILED_DIR}
ENV HF_TOKEN=${HF_TOKEN}

# -- Scarica e compila modello con Pruna (usando parametri)
# RUN if [ -n "${HF_TOKEN}" ]; then \
#         python3 download_model_and_compile.py --torch-dtype ${TORCH_DTYPE} --model-id ${MODEL_DIFF} --download-dir ${DOWNLOAD_DIR} --compiled-dir ${PRUNA_COMPILED_DIR} --hf-token ${HF_TOKEN}; \
#     fi


FROM setup-server AS final 

# Copia i modelli scaricati e compilati dalla fase precedente
COPY --from=setup-server /app/server.py /app/server.py
COPY --from=setup-server /app/lib/ /app/lib/

# ONLY for TESTING: Esegui un test di inferenza con Pruna
# COPY test_pruna_infer.py /test_pruna_infer.py
# CMD ["python3", "/test_pruna_infer.py"]

# -- Espone la porta 8000 utilizzata dal server Flask
EXPOSE 8000

# -- Avvio backend (modifica path/script come necessario!)
CMD ["python3", "server.py"]
