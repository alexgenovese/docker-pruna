# Example Dockerfile with build-time ARGs for customization
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

# -- Build-time arguments for customization
ARG MODEL_DIFF=runwayml/stable-diffusion-v1-5
ARG DOWNLOAD_DIR=/app/models
ARG PRUNA_COMPILED_DIR=/app/compiled_models
ARG PRUNA_COMPILED_MODEL=
ARG TORCH_DTYPE=float16

# NOTE: Do not put sensitive tokens into ARG/ENV in Dockerfiles for public images.
# Use BuildKit secrets for build-time sensitive data so tokens are not stored in
# image layers. During a RUN you can mount the secret with --mount=type=secret
# (available when building with BuildKit / buildx) and read it from
# /run/secrets/<id> without adding it to the image.

# -- Basic OS deps: Python, pip, aria2, git, etc.
RUN apt-get update && \
    apt-get install -y python3-pip aria2 git && \
    rm -rf /var/lib/apt/lists/*

FROM base AS pip-env

WORKDIR /app

# -- Install Python dependencies (Pruna, torch, etc.)
COPY requirements.txt .
RUN pip install --upgrade pip

# -- Set CUDA_HOME for packages that compile CUDA extensions
ENV CUDA_HOME=/usr/local/cuda

# -- Install Python requirements
RUN pip install -r requirements.txt

FROM pip-env AS setup-server

# -- Main scripts
COPY utilities/download_model_and_compile.py ./download_model_and_compile.py
COPY server.py ./server.py

# -- Copy local library modules
COPY lib/ ./lib/

# -- Imposta variabili d'ambiente dai parametri build
ENV MODEL_DIFF=${MODEL_DIFF}
# Use runtime ENV so values passed as build args are also available when container runs.
ENV DOWNLOAD_DIR=${DOWNLOAD_DIR}
ENV PRUNA_COMPILED_DIR=${PRUNA_COMPILED_DIR}
ENV PRUNA_COMPILED_MODEL=${PRUNA_COMPILED_MODEL}

# If a precompiled model is specified at build time, download and save it in the
# compiled models directory using the existing download/compile script. This
# keeps the image self-contained with the compiled model ready to use.
# NOTE: it will increase build-time and image size.
RUN --mount=type=secret,id=hf_token \
    if [ -n "${PRUNA_COMPILED_MODEL}" ]; then \
      echo "Downloading and compiling specified precompiled model: ${PRUNA_COMPILED_MODEL}"; \
      # read token from the mounted secret (silent if missing) and pass to script
      HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
      python3 download_model_and_compile.py --model-id "${PRUNA_COMPILED_MODEL}" --compiled-dir "${PRUNA_COMPILED_DIR}" --hf-token "${HF_TOKEN}"; \
    else \
      echo "No PRUNA_COMPILED_MODEL specified, skipping model download at build-time"; \
    fi

FROM setup-server AS final

# Copy server and library from the setup stage
COPY --from=setup-server /app/server.py /app/server.py
COPY --from=setup-server /app/lib/ /app/lib/

# -- Expose port 8000 used by the Flask server
EXPOSE 8000

# -- Start backend (adjust script/path if needed)
CMD ["python3", "server.py"]
