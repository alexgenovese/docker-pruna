# Docker Pruna — Download & Compile Diffusers Models

![CI](https://img.shields.io/badge/CI-pending-lightgrey) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![Docker](https://img.shields.io/badge/docker-ready-blue) ![License](https://img.shields.io/badge/license-Unspecified-lightgrey)

Docker Pruna provides a configurable Docker environment, CLI tools and a small HTTP API to download, compile and serve diffusion models (Stable Diffusion, FLUX and others) optimized with Pruna for faster inference.

This README is a cleaned, English-first landing page: features, quick-start, configuration, API examples, troubleshooting and developer notes.

## Key features

- Download models from Hugging Face into `./models/`.
- Compile models with Pruna and store optimized artifacts in `./compiled_models/`.
- Lightweight Flask API to trigger download, compile, generate and delete operations.
- Smart `PrunaModelConfigurator` that provides device-aware, safe Pruna configurations and fallbacks.
- Compilation modes: `fast`, `moderate`, `normal` (speed vs quality trade-offs).
- Helpers for CUDA/MPS/CPU diagnostics and memory-aware compilation.

## 📋 Environment Variables

- `MODEL_DIFF`: model ID on Hugging Face (default: `CompVis/stable-diffusion-v1-4`)
- `DOWNLOAD_DIR`: Directory to download the models (default: `./models`)
- `PRUNA_COMPILED_DIR`: Directory to store compiled models with Pruna (default: `./compiled_models`)

### How to use by CLI
```bash
python3 main.py --help

optional arguments:
  --model-id MODEL_ID    Hugging Face model ID to download
  --download-dir DIR     Directory to download models
  --compiled-dir DIR     Directory to save compiled Pruna models
  --skip-download        Skip download step (use existing model)
  --skip-compile         Skip compilation step (only download)
  --torch-dtype TYPE     Torch dtype for model loading (float16/float32)
```


## Quick start

Prerequisites:

- Python 3.8+
- (Optional) CUDA 12.1+ for GPU-based compilation/inference
- Enough disk space (3–7 GB per model)

Clone and install dependencies:

```bash
git clone <your-repo>
cd docker-pruna
pip install -r requirements.txt
```

Download and compile a model (moderate mode):

```bash
python3 download_model_and_compile.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --compilation-mode moderate
```

Run the Flask API locally:

```bash
python3 server.py --host 127.0.0.1 --port 8000 --debug &
```

## Configuration

Environment variables (defaults shown):

- `MODEL_DIFF` — default model id (default: `CompVis/stable-diffusion-v1-4`)
- `DOWNLOAD_DIR` — where models are downloaded (default: `./models`)
- `PRUNA_COMPILED_DIR` — where compiled Pruna models are saved (default: `./compiled_models`)

CLI arguments (see `download_model_and_compile.py --help`):

```bash
python3 download_model_and_compile.py --help

# common options: --model-id, --download-dir, --compiled-dir, --skip-download, --skip-compile,
# --torch-dtype, --compilation-mode, --device, --force-cpu
```

## API endpoints (JSON)

POST /download
- Download a Hugging Face model into the download directory.

Body example:

```json
{ "model_id": "runwayml/stable-diffusion-v1-5" }
```

POST /compile
- Compile an already downloaded model with Pruna and save into the compiled models directory.

Body example:

```json
{ "model_id": "runwayml/stable-diffusion-v1-5", "compilation_mode": "fast" }
```

POST /generate
- Generate images from a prompt using a compiled model.

Body example:

```json
{
  "model_id": "runwayml/stable-diffusion-v1-5",
  "prompt": "A beautiful sunset over the ocean",
  "num_inference_steps": 20,
  "guidance_scale": 7.5
}
```

Response contains base64-encoded images and optional saved file paths when `debug: true`.

POST /delete-model
- Delete downloaded and/or compiled folders for a given model.

Body example:

```json
{ "model_id": "runwayml/stable-diffusion-v1-5", "type": "all" }
```

GET /ping — basic liveness check

GET /health — server configuration, system info, warnings and errors

## Practical CLI examples

Download only:

```bash
python3 download_model_and_compile.py --model-id runwayml/stable-diffusion-v1-5
```

Download + compile (fast):

```bash
python3 download_model_and_compile.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --compilation-mode fast
```

Force CUDA compile (helper):

```bash
python3 force_cuda_compile.py --model-id runwayml/stable-diffusion-v1-5 --mode fast
```

Run server and call compile endpoint (example):

```bash
# start server
python3 server.py --host 127.0.0.1 --port 8000 --debug &

# request compilation
curl -X POST http://127.0.0.1:8000/compile \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "compilation_mode": "fast"}'

# stop server
pkill -f server.py
```

Generate images via API:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "prompt": "A scenic landscape at sunset"}'
```

## Compilation modes

- Fast — quick compile for development (DeepCache, half precision). Good for rapid iterations.
- Moderate — balanced quality/speed (TorchCompile + HQQ 8-bit where compatible).
- Normal — full optimizations for production (FORA, factorizer, autotune). Longer compile time.

Use `--compilation-mode` to pick the mode when running the CLI or API compile endpoint.

## Diagnostics & helper scripts

- `test_pruna_cuda.py` — CUDA and Pruna diagnostics.
- `check_pruna_setup.py` — check versions and environment.
- `compile_with_memory_mgmt.py` — memory-aware compilation helper.
- `restart_clean_compile.sh` — restart process and free GPU memory before compiling.

## File layout

```
lib/
├── pruna_config.py          # Smart configurator
├── const.py                 # Constants
└── utils.py                 # Utilities

download_model_and_compile.py # Main download/compile CLI
server.py                     # Flask API server
test_pruna_config.py          # Configurator tests
test_pruna_cuda.py            # CUDA/Pruna diagnostics
force_cuda_compile.py         # Force CUDA compile helper
compile_with_memory_mgmt.py   # Memory-aware compilation
restart_clean_compile.sh      # Restart + cleanup helper
```

## Troubleshooting & common issues

- "Model is not compatible with FORA" — switched to DeepCache for some SD 1.5/1.4 variants.
- "DeepCache is not compatible with device MPS" — DeepCache disabled on MPS.
- CUDA OOM during compilation — try `restart_clean_compile.sh` or `compile_with_memory_mgmt.py`.
- If Pruna compiles for CPU despite CUDA availability, try `force_cuda_compile.py`.

## System requirements

Minimum:
- Python 3.8+
- 4 GB RAM
- 10 GB disk

Recommended:
- CUDA 12.1+ and compatible NVIDIA driver for GPU workflows
- 16 GB+ RAM for larger models
- Apple Silicon (M1/M2/M3) supported with device-specific fallbacks

## Credits

- Project maintainer: repository owner
- Libraries and tools: Pruna (smash), Hugging Face `diffusers`, `huggingface_hub`, PyTorch, Flask

If you want a shorter GitHub front-page summary (1–2 paragraphs, badges, and quick links) I can prepare that and move the long-form content to `docs/README-full.md`.

# 1. Force CUDA explicitly
python3 force_cuda_compile.py --model-id MODEL_ID --mode fast

# 2. Use memory-aware compilation
python3 compile_with_memory_mgmt.py --model-id MODEL_ID --mode fast

# 3. Restart with clean memory
./restart_clean_compile.sh MODEL_ID fast

# 4. Set recommended env vars
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES=0
python3 download_model_and_compile.py --device cuda --model-id MODEL_ID
```

#### Problem: "CUDA out of memory" during compilation
Cause: GPU memory is already occupied by other processes.

Automatic fixes:
```bash
./restart_clean_compile.sh runwayml/stable-diffusion-v1-5 fast

python3 compile_with_memory_mgmt.py --model-id MODEL_ID --mode fast

python3 download_model_and_compile.py \
  --model-id MODEL_ID \
  --compilation-mode fast \
  --device cuda
```

#### Diagnostics
```bash
python3 test_pruna_cuda.py

# Example output:
# ✅ CUDA available: True
# ✅ GPU: NVIDIA GeForce RTX 4090
# ✅ Total GPU memory: 24.0 GB
# ❌ Pruna CUDA: configuration error
# 💡 Recommendation: reinstall Pruna
```

### Common issues and automatic fixes

| Issue | Affected models | Automatic fix |
|------|------------------|---------------|
| "Model is not compatible with fora" | SD 1.5, SD 1.4 | Switch to DeepCache instead of FORA |
| "deepcache is not compatible with device mps" | All on MPS | Disable DeepCache on MPS |
| Missing optional deps on MPS | HQQ & others | Disable HQQ on MPS |
| Missing packages | Various optimizations | Fallback to safe minimal config |

## File layout and helper scripts

```
lib/
├── pruna_config.py          # Smart configurator
├── const.py                 # Constants
└── utils.py                 # Utilities

download_model_and_compile.py # Main script
test_pruna_config.py         # Configurator tests
test_pruna_cuda.py           # CUDA/Pruna diagnostics
check_pruna_setup.py         # Setup checker
force_cuda_compile.py        # Force CUDA compile helper
compile_with_memory_mgmt.py  # Memory-aware compilation
restart_clean_compile.sh     # Restart and clean memory helper
```

## 🔧 Docker usage

### Build examples
```bash
docker build -t docker-pruna .
docker build --build-arg COMPILATION_MODE=fast -t docker-pruna .
docker build \
  --build-arg MODEL_DIFF="runwayml/stable-diffusion-v1-5" \
  --build-arg COMPILATION_MODE=moderate \
  -t docker-pruna .
```

### Test and validation
```bash
python3 test_pruna_config.py
python3 test_pruna_infer.py
./test_main.sh
```

## 🎯 ComfyUI integration

To integrate with ComfyUI, copy compiled models into ComfyUI's model path or point ComfyUI to the `compiled_models` directory. The project includes examples to export compiled models into ComfyUI checkpoints.

## 🚨 System requirements

Minimum:
- Python 3.8+
- 4 GB RAM
- 10 GB disk

Recommended:
- CUDA 12.1+ for full GPU support
- 16 GB+ RAM for large models
- Apple Silicon M1/M2/M3 supported with device-aware fallbacks

## 🆕 What's new in this release

### Fixed
- "Model is not compatible with fora" for Stable Diffusion 1.5
- Compatibility problems on Apple Silicon (MPS)
- Automatic handling for cases where Pruna would compile for CPU despite available CUDA
- Memory handling improvements to avoid common OOMs

### New features
- `PrunaModelConfigurator` smart class
- Auto-detection for several model families
- Device-specific configuration recommendations
- Tests and diagnostic scripts

## GitHub: Quick Usage, Features, Examples & Credits (English)

### What this repository provides

This repository is a Docker-ready toolkit and a lightweight Flask API to download, compile and serve diffusion models (Stable Diffusion, FLUX and others) optimized with Pruna for faster inference. It includes an intelligent configurator that manages device compatibility (CUDA, CPU, Apple MPS), automatic fallbacks, and memory-aware compilation modes.

Key features:
- Download models from Hugging Face into `./models/`.
- Compile models with Pruna and store optimized artifacts in `./compiled_models/`.
- Expose a small HTTP API to download, compile, generate images and delete models.
- Auto-detection and recommended Pruna configuration per model and device.
- Multiple compilation modes: `fast`, `moderate`, `normal` to trade off speed vs quality.
- Device-aware fallbacks for MPS/CPU to avoid incompatible Pruna options.
- Diagnostic and memory-management helper scripts.

API Endpoints
- `POST /download` — download a model to `models/`.
- `POST /compile` — compile an existing downloaded model into `compiled_models/`.
- `POST /generate` — generate images with a compiled model.
- `POST /delete-model` — remove downloaded/compiled model folders.
- `GET /ping` — basic liveness check.
- `GET /health` — health and configuration report.

Quick CLI examples

1) Download a model (CLI):

```bash
python3 download_model_and_compile.py --model-id runwayml/stable-diffusion-v1-5
```

2) Download + compile (fast mode):

```bash
python3 download_model_and_compile.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --compilation-mode fast
```

3) Force CUDA compilation (if you have a GPU):

```bash
python3 force_cuda_compile.py --model-id runwayml/stable-diffusion-v1-5 --mode fast
```

4) Run the Flask API locally and test compile endpoint (example):

```bash
# start server in background
python3 server.py --host 127.0.0.1 --port 8000 --debug &

# request compilation (replace host/port if needed)
curl -X POST http://127.0.0.1:8000/compile \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "compilation_mode": "fast"}'

# stop server
pkill -f server.py
```

5) Generate images via API:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "prompt": "A scenic landscape at sunset"}'
```

Environment variables and Docker
- `MODEL_DIFF` — default model id used by CLI tools (override per-run or via Docker build arg).
- `DOWNLOAD_DIR` — path where models are downloaded (default `./models`).
- `PRUNA_COMPILED_DIR` — path where compiled models are written (default `./compiled_models`).

You can pass these as Docker build args or environment variables when running the container.

Practical tips
- Use `--skip-download` or `--skip-compile` when you want only one step.
- Prefer `fast` for quick iterations and `moderate/normal` for production-quality results.
- On Apple Silicon prefer `--device mps` and `fast` to avoid Pruna incompatibilities.
- If you see `CUDA out of memory` during compilation, use `restart_clean_compile.sh` or `compile_with_memory_mgmt.py`.

Credits
- Project maintainer: repository owner
- Major libraries and tools used:
  - Pruna (smash)
  - Hugging Face `diffusers` and `huggingface_hub`
  - PyTorch
  - Flask for the lightweight API

If you want a compact GitHub landing section (badges, short TL;DR, GIF) I can prepare a shorter front-page variant.

```
### 🔮 Roadmap Futura
- Support per nuovi modelli (SD 4.0, Cascade, etc.)
- Ottimizzazioni specifiche per architetture GPU
- Configurazioni dinamiche basate su hardware detection
- Plugin ComfyUI nativo con configurazione automatica
- **🆕 Monitoring continuo memoria** durante training/inference
- **🆕 Auto-scaling GPU** per workload variabili

---

**💡 TL;DR**: Ora puoi compilare qualsiasi modello supportato senza preoccuparti di errori di compatibilità. Il sistema rileva automaticamente modello, dispositivo e configura Pruna in modo ottimale. Zero configurazione manuale richiesta! 🎉

## GitHub: Quick Usage, Features, Examples & Credits (English)

### What this repository provides

This repository is a Docker-ready toolkit and a lightweight Flask API to download, compile and serve diffusion models (Stable Diffusion, FLUX and others) optimized with Pruna for faster inference. It includes an intelligent configurator that manages device compatibility (CUDA, CPU, Apple MPS), automatic fallbacks, and memory-aware compilation modes.

Key features:
- Download models from Hugging Face into `./models/`.
- Compile models with Pruna and store optimized artifacts in `./compiled_models/`.
- Expose a small HTTP API to download, compile, generate images and delete models.
- Auto-detection and recommended Pruna configuration per model and device.
- Multiple compilation modes: `fast`, `moderate`, `normal` to trade off speed vs quality.
- Device-aware fallbacks for MPS/CPU to avoid incompatible Pruna options.
- Diagnostic and memory-management helper scripts.

API Endpoints
- `POST /download` — download a model to `models/`.
- `POST /compile` — compile an existing downloaded model into `compiled_models/`.
- `POST /generate` — generate images with a compiled model.
- `POST /delete-model` — remove downloaded/compiled model folders.
- `GET /ping` — basic liveness check.
- `GET /health` — health and configuration report.

Quick CLI examples

1) Download a model (CLI):

```bash
python3 download_model_and_compile.py --model-id runwayml/stable-diffusion-v1-5
```

2) Download + compile (fast mode):

```bash
python3 download_model_and_compile.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --compilation-mode fast
```

3) Force CUDA compilation (if you have a GPU):

```bash
python3 force_cuda_compile.py --model-id runwayml/stable-diffusion-v1-5 --mode fast
```

4) Run the Flask API locally and test compile endpoint (example):

```bash
# start server in background
python3 server.py --host 127.0.0.1 --port 8000 --debug &

# request compilation (replace host/port if needed)
curl -X POST http://127.0.0.1:8000/compile \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "compilation_mode": "fast"}'

# stop server
pkill -f server.py
```

5) Generate images via API:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "prompt": "A scenic landscape at sunset"}'
```

Environment variables and Docker
- `MODEL_DIFF` — default model id used by CLI tools (override per-run or via Docker build arg).
- `DOWNLOAD_DIR` — path where models are downloaded (default `./models`).
- `PRUNA_COMPILED_DIR` — path where compiled models are written (default `./compiled_models`).

You can pass these as Docker build args or environment variables when running the container.

Practical tips
- Use `--skip-download` or `--skip-compile` when you want only one step.
- Prefer `fast` for quick iterations and `moderate/normal` for production-quality results.
- On Apple Silicon prefer `--device mps` and `fast` to avoid Pruna incompatibilities.
- If you see `CUDA out of memory` during compilation, use `restart_clean_compile.sh` or `compile_with_memory_mgmt.py`.

Credits
- Project maintainer: repository owner
- Major libraries and tools used:
  - Pruna (smash)
  - Hugging Face `diffusers` and `huggingface_hub`
  - PyTorch
  - Flask for the lightweight API

If you want a compact GitHub landing section (badges, short TL;DR, GIF) I can prepare a shorter front-page variant.