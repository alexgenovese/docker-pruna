<!-- Title and badges -->
# Docker Pruna — Download & Compile Diffusers Models

![CI](https://img.shields.io/badge/CI-pending-lightgrey) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![Docker](https://img.shields.io/badge/docker-ready-blue) ![License](https://img.shields.io/badge/license-Unspecified-lightgrey)

## Overview
Docker Pruna is a Docker-ready toolkit and lightweight Flask API to download, compile, and serve diffusion models (e.g., Stable Diffusion, FLUX) optimized with Pruna for faster inference. It includes:
- Configurable download and compilation pipelines
- Smart device-aware configuration (CUDA, CPU, MPS)
- Multiple compilation modes (fast, moderate, normal)
- Diagnostics and memory-aware helpers

This repository is a Docker-ready toolkit and a lightweight Flask API to download, compile and serve diffusion models (Stable Diffusion, FLUX and others) optimized with Pruna for faster inference. 

**It includes an intelligent configurator that manages device compatibility (CUDA, CPU, Apple MPS), automatic fallbacks, and memory-aware compilation modes.**

## TODO
- [x] Async Download Opt
- [ ] Push to Hub (compiled model)
- [ ] Qwen
- [ ] WAN 2.2

## Table of Contents
1. [Key Features](#key-features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Quick Start](#quick-start)
6. [API Endpoints](#api-endpoints)
7. [CLI Examples](#cli-examples)
8. [Compilation Modes](#compilation-modes)
9. [Diagnostics & Helper Scripts](#diagnostics--helper-scripts)
10. [File Layout](#file-layout)
11. [Docker Usage](#docker-usage)
12. [Troubleshooting](#troubleshooting)
13. [System Requirements](#system-requirements)
14. [Credits](#credits)

## Key Features
- Download models from Hugging Face into `./models/`
- Compile models with Pruna and store artifacts in `./compiled_models/`
- Lightweight Flask API for download, compile, generate, delete operations
- Smart `PrunaModelConfigurator` with device-aware fallback
- Compilation modes: `fast`, `moderate`, `normal`
- Helpers for CUDA/MPS/CPU diagnostics and memory-aware compilation

### New features
- `PrunaModelConfigurator` smart class
- Auto-detection for several model families
- Device-specific configuration recommendations
- Tests and diagnostic scripts


## Installation
```bash
git clone https://github.com/alexgenovese/docker-pruna.git
cd docker-pruna
pip install -r requirements.txt
```

## Configuration
### Environment Variables
- `MODEL_DIFF` — default model ID (default: `CompVis/stable-diffusion-v1-4`)
- `DOWNLOAD_DIR` — local models directory (default: `./models`)
- `PRUNA_COMPILED_DIR` — compiled models directory (default: `./compiled_models`)

### CLI Arguments
Run `python3 download_model_and_compile.py --help` to view options:
```bash
--model-id MODEL_ID        Hugging Face model ID
--download-dir DIR         Download directory
--compiled-dir DIR         Compiled models directory
--skip-download            Skip download step
--skip-compile             Skip compilation step
--torch-dtype TYPE         Torch dtype (float16/float32)
--compilation-mode MODE    fast|moderate|normal
--device DEVICE            cuda|cpu|mps
--force-cpu                Force CPU compilation
```

## Quick Start
Download and compile a model:
```bash
python3 download_model_and_compile.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --compilation-mode moderate
```
Start the API server:
```bash
python3 server.py --host 127.0.0.1 --port 8000 --debug &
```

Asynchronous downloads
----------------------
The API now runs potentially long-running downloads in a background task to avoid HTTP timeouts (eg. 524). When you POST to `/download` the server will immediately respond with a 202 Accepted and a `task_id` plus a `status_url` you can poll for progress and result.

Example (enqueue download):

```bash
curl -X POST http://127.0.0.1:8000/download \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"runwayml/stable-diffusion-v1-5"}'
```

Sample response:

```json
{ "status": "accepted", "task_id": "...", "status_url": "http://.../tasks/<task_id>" }
```

Poll the task status:

```bash
curl http://127.0.0.1:8000/tasks/<task_id>
```

The task JSON will include `status` (queued|running|finished|error) and, when finished, a `result` field with the downloaded model path or an `error` message.


## API Endpoints
All endpoints accept and return JSON.
| Method | Endpoint               | Description                          |
|--------|------------------------|--------------------------------------|
| POST   | `/download`            | Enqueue a model download (async)     |
| GET    | `/tasks/<task_id>`     | Get status/result for an async task  |
| POST   | `/compile`             | Compile a downloaded model           |
| POST   | `/generate`            | Generate images from a prompt        |
| POST   | `/delete-model`        | Delete downloaded/compiled model     |
| GET    | `/ping`                | Liveness check                       |
| GET    | `/health`              | Server health and configuration      |

Example — generate:
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"runwayml/stable-diffusion-v1-5","prompt":"A sunset"}'
```

## CLI Examples
- Download only: `python3 download_model_and_compile.py --model-id runwayml/stable-diffusion-v1-5`
- Download + compile (fast):
  ```bash
  python3 download_model_and_compile.py \
    --model-id runwayml/stable-diffusion-v1-5 \
    --compilation-mode fast
  ```
- Force CUDA compile:
  ```bash
  python3 force_cuda_compile.py --model-id runwayml/stable-diffusion-v1-5 --mode fast
  ```

## Compilation Modes
- **fast**: Quick development compile (DeepCache, half precision). Good for rapid iterations.
- **moderate**: Balanced speed and quality (TorchCompile + 8-bit HQQ)
- **normal**: Full optimizations (FORA, factorizer, autotune). Full optimizations for production, longer compile time.

Use `--compilation-mode` to pick the mode when running the CLI or API compile endpoint.

## Diagnostics & Helper Scripts
- `test_pruna_cuda.py` — CUDA and Pruna diagnostics
- `check_pruna_setup.py` — environment checks
- `compile_with_memory_mgmt.py` — memory-aware compilation
- `restart_clean_compile.sh` — clean GPU memory before compile

## File Layout
```text
lib/
├ pruna_config.py    Smart configurator
├ const.py           Constants
└ utils.py           Utilities

download_model_and_compile.py  Main download/compile CLI
download_model_and_compile.py  Main CLI
server.py                     Flask API server
*test_*.py                    Test and diagnostic scripts
*_compile.py                  Compilation helpers
```

## Docker Usage
Build the image:
```bash
docker build -t docker-pruna .
```
Run container:
```bash
docker run --rm -e MODEL_DIFF=runwayml/stable-diffusion-v1-5 docker-pruna
```

## Key features

- Download models from Hugging Face into `./models/`.
- Compile models with Pruna and store optimized artifacts in `./compiled_models/`.
- Lightweight Flask API to trigger download, compile, generate and delete operations.
- Smart `PrunaModelConfigurator` that provides device-aware, safe Pruna configurations and fallbacks.
- Compilation modes: `fast`, `moderate`, `normal` (speed vs quality trade-offs).
- Helpers for CUDA/MPS/CPU diagnostics and memory-aware compilation.

## Environment Variables

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

**POST /download**
- Enqueue a Hugging Face model download. The endpoint is asynchronous and returns a `task_id` and `status_url` you can poll.

See the "Asynchronous downloads" section above for examples.


**POST /compile**
- Compile an already downloaded model with Pruna and save into the compiled models directory.

Example:

```bash
curl -X POST http://127.0.0.1:8000/compile \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "compilation_mode" : "fast"}'
```

**POST /generate**
- Generate images from a prompt using a compiled model.

Example:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "prompt" : "A beautiful sunset over the ocean", "num_inference_steps": 20, "guidance_scale": 7.5}'
```

Response contains base64-encoded images and optional saved file paths and returns the url to downlaod the image when `debug: true`.

**POST /delete-model**
- Delete downloaded and/or compiled folders for a given model.

Example:
```bash
curl -X POST http://127.0.0.1:8000/delete-model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "type" : "all"}'
```

**GET /ping** — basic liveness check

**GET /health** — server configuration, system info, warnings and errors

## Run server and call compile endpoint (example):

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


# Single Files Explanation

## 1. Force CUDA explicitly
```bash
python3 force_cuda_compile.py --model-id MODEL_ID --mode fast
```

## 2. Use memory-aware compilation
```bash
python3 compile_with_memory_mgmt.py --model-id MODEL_ID --mode fast
```

## 3. Restart with clean memory
```bash
./restart_clean_compile.sh MODEL_ID fast
```

## 4. Set recommended env vars
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES=0
python3 download_model_and_compile.py --device cuda --model-id MODEL_ID
```

# Docker usage

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

## Quick CLI examples

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


# Practical tips
- Use `--skip-download` or `--skip-compile` when you want only one step.
- Prefer `fast` for quick iterations and `moderate/normal` for production-quality results.
- On Apple Silicon prefer `--device mps` and `fast` to avoid Pruna incompatibilities.
- If you see `CUDA out of memory` during compilation, use `restart_clean_compile.sh` or `compile_with_memory_mgmt.py`.


# Troubleshooting & common issues
| Issue | Affected models | Automatic fix |
|------|------------------|---------------|
| "Model is not compatible with fora" | SD 1.5, SD 1.4 | Switch to DeepCache instead of FORA |
| "deepcache is not compatible with device mps" | All on MPS | Disable DeepCache on MPS |
| Missing optional deps on MPS | HQQ & others | Disable HQQ on MPS |
| Missing packages | Various optimizations | Fallback to safe minimal config |


## Problem: "CUDA out of memory" during compilation
```
Cause: GPU memory is already occupied by other processes.
```
**Automatic fixes:**

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


# System requirements

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

## 🤝 Contributing

Contributions are welcome! Please fork, branch, and submit a pull request:

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

## 📄 License

This project is Apache 2.0 licensed. See [LICENSE](LICENSE) for details.