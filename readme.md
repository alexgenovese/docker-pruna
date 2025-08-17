# Docker Pruna - Download e Compilazione Modelli Diffusers

Questo progetto fornisce un ambiente Docker parametrizzabile per scaricare e compilare modelli diffusion con l'ottimizzazione Pruna per inferenze più veloci.

## 🚀 Caratteristiche Principali

- **Parametrizzabile**: Configura facilmente modello, directory di download e compilazione
- **Flessibile**: Supporta diversi modelli da Hugging Face
- **Ottimizzato**: Usa Pruna per accelerare le inferenze
- **Docker-ready**: Container pronto per produzione

## 📋 Parametri Configurabili

Il nuovo script `main.py` accetta questi parametri:

### Variabili d'Ambiente
- `MODEL_DIFF`: ID del modello su Hugging Face (default: `CompVis/stable-diffusion-v1-4`)
- `DOWNLOAD_DIR`: Directory per scaricare i modelli (default: `./models`)
- `PRUNA_COMPILED_DIR`: Directory per salvare i modelli compilati (default: `./compiled_models`)

### Argomenti CLI
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

## 🛠️ Utilizzo

### Docker Build con Parametri

```bash
# Build con modello di default
docker build -t docker-pruna .

# Build con modello personalizzato
docker build --build-arg MODEL_DIFF="runwayml/stable-diffusion-v1-5" -t docker-pruna .

# Build con variabili d'ambiente
docker build \
  --build-arg MODEL_DIFF="stabilityai/stable-diffusion-2-1" \
  --build-arg DOWNLOAD_DIR="/app/custom_models" \
  --build-arg PRUNA_COMPILED_DIR="/app/custom_compiled" \
  -t docker-pruna .
```

### Uso Locale

```bash
# Esempio 1: Verifica automatica - se il modello esiste, non fa nulla
python3 download_model_and_compile.py --model-id runwayml/stable-diffusion-v1-5

# Esempio 2: Scarica e compila un nuovo modello con modalità veloce  
python3 download_model_and_compile.py --model-id CompVis/stable-diffusion-v1-4 --compilation-mode fast

# Esempio 3: Usa cartelle personalizzate
python3 download_model_and_compile.py --model-id <MODEL_ID> --download-dir /custom/path --compiled-dir /custom/compiled

```

### Test

```bash
# Esegui tutti i test
./test_main.sh

# Test manuale del modello compilato
python3 test_pruna_infer.py
```

## 📁 Struttura Directory

```
/app/
├── main.py                    # Script principale parametrizzabile
├── models/                    # Modelli scaricati originali
│   └── CompVis--stable-diffusion-v1-4/
├── compiled_models/           # Modelli ottimizzati con Pruna
│   └── CompVis--stable-diffusion-v1-4/
├── test_pruna_infer.py       # Test di inferenza
└── test_main.sh              # Script di test
```

## 🔧 Configurazione Avanzata

### Dockerfile Personalizzato

Modifica il Dockerfile per cambiare i default:

```dockerfile
# Cambia modello di default
ENV MODEL_DIFF=runwayml/stable-diffusion-v1-5
ENV DOWNLOAD_DIR=/app/models  
ENV PRUNA_COMPILED_DIR=/app/compiled_models
```

### ComfyUI Integration

Il modello compilato sarà disponibile in `/app/compiled_models/` e può essere utilizzato direttamente dai nodi Pruna in ComfyUI configurando il path appropriato.

## 🎯 Cosa Succede all'Avvio Container

1. **Download**: Il modello specificato viene scaricato da Hugging Face
2. **Compilazione**: Pruna ottimizza il modello per inferenze più veloci  
3. **Ready**: Il modello ottimizzato è disponibile per l'uso immediato

L'inferenza parte istantaneamente senza compilazioni ripetute!

## 💡 Suggerimenti

- Usa `--skip-download` se hai già il modello scaricato
- Usa `--skip-compile` per solo scaricare il modello
- Sincronizza le versioni di PyTorch/Pruna tra build e runtime
- Considera multi-stage builds per container di produzione più piccoli

## 🚨 Requisiti

- CUDA 12.1+ per GPU support
- Python 3.8+
- Spazio disco sufficiente per modelli (3-7GB per modello)
- Token Hugging Face per modelli privati (opzionale)a for Faster inferences

# Docker Pruna - Download e Compilazione Modelli Diffusers con Configurazione Intelligente

Questo progetto fornisce un ambiente Docker parametrizzabile per scaricare e compilare modelli diffusion con l'ottimizzazione Pruna per inferenze più veloci. Include una **classe configuratore intelligente** che gestisce automaticamente la compatibilità tra modelli, dispositivi e ottimizzazioni Pruna.

## 🚀 Caratteristiche Principali

- **Configurazione Intelligente**: Gestione automatica della compatibilità Pruna per ogni modello
- **Multi-Modello**: Supporto ottimizzato per SD 1.5/XL/3.5, FLUX, Qwen, Wan
- **Multi-Dispositivo**: Configurazioni specifiche per CUDA, CPU e Apple Silicon (MPS)
- **Tre Modalità di Compilazione**: Fast, Moderate, Normal per bilanciare velocità vs qualità
- **Auto-Rilevamento**: Riconoscimento automatico del tipo di modello e compatibilità
- **Error-Free**: Evita errori di incompatibilità come "Model is not compatible with fora"
- **Docker-ready**: Container pronto per produzione

## 🚀 Quick Start

### Setup Iniziale
```bash
# 1. Clona e installa
git clone <your-repo>
cd docker-pruna
pip install -r requirements.txt

# 2. 🆕 Test sistema (RACCOMANDATO)
python3 test_pruna_cuda.py        # Diagnostica completa
python3 check_pruna_setup.py      # Verifica versioni
```

### Compilazione Standard
```bash
# Modalità automatica (gestisce tutto il sistema)
python3 download_model_and_compile.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --compilation-mode moderate
```

### 🆕 Risoluzione Problemi Immediata

#### Se hai problemi di memoria GPU:
```bash
# Soluzione automatica (RACCOMANDATO)
./restart_clean_compile.sh runwayml/stable-diffusion-v1-5 fast
```

#### Se Pruna compila per CPU invece di GPU:
```bash
# Forza CUDA
python3 force_cuda_compile.py --model-id runwayml/stable-diffusion-v1-5 --mode fast
```

#### Per diagnostica setup:
```bash
# Verifica completa ambiente
python3 test_pruna_cuda.py
```

## 🧠 Nuova Classe PrunaModelConfigurator

La classe `PrunaModelConfigurator` in `lib/pruna_config.py` gestisce automaticamente:

### 🔍 **Riconoscimento Modelli Supportati**
- **Stable Diffusion 1.5**: `runwayml/stable-diffusion-v1-5`, `CompVis/stable-diffusion-v1-4`
- **Stable Diffusion XL**: `stabilityai/stable-diffusion-xl-base-1.0` 
- **Stable Diffusion 3.5**: `stabilityai/stable-diffusion-3.5-large`
- **FLUX**: `black-forest-labs/FLUX.1-dev`
- **Qwen**: `Qwen/Qwen2-7B`
- **Wan**: Modelli Wan Labs
- **Generici**: Fallback sicuro per modelli non riconosciuti

### 💻 **Compatibilità Dispositivi**

| Caratteristica | CUDA | CPU | MPS (Apple) |
|----------------|------|-----|-------------|
| FORA Cacher | ✅ (solo SDXL/FLUX) | ⚠️ (limitato) | ❌ |
| DeepCache | ✅ | ✅ | ❌ |
| Factorizer | ✅ | ⚠️ (escluso FLUX) | ❌ |
| TorchCompile | ✅ | ✅ | ❌ |
| HQQ Quantizer | ✅ | ✅ | ⚠️ (solo SD) |
| TorchAO Backend | ✅ | ❌ | ❌ |

### 🛡️ **Protezione dagli Errori**
- **NO più "Model is not compatible with fora"** per SD 1.5
- **NO più "deepcache is not compatible with device mps"** su Apple Silicon
- **Configurazioni ultra-minimali** per MPS che evitano tutti i problemi di compatibilità
- **Fallback automatici** per combinazioni non supportate

## 🎛️ Modalità di Compilazione Intelligenti

### 🚀 Fast (Veloce)
**Uso**: Prototipazione rapida, test, sviluppo
- **CUDA/CPU**: DeepCache + quantizzazione half
- **MPS**: Configurazione minimale device-only
- **Tempo**: ~5-10 minuti
- **Qualità**: Buona, perdita minima

### ⚖️ Moderate (Moderata)  
**Uso**: Produzione bilanciata, deploy standard
- **CUDA**: DeepCache + TorchCompile + HQQ 8-bit
- **CPU**: Configurazione simile senza TorchAO
- **MPS**: Solo configurazione device-safe
- **Tempo**: ~15-25 minuti
- **Qualità**: Ottima, rapporto ideale

### 🎯 Normal (Completa)
**Uso**: Massima qualità, produzione critica
- **CUDA**: FORA + Factorizer + TorchCompile max-autotune + HQQ 4-bit
- **CPU**: DeepCache + ottimizzazioni conservative
- **MPS**: Configurazione ultra-sicura
- **SD 1.5**: USA DeepCache invece di FORA (risolve incompatibilità)
- **Tempo**: ~30-60 minuti
- **Qualità**: Massima


## 📋 Parametri Configurabili

### Variabili d'Ambiente
- `MODEL_DIFF`: ID del modello su Hugging Face (default: `CompVis/stable-diffusion-v1-4`)
- `DOWNLOAD_DIR`: Directory per scaricare i modelli (default: `./models`)
```markdown
### /delete-model
Removes the downloaded or compiled model folder (or both) for a given `model_id`.

**POST /delete-model**

**Body JSON:**
```json
{
  "model_id": "runwayml/stable-diffusion-v1-5",
  "type": "all" // Optional: "downloaded", "compiled" or "all" (default)
}
```

**CURL example:**
```bash
curl -X POST http://localhost:8000/delete-model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "type": "all"}'
```

**Response:**
- `status`: "success" if all folders were deleted, "partial" if only some, "not_found" if none found, "error" on internal error.
- `deleted`: list of deleted folders.
- `errors`: any errors encountered.

---
# Docker Pruna - Download and Compile Diffusers Models

This project provides a configurable Docker environment and CLI to download and compile diffusion models with Pruna optimizations for faster inference.

## 🚀 Main Features

- Configurable: easily set model id, download and compiled directories
- Flexible: supports multiple Hugging Face models
- Optimized: uses Pruna to accelerate inference
- Docker-ready: container-friendly workflow

## 📋 Configurable Parameters

The main CLI script accepts these parameters.

### Environment Variables
- `MODEL_DIFF`: Hugging Face model ID (default: `CompVis/stable-diffusion-v1-4`)
- `DOWNLOAD_DIR`: Directory for downloaded models (default: `./models`)
- `PRUNA_COMPILED_DIR`: Directory for compiled Pruna models (default: `./compiled_models`)

### CLI Arguments
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

## 🛠️ Usage

### Docker Build Examples

```bash
# Build with default model
docker build -t docker-pruna .

# Build with a custom model
docker build --build-arg MODEL_DIFF="runwayml/stable-diffusion-v1-5" -t docker-pruna .

# Build with custom directories
docker build \
  --build-arg MODEL_DIFF="stabilityai/stable-diffusion-2-1" \
  --build-arg DOWNLOAD_DIR="/app/custom_models" \
  --build-arg PRUNA_COMPILED_DIR="/app/custom_compiled" \
  -t docker-pruna .
```

### Local Usage

```bash
# Example 1: Auto-check — does nothing if the model is already present
python3 download_model_and_compile.py --model-id runwayml/stable-diffusion-v1-5

# Example 2: Download and compile with fast compilation mode
python3 download_model_and_compile.py --model-id CompVis/stable-diffusion-v1-4 --compilation-mode fast

# Example 3: Use custom folders
python3 download_model_and_compile.py --model-id <MODEL_ID> --download-dir /custom/path --compiled-dir /custom/compiled
```

### Tests

```bash
# Run all tests
./test_main.sh

# Manual inference test
python3 test_pruna_infer.py
```

## 📁 Directory Structure

```
/app/
├── main.py                    # Main parameterizable script
├── models/                    # Downloaded original models
│   └── CompVis--stable-diffusion-v1-4/
├── compiled_models/           # Pruna-optimized models
│   └── CompVis--stable-diffusion-v1-4/
├── test_pruna_infer.py        # Inference test
└── test_main.sh               # Test script
```

## 🔧 Advanced Configuration

### Customizing Dockerfile

Modify the Dockerfile to change defaults:

```dockerfile
# Change default model
ENV MODEL_DIFF=runwayml/stable-diffusion-v1-5
ENV DOWNLOAD_DIR=/app/models
ENV PRUNA_COMPILED_DIR=/app/compiled_models
```

### ComfyUI Integration

Compiled models live in `/app/compiled_models/` and can be consumed by ComfyUI Pruna nodes by pointing ComfyUI to that path.

## 🎯 What Happens When the Container Starts

1. Download: the specified model is downloaded from Hugging Face.
2. Compile: Pruna optimizes the model for faster inference.
3. Ready: the optimized model is available for immediate use.

Inference can start without repeated compilations.

## 💡 Tips

- Use `--skip-download` if you already have the model downloaded.
- Use `--skip-compile` to only download the model.
- Align PyTorch/Pruna versions between build and runtime.
- Consider multi-stage Docker builds for smaller production images.

## 🚨 Requirements

- CUDA 12.1+ for GPU support
- Python 3.8+
- Enough disk space for models (3-7GB per model)
- Hugging Face token for private models (optional)

# Docker Pruna — Diffusers Models with Smart Pruna Configuration

This repository adds a smart configurator that automatically handles compatibility between models, devices and Pruna optimizations.

## 🚀 Highlights

- Smart configuration: automatic Pruna compatibility per model
- Multi-model support: SD 1.5/XL/3.5, FLUX, Qwen, Wan and others
- Multi-device support: CUDA, CPU and Apple Silicon (MPS)
- Three compilation modes: Fast, Moderate, Normal (speed vs quality)
- Auto-detection: model type and compatibility detection
- Robustness: avoids common incompatibilities (e.g. "Model is not compatible with fora")
- Docker-friendly

## 🚀 Quick Start

### Initial Setup
```bash
# 1. Clone and install
git clone <your-repo>
cd docker-pruna
pip install -r requirements.txt

# 2. Optional system checks (recommended)
python3 test_pruna_cuda.py        # CUDA/Pruna diagnostics
python3 check_pruna_setup.py      # Check versions and deps
```

### Standard Compilation
```bash
# Automatic mode (handles whole workflow)
python3 download_model_and_compile.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --compilation-mode moderate
```

### Quick Troubleshooting

#### If you hit GPU memory issues:
```bash
# Recommended automatic restart + clean
./restart_clean_compile.sh runwayml/stable-diffusion-v1-5 fast
```

#### If Pruna compiles for CPU while you have a GPU:
```bash
# Force CUDA compilation
python3 force_cuda_compile.py --model-id runwayml/stable-diffusion-v1-5 --mode fast
```

#### To run diagnostics:
```bash
python3 test_pruna_cuda.py
```

## 🧠 PrunaModelConfigurator

`PrunaModelConfigurator` (in `lib/pruna_config.py`) handles model detection and produces safe, device-aware Pruna smash configurations.

### Supported model types
- Stable Diffusion 1.5: `runwayml/stable-diffusion-v1-5`, `CompVis/stable-diffusion-v1-4`
- Stable Diffusion XL: `stabilityai/stable-diffusion-xl-base-1.0`
- Stable Diffusion 3.5: `stabilityai/stable-diffusion-3.5-large`
- FLUX: `black-forest-labs/FLUX.1-dev`
- Qwen: `Qwen/Qwen2-7B`
- Wan models and other generics (fallback)

### Device compatibility table (summary)

| Feature | CUDA | CPU | MPS (Apple) |
|--------:|:----:|:---:|:-----------:|
| FORA Cacher | ✅ (SDXL/FLUX) | ⚠️ limited | ❌ |
| DeepCache | ✅ | ✅ | ❌ |
| Factorizer | ✅ | ⚠️ | ❌ |
| TorchCompile | ✅ | ✅ | ❌ |
| HQQ Quantizer | ✅ | ✅ | ⚠️ (SD only) |
| TorchAO Backend | ✅ | ❌ | ❌ |

### Safety & fallbacks

- Avoids "Model is not compatible with fora" for SD 1.5 by switching to DeepCache.
- Disables DeepCache and other incompatible options on MPS.
- Provides conservative fallbacks for devices with limited support.

## 🎛️ Compilation Modes

### 🚀 Fast
Use for rapid prototyping and testing.
- CUDA/CPU: DeepCache + half quantization
- MPS: minimal device-only config
- Time: ~5-10 minutes
- Quality: good

### ⚖️ Moderate
Balanced production mode.
- CUDA: DeepCache + TorchCompile + HQQ 8-bit
- CPU: similar without TorchAO
- Time: ~15-25 minutes
- Quality: high

### 🎯 Normal
Max quality (longer), intended for critical production.
- CUDA: FORA + Factorizer + TorchCompile autotune + HQQ 4-bit
- CPU: DeepCache with conservative opts
- MPS: ultra-safe config
- Time: ~30-60 minutes

## 📋 Configurable Parameters (CLI & ENV)

### Environment variables
- `MODEL_DIFF` — default model id
- `DOWNLOAD_DIR` — where models are stored
- `PRUNA_COMPILED_DIR` — where compiled models are written

### CLI flags
```bash
python3 download_model_and_compile.py --help

optional arguments:
  --model-id MODEL_ID              Hugging Face model ID to download
  --download-dir DIR               Directory to download models
  --compiled-dir DIR               Directory to save compiled Pruna models
  --skip-download                  Skip download step (use existing model)
  --skip-compile                   Skip compilation step (only download)
  --torch-dtype TYPE               Torch dtype for model loading (float16/float32)
  --compilation-mode MODE          Pruna compilation mode (fast/moderate/normal)
  --force-cpu                      Force CPU usage for compilation
  --device DEVICE                  Override device selection (auto/cuda/mps/cpu)
```

## 🔧 Programmatic Usage

```python
from lib.pruna_config import PrunaModelConfigurator

# Create configurator
configurator = PrunaModelConfigurator()

# Detect model type
model_type = configurator.detect_model_type("runwayml/stable-diffusion-v1-5")
print(f"Type: {model_type}")  # e.g. stable-diffusion-1.5

# Get model info
info = configurator.get_model_info("runwayml/stable-diffusion-v1-5")
print(f"Device: {info['device']}")
print(f"FORA compatibility: {info['compatibility']['fora_cacher']}")

# Generate an optimized smash config
config = configurator.get_smash_config(
    model_id="runwayml/stable-diffusion-v1-5",
    compilation_mode="moderate",
    device="auto"
)
```

### Configuration tests
```bash
# Validate configurator recommendations for multiple models
python3 test_pruna_config.py
```

## REST API Endpoints

### /download
Download a Hugging Face model into `models/` if not already present.

**POST /download**

**Body JSON:**
```json
{
  "model_id": "runwayml/stable-diffusion-v1-5"
}
```

**CURL example:**
```bash
curl -X POST http://localhost:8000/download \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5"}'
```

---

### /compile
Compile a previously downloaded model with Pruna and save to `compiled_models/`. Returns an error if the model is not available.

**POST /compile**

**Body JSON:**
```json
{
  "model_id": "runwayml/stable-diffusion-v1-5",
  "compilation_mode": "moderate"  // Optional: fast, moderate, normal
}
```

**CURL example:**
```bash
curl -X POST http://localhost:8000/compile \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "compilation_mode": "fast"}'
```

---

### /generate
Generate images from a prompt using a compiled model.

**POST /generate**

**Body JSON (example):**
```json
{
  "model_id": "runwayml/stable-diffusion-v1-5",
  "prompt": "A beautiful sunset over the ocean"
}
```

**CURL example:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "prompt": "A beautiful sunset over the ocean"}'
```

---

### /ping
Check server liveness.

**GET /ping**

**CURL example:**
```bash
curl http://localhost:8000/ping
```

---

### /health
Return health and configuration details about the server.

**GET /health**

**CURL example:**
```bash
curl http://localhost:8000/health
```

---

**Note:**
- All endpoints return JSON responses.
- Change the port (`8000`) if the server runs on a different port.

#### Supported parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | string | **required** | Model identifier (folder name in compiled_models/) |
| `prompt` | string | **required** | Generation prompt |
| `negative_prompt` | string | `""` | Negative prompt (what to avoid) |
| `num_inference_steps` | int | `20` | Denoising steps (10-100) |
| `guidance_scale` | float | `7.5` | Prompt adherence (1.0-20.0) |
| `width` | int | `512` | Image width (multiple of 8) |
| `height` | int | `512` | Image height (multiple of 8) |
| `num_images` | int | `1` | Number of images to generate |
| `debug` | bool | `false` | Save images locally in `./generated_images/` |

#### Example API response

```json
{
  "status": "success",
  "images": ["base64_encoded_image_1", "base64_encoded_image_2"],
  "prompt": "A magical forest with glowing trees",
  "model_used": "stable-diffusion-v1-5",
  "inference_time": 2.34,
  "saved_files": ["./generated_images/20250813_143022_img_00.png"]
}
```

**Notes:**
- Images are returned as base64 strings in the `images` field.
- With `debug: true`, generated images are saved in `./generated_images/`.
- The server uses Pruna-compiled models automatically when available for faster inference.
- Typical response time varies (1-5 seconds) depending on hardware and parameters.

## 🛠️ Practical Usage

### Compilation examples

```bash
# 1. Stable Diffusion 1.5 (auto-fallback avoids FORA incompatibility)
python3 download_model_and_compile.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --compilation-mode normal

# 2. FLUX quick mode for testing
python3 download_model_and_compile.py \
  --model-id black-forest-labs/FLUX.1-dev \
  --compilation-mode fast

# 3. SDXL full optimizations on CUDA
python3 download_model_and_compile.py \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --compilation-mode normal \
  --device cuda

# 4. Safe compile on Apple Silicon
python3 download_model_and_compile.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --compilation-mode moderate \
  --device mps

# 5. Only download
python3 download_model_and_compile.py \
  --model-id CompVis/stable-diffusion-v1-4 \
  --skip-compile

# 6. Only compile an existing model
python3 download_model_and_compile.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --skip-download \
  --compilation-mode fast
```

### New helper scripts

#### CUDA / Pruna diagnostics
```bash
# Full CUDA / Pruna diagnostics
python3 test_pruna_cuda.py

# Check versions and dependencies
python3 check_pruna_setup.py
```

#### Force CUDA compile
```bash
python3 force_cuda_compile.py --model-id runwayml/stable-diffusion-v1-5 --mode fast

# Modes: fast, moderate, normal
python3 force_cuda_compile.py --model-id black-forest-labs/FLUX.1-dev --mode moderate
```

#### Memory-aware compilation
```bash
python3 compile_with_memory_mgmt.py --model-id runwayml/stable-diffusion-v1-5 --mode fast

./restart_clean_compile.sh runwayml/stable-diffusion-v1-5 fast
```

### Advanced troubleshooting

#### Problem: "Compiling for CPU" even if CUDA is available

Cause: conservative auto-detection by Pruna or insufficient GPU memory.

Solutions:
```bash
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