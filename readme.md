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
# Scarica e compila modello di default
python3 main.py

# Modello personalizzato
python3 main.py --model-id "runwayml/stable-diffusion-v1-5"

# Solo download
python3 main.py --skip-compile

# Solo compilazione (modello esistente)
python3 main.py --skip-download

# Con variabili d'ambiente
export MODEL_DIFF="stabilityai/stable-diffusion-2-1"
export DOWNLOAD_DIR="./my_models"
export PRUNA_COMPILED_DIR="./my_compiled"
python3 main.py
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

# Docker Pruna - Download e Compilazione Modelli Diffusers

Questo progetto fornisce un ambiente Docker parametrizzabile per scaricare e compilare modelli diffusion con l'ottimizzazione Pruna per inferenze più veloci.

## 🚀 Caratteristiche Principali

- **Parametrizzabile**: Configura facilmente modello, directory di download e compilazione
- **Tre Modalità di Compilazione**: Fast, Moderate, Normal per bilanciare velocità vs qualità
- **Flessibile**: Supporta diversi modelli da Hugging Face
- **Ottimizzato**: Usa Pruna per accelerare le inferenze
- **Docker-ready**: Container pronto per produzione

## 📋 Parametri Configurabili

Il script `main.py` accetta questi parametri:

### Variabili d'Ambiente
- `MODEL_DIFF`: ID del modello su Hugging Face (default: `CompVis/stable-diffusion-v1-4`)
- `DOWNLOAD_DIR`: Directory per scaricare i modelli (default: `./models`)
- `PRUNA_COMPILED_DIR`: Directory per salvare i modelli compilati (default: `./compiled_models`)

### Argomenti CLI
```bash
python3 main.py --help

optional arguments:
  --model-id MODEL_ID              Hugging Face model ID to download
  --download-dir DIR               Directory to download models
  --compiled-dir DIR               Directory to save compiled Pruna models
  --skip-download                  Skip download step (use existing model)
  --skip-compile                   Skip compilation step (only download)
  --torch-dtype TYPE               Torch dtype for model loading (float16/float32)
  --compilation-mode MODE          Pruna compilation mode (fast/moderate/normal)
```

## 🎛️ Modalità di Compilazione

### 🚀 Fast (Veloce)
**Uso**: Prototipazione rapida, test, sviluppo
- **Caching**: DeepCache con interval 3 (più veloce)
- **Compiler**: stable_fast (ottimizzazioni rapide)
- **Quantization**: half (FP16, leggera)
- **Tempo**: ~5-10 minuti
- **Qualità**: Buona, perdita minima di qualità

### ⚖️ Moderate (Moderata)
**Uso**: Produzione bilanciata, deploy standard
- **Caching**: DeepCache con interval 2 (bilanciato)
- **Compiler**: torch_compile con mode default
- **Quantization**: HQQ 8-bit con group size 64
- **Tempo**: ~15-25 minuti
- **Qualità**: Ottima, rapporto qualità/velocità ideale

### 🎯 Normal (Completa)
**Uso**: Massima qualità, produzione critica
- **Caching**: FORA avanzato con factorization
- **Factorizer**: QKV diffusers per ottimizzazioni complete
- **Compiler**: torch_compile con max-autotune
- **Quantization**: HQQ 4-bit con backend torchao_int4
- **Tempo**: ~30-60 minuti
- **Qualità**: Massima, ottimizzazioni complete

## 🛠️ Utilizzo

### Docker Build con Parametri

```bash
# Build con modello di default (modalità normal)
docker build -t docker-pruna .

# Build con modalità veloce
docker build --build-arg COMPILATION_MODE=fast -t docker-pruna .

# Build con modello personalizzato e modalità moderata
docker build 
  --build-arg MODEL_DIFF="runwayml/stable-diffusion-v1-5" 
  --build-arg COMPILATION_MODE=moderate 
  -t docker-pruna .
```

### Uso Locale

```bash
# Modalità normale (default)
python3 main.py

# Modalità veloce
python3 main.py --compilation-mode fast

# Modalità moderata con modello personalizzato
python3 main.py --model-id "runwayml/stable-diffusion-v1-5" --compilation-mode moderate

# Solo download
python3 main.py --skip-compile

# Solo compilazione modalità veloce (modello esistente)
python3 main.py --skip-download --compilation-mode fast

# Con variabili d'ambiente
export MODEL_DIFF="stabilityai/stable-diffusion-2-1"
export DOWNLOAD_DIR="./my_models"
export PRUNA_COMPILED_DIR="./my_compiled"
python3 main.py --compilation-mode moderate
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

## 🎯 Cosa Succede all'Avvio Container

1. **Download**: Il modello specificato viene scaricato da Hugging Face
2. **Compilazione**: Pruna ottimizza il modello secondo la modalità scelta
3. **Ready**: Il modello ottimizzato è disponibile per l'uso immediato

L'inferenza parte istantaneamente senza compilazioni ripetute!

## 🔧 Configurazione Avanzata

### ComfyUI Integration

Il modello compilato sarà disponibile in `/compiled_models/` e può essere utilizzato direttamente dai nodi Pruna in ComfyUI configurando il path appropriato.

### Dockerfile Personalizzato

Modifica il Dockerfile per cambiare i default:

```dockerfile
# Cambia modello e modalità di default
ENV MODEL_DIFF=runwayml/stable-diffusion-v1-5
ENV DOWNLOAD_DIR=/app/models  
ENV PRUNA_COMPILED_DIR=/app/compiled_models
```

## 💡 Suggerimenti

### Scegliere la Modalità Giusta

- **Fast**: Per sviluppo, test rapidi, prototipazione
- **Moderate**: Per la maggior parte dei casi di produzione
- **Normal**: Per applicazioni critiche dove la qualità è prioritaria

### Performance Tips

- Usa `--skip-download` se hai già il modello scaricato
- Usa `--skip-compile` per solo scaricare il modello  
- Sincronizza le versioni di PyTorch/Pruna tra build e runtime
- Considera multi-stage builds per container di produzione più piccoli
- La modalità Fast può accelerare la compilazione del 70-80%
- La modalità Normal offre fino al 30% di miglioramento nelle prestazioni di inferenza

## 🚨 Requisiti

- CUDA 12.1+ per GPU support (richiesto per stable_fast e HQQ)
- Python 3.8+
- Spazio disco sufficiente per modelli (3-7GB per modello)
- Token Hugging Face per modelli privati (opzionale)
- 8GB+ RAM per modalità Normal con modelli grandi

## 📊 Confronto Modalità

| Modalità | Tempo Compilazione | Qualità Output | Velocità Inferenza | Uso Consigliato |
|----------|-------------------|----------------|-------------------|------------------|
| Fast     | ⭐⭐⭐⭐⭐            | ⭐⭐⭐⭐          | ⭐⭐⭐⭐              | Sviluppo, Test   |
| Moderate | ⭐⭐⭐              | ⭐⭐⭐⭐⭐        | ⭐⭐⭐⭐⭐            | Produzione       |
| Normal   | ⭐                 | ⭐⭐⭐⭐⭐        | ⭐⭐⭐⭐⭐            | Critico          |

## Suggerimenti

Se vuoi integrare tutto nella pipeline ComfyUI, aggiungi le estensioni/plugin Pruna con COPY . e assicurati che nella tua UI si possa configurare il path /compiled_models/.

Sincronizza le versioni di torch/Pruna tra build e run.

Puoi dividere build & run in multistage Docker se necessario, ma in casi semplici lo script sopra funziona direttamente.

Così ottieni un container "ready-to-go": il modello viene scaricato, compilato da Pruna, e ComfyUI/Pruna Node reimpiega direttamente la compilazione per le inferenze.