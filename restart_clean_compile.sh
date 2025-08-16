#!/bin/bash
"""
Script per riavviare la compilazione con memoria GPU pulita
"""

echo "🧹 PULIZIA MEMORIA GPU E RIAVVIO COMPILAZIONE"
echo "============================================="

# Controlla se NVIDIA-SMI è disponibile
if command -v nvidia-smi &> /dev/null; then
    echo "📊 Stato GPU prima della pulizia:"
    nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️  nvidia-smi non disponibile"
fi

# Termina processi Python che potrebbero usare GPU
echo "🔪 Terminazione processi Python esistenti..."
pkill -f "python.*server.py" 2>/dev/null || true
pkill -f "python.*download_model_and_compile.py" 2>/dev/null || true
sleep 2

# Pulisce cache GPU se disponibile
if command -v nvidia-smi &> /dev/null; then
    echo "🧹 Reset GPU..."
    nvidia-smi --gpu-reset-clocks=0 2>/dev/null || true
fi

echo "⏳ Attesa 3 secondi per stabilizzazione..."
sleep 3

# Controlla memoria dopo pulizia
if command -v nvidia-smi &> /dev/null; then
    echo "📊 Stato GPU dopo pulizia:"
    nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader,nounits
    echo ""
fi

# Impostazioni ottimali per memoria
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

echo "🚀 Avvio compilazione con gestione memoria ottimizzata..."

# Determina quale script usare
MODEL_ID="${1:-runwayml/stable-diffusion-v1-5}"
MODE="${2:-fast}"

if [ -f "compile_with_memory_mgmt.py" ]; then
    echo "   Usando script ottimizzato per memoria..."
    python3 compile_with_memory_mgmt.py --model-id "$MODEL_ID" --mode "$MODE"
else
    echo "   Usando script standard con parametri ottimizzati..."
    python3 download_model_and_compile.py \
        --model-id "$MODEL_ID" \
        --compilation-mode "$MODE" \
        --device cuda \
        --skip-download
fi

echo ""
echo "✅ Processo completato!"

# Stato finale GPU
if command -v nvidia-smi &> /dev/null; then
    echo "📊 Stato finale GPU:"
    nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader,nounits
fi
