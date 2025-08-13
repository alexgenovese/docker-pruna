#!/bin/bash

# Test specifico per macOS con supporto MPS
# Questo script testa diverse configurazioni per trovare quella più compatibile

echo "🍎 Test di compatibilità per macOS/MPS"
echo "========================================="

# Test 1: Modalità fast con device auto-detect
echo "📝 Test 1: Modalità FAST con auto-detection"
python3 main.py \
    --model-id "CompVis/stable-diffusion-v1-4" \
    --download-dir "./models/stable-diffusion-v1-4" \
    --compiled-dir "./compiled_models/stable-diffusion-v1-4-fast" \
    --compilation-mode fast \
    --skip-download

if [ $? -eq 0 ]; then
    echo "✅ Test 1 completato con successo!"
else
    echo "❌ Test 1 fallito, provo con CPU..."
    
    # Test 2: Modalità fast con CPU forzata
    echo "📝 Test 2: Modalità FAST con CPU forzata"
    python3 main.py \
        --model-id "CompVis/stable-diffusion-v1-4" \
        --download-dir "./models/stable-diffusion-v1-4" \
        --compiled-dir "./compiled_models/stable-diffusion-v1-4-fast-cpu" \
        --compilation-mode fast \
        --force-cpu \
        --skip-download
    
    if [ $? -eq 0 ]; then
        echo "✅ Test 2 (CPU) completato con successo!"
    else
        echo "❌ Test 2 fallito anche con CPU"
    fi
fi

echo "🏁 Test completati"
