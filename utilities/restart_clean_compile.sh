#!/bin/bash
# Script to restart compilation with cleaned GPU memory

set -euo pipefail

MODEL_ID_DEFAULT="runwayml/stable-diffusion-v1-5"
MODE_DEFAULT="fast"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.0;7.5;8.0;8.6"}

print_header() {
    echo "🧹 CLEANING GPU MEMORY AND RESTARTING COMPILATION"
    echo "============================================="
}

show_inputs() {
        cat <<EOF
Variables/arguments you can pass to this script:

Environment variables (export before running) - default values used if not present:
    PYTORCH_CUDA_ALLOC_CONF = ${PYTORCH_CUDA_ALLOC_CONF}
    CUDA_VISIBLE_DEVICES      = ${CUDA_VISIBLE_DEVICES}
    TORCH_CUDA_ARCH_LIST      = ${TORCH_CUDA_ARCH_LIST}

Positional arguments (passed to the script):
    MODEL_ID  (default: ${MODEL_ID_DEFAULT})  -> model id to compile (e.g.: runwayml/stable-diffusion-v1-5)
    MODE      (default: ${MODE_DEFAULT})      -> compilation mode (e.g.: fast, full, debug)

Invocation examples:
    ./restart_clean_compile.sh                      # show interactive menu
    ./restart_clean_compile.sh ${MODEL_ID_DEFAULT} ${MODE_DEFAULT}  # run cleanup + compilation with args
    MODEL_ID=my/model CUDA_VISIBLE_DEVICES=1 ./restart_clean_compile.sh my/model fast

EOF
}

gpu_status() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader,nounits || true
    else
        echo "⚠️  nvidia-smi not available"
    fi
}

perform_cleanup() {
    echo "🔪 Terminating existing Python processes..."
    pkill -f "python.*server.py" 2>/dev/null || true
    pkill -f "python.*download_model_and_compile.py" 2>/dev/null || true
    sleep 2

    if command -v nvidia-smi &>/dev/null; then
        echo "🧹 Attempting GPU reset (if supported)..."
        nvidia-smi --gpu-reset-clocks=0 2>/dev/null || true
    fi

    echo "⏳ Waiting 3 seconds for stabilization..."
    sleep 3
}

perform_compile() {
    local model_id="$1"
    local mode="$2"

    echo "🚀 Starting compilation for model=$model_id mode=$mode"

    if [ -f "compile_with_memory_mgmt.py" ]; then
        echo "   Using memory-optimized script..."
        python3 compile_with_memory_mgmt.py --model-id "$model_id" --mode "$mode"
    else
        echo "   Using standard script with optimized parameters..."
        python3 download_model_and_compile.py \
            --model-id "$model_id" \
            --compilation-mode "$mode" \
            --device cuda \
            --skip-download
    fi
}

run_full() {
    local model_id=${1:-${MODEL_ID_DEFAULT}}
    local mode=${2:-${MODE_DEFAULT}}

    print_header
    echo "📊 GPU status before cleanup:"
    gpu_status || true
    echo ""

    perform_cleanup

    echo "📊 GPU status after cleanup:"
    gpu_status || true
    echo ""

    perform_compile "$model_id" "$mode"

    echo ""
    echo "✅ Process completed!"

    echo "📊 Final GPU status:"
    gpu_status || true
}

print_menu() {
        cat <<MENU
Choose an option:
    1) Show input variables and examples
    2) GPU Cleanup + Compilation (default)
    3) GPU Cleanup only
    4) Compilation only (you will be prompted for MODEL_ID and MODE)
    5) Exit
MENU
}

# If arguments provided and not explicitly asking for menu/help, run non-interactive
if [ "$#" -ge 1 ] && [ "$1" != "--menu" ] && [ "$1" != "-m" ] && [ "$1" != "--help" ] && [ "$1" != "-h" ]; then
    MODEL_ID="${1:-${MODEL_ID_DEFAULT}}"
    MODE="${2:-${MODE_DEFAULT}}"
    run_full "$MODEL_ID" "$MODE"
    exit 0
fi

# If user requested help, or no args, show interactive menu
if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
    show_inputs
    exit 0
fi

print_header

while true; do
    print_menu
    read -rp "Select (1-5): " choice
    case "$choice" in
        1)
            show_inputs
            ;;
        2)
            read -rp "MODEL_ID [${MODEL_ID_DEFAULT}]: " m
            m=${m:-${MODEL_ID_DEFAULT}}
            read -rp "MODE [${MODE_DEFAULT}]: " mm
            mm=${mm:-${MODE_DEFAULT}}
            run_full "$m" "$mm"
            ;;
        3)
            echo "📊 GPU status before cleanup:"
            gpu_status || true
            echo ""
            perform_cleanup
            echo "📊 GPU status after cleanup:"
            gpu_status || true
            ;;
        4)
            read -rp "MODEL_ID [${MODEL_ID_DEFAULT}]: " m2
            m2=${m2:-${MODEL_ID_DEFAULT}}
            read -rp "MODE [${MODE_DEFAULT}]: " mm2
            mm2=${mm2:-${MODE_DEFAULT}}
            perform_compile "$m2" "$mm2"
            ;;
        5)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo "Invalid option. Try again."
            ;;
    esac
    echo ""
done