#!/usr/bin/env python3
"""
Script principale per il download e la compilazione di modelli diffusers con Pruna.
Parametrizzabile tramite variabili d'ambiente o argomenti CLI.
"""
import os
import sys
import argparse
import torch
from pathlib import Path
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from pruna import SmashConfig, smash


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download and compile diffusion models with Pruna')
    parser.add_argument('--model-id', 
                       default=os.environ.get('MODEL_DIFF', 'CompVis/stable-diffusion-v1-4'),
                       help='Hugging Face model ID to download (default: %(default)s)')
    parser.add_argument('--download-dir', 
                       default=os.environ.get('DOWNLOAD_DIR', './models'),
                       help='Directory to download models (default: %(default)s)')
    parser.add_argument('--compiled-dir', 
                       default=os.environ.get('PRUNA_COMPILED_DIR', './compiled_models'),
                       help='Directory to save compiled Pruna models (default: %(default)s)')
    parser.add_argument('--skip-download', 
                       action='store_true',
                       help='Skip download step (use existing model)')
    parser.add_argument('--skip-compile',
                       action='store_true', 
                       help='Skip compilation step (only download)')
    parser.add_argument('--torch-dtype',
                       default='float16',
                       choices=['float16', 'float32'],
                       help='Torch dtype for model loading (default: %(default)s)')
    parser.add_argument('--compilation-mode',
                       default='normal',
                       choices=['fast', 'moderate', 'normal'],
                       help='Pruna compilation mode - fast: speed over quality, moderate: balanced, normal: quality over speed (default: %(default)s)')
    parser.add_argument('--force-cpu',
                       action='store_true',
                       help='Force CPU usage for compilation (useful if GPU has compatibility issues)')
    parser.add_argument('--device',
                       default=None,
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Force specific device for compilation (default: auto-detect)')
    
    return parser.parse_args()


def download_model(model_id, download_dir, torch_dtype=torch.float16):
    """
    Download a diffusion model from Hugging Face
    
    Args:
        model_id (str): Hugging Face model identifier
        download_dir (str): Local directory to save the model
        torch_dtype: Torch data type for the model
    
    Returns:
        str: Path to the downloaded model directory
    """
    print(f"🔄 Scaricando modello '{model_id}' da Hugging Face...")
    print(f"📁 Directory di destinazione: {download_dir}")
    
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    
    # Determine model save path
    model_name = model_id.replace('/', '--')
    model_path = os.path.join(download_dir, model_name)
    
    # Check if model already exists
    if os.path.exists(model_path) and os.listdir(model_path):
        print(f"✅ Modello già presente in {model_path}")
        return model_path
    
    try:
        # Try to load as StableDiffusionPipeline first
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            safety_checker=None  # Optional: disable safety checker for faster loading
        )
    except Exception as e:
        print(f"⚠️  Fallback: tentativo con DiffusionPipeline generico...")
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
        except Exception as e2:
            raise RuntimeError(f"❌ Impossibile scaricare il modello {model_id}. Errori: {e}, {e2}")
    
    # Save model locally
    os.makedirs(model_path, exist_ok=True)
    pipeline.save_pretrained(model_path)
    
    print(f"✅ Modello scaricato e salvato in {model_path}")
    return model_path


def get_best_available_device(force_cpu=False, device_override=None):
    """
    Determine the best available device for Pruna compilation
    
    Args:
        force_cpu (bool): Force CPU usage
        device_override (str): Override device selection
    
    Returns:
        tuple: (device_name, is_mps_compatible)
    """
    if force_cpu or device_override == 'cpu':
        return "cpu", True
    
    if device_override:
        if device_override == 'cuda' and torch.cuda.is_available():
            return "cuda", True
        elif device_override == 'mps' and torch.backends.mps.is_available():
            return "mps", False
        elif device_override == 'cuda' and not torch.cuda.is_available():
            print(f"⚠️  CUDA richiesto ma non disponibile, fallback automatico")
        elif device_override == 'mps' and not torch.backends.mps.is_available():
            print(f"⚠️  MPS richiesto ma non disponibile, fallback automatico")
    
    # Auto-detection
    if torch.cuda.is_available():
        return "cuda", True
    elif torch.backends.mps.is_available():
        return "mps", False  # MPS has limited compatibility with some Pruna features
    else:
        return "cpu", True


def compile_model_with_pruna(model_path, compiled_dir, torch_dtype=torch.float16, compilation_mode='normal', force_cpu=False, device_override=None):
    """
    Compile a diffusion model with Pruna optimization
    
    Args:
        model_path (str): Path to the model directory
        compiled_dir (str): Directory to save compiled model
        torch_dtype: Torch data type for the model
        compilation_mode (str): Compilation mode - 'fast', 'moderate', or 'normal'
        force_cpu (bool): Force CPU usage
        device_override (str): Override device selection
    
    Returns:
        str: Path to the compiled model directory
    """
    print(f"🔧 Compilazione modello con Pruna...")
    print(f"📂 Modello sorgente: {model_path}")
    print(f"📁 Directory compilazione: {compiled_dir}")
    
    # Detect best available device and compatibility
    device_name, is_fully_compatible = get_best_available_device(force_cpu, device_override)
    print(f"💻 Dispositivo rilevato: {device_name}")
    if not is_fully_compatible:
        print(f"⚠️  Nota: {device_name} ha compatibilità limitata con alcune ottimizzazioni Pruna")
    
    # Determine compiled model save path
    model_name = os.path.basename(model_path)
    compiled_path = os.path.join(compiled_dir, model_name)
    
    # Create compiled directory
    os.makedirs(compiled_path, exist_ok=True)
    
    # Check if compiled model already exists
    if os.path.exists(compiled_path) and os.listdir(compiled_path):
        config_files = ['model_index.json', 'smash_config.json']
        if any(os.path.exists(os.path.join(compiled_path, f)) for f in config_files):
            print(f"✅ Modello compilato già presente in {compiled_path}")
            return compiled_path
    
    # Load the model for compilation
    if os.path.isdir(model_path):
        try:
            # Try loading as StableDiffusionPipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                safety_checker=None
            )
        except Exception as e:
            print(f"⚠️  Fallback: caricamento con DiffusionPipeline generico...")
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
    else:
        raise RuntimeError(f"❌ Directory modello non trovata: {model_path}")
    
    # Configure Pruna optimization based on compilation mode and device compatibility
    smash_config = SmashConfig(device=device_name)
    
    if compilation_mode == 'fast':
        # Modalità veloce: compilazione rapida con ottimizzazioni leggere
        print("🚀 Modalità VELOCE: ottimizzazioni rapide per tempi di compilazione ridotti")
        if device_name == "mps":
            # Per MPS configurazione ultra-minimale
            pass  # Configurazione vuota
        else:
            # Caching leggero per velocità (solo non-MPS)
            smash_config["cacher"] = "deepcache"
            smash_config["deepcache_interval"] = 3  # Interval più alto per maggiore velocità
            smash_config["compiler"] = "stable_fast"
            # Quantizzazione leggera solo per non-MPS
            smash_config["quantizer"] = "half"
        
    elif compilation_mode == 'moderate':
        # Modalità moderata: bilanciamento tra velocità e qualità
        print("⚖️  Modalità MODERATA: bilanciamento tra velocità di compilazione e qualità")
        if device_name == "mps":
            # Per MPS configurazione ultra-minimale
            pass  # Configurazione vuota
        else:
            # Caching bilanciato (solo non-MPS)
            smash_config["cacher"] = "deepcache"
            smash_config["deepcache_interval"] = 2  # Interval bilanciato
            # Compiler e quantizzazione moderati
            smash_config["compiler"] = "torch_compile"
            smash_config["torch_compile_mode"] = "default"
            smash_config["quantizer"] = "hqq_diffusers"
            smash_config["hqq_diffusers_weight_bits"] = 8
            smash_config["hqq_diffusers_group_size"] = 64
        
    else:  # compilation_mode == 'normal'
        # Modalità normale: qualità massima, tempi più lunghi
        print("🎯 Modalità NORMALE: ottimizzazioni complete per la massima qualità")
        
        if device_name == "mps":
            # Configurazione ultra-minimale per Apple Silicon (MPS)
            print("🍎 Configurazione ultra-minimale per Apple Silicon (MPS)")
            # Per MPS evitiamo tutte le ottimizzazioni incompatibili
            # Usiamo solo quello che funziona certamente
            pass  # Configurazione vuota - solo device targeting
        else:
            # Configurazione completa per CUDA/CPU
            print("🖥️  Configurazione completa per CUDA/CPU")
            # Caching avanzato con fattorizzazione
            smash_config["cacher"] = "fora"
            smash_config["fora_interval"] = 2
            smash_config["fora_start_step"] = 2
            # Factorizer per ottimizzazioni avanzate (solo su CUDA/CPU)
            smash_config["factorizer"] = "qkv_diffusers"
            # Compiler ottimizzato
            smash_config["compiler"] = "torch_compile"
            smash_config["torch_compile_mode"] = "max-autotune"
            # Quantizzazione di alta qualità
            smash_config["quantizer"] = "hqq_diffusers"
            smash_config["hqq_diffusers_weight_bits"] = 4
            smash_config["hqq_diffusers_group_size"] = 32
            if device_name == "cuda":
                smash_config["hqq_diffusers_backend"] = "torchao_int4"
    
    print(f"📊 Configurazione Pruna applicata per modalità: {compilation_mode}")
    
    print("🚀 Avvio compilazione Pruna...")
    compiled = smash(pipeline, smash_config=smash_config)
    
    # Save compiled model
    compiled.save_pretrained(compiled_path)
    
    print(f"✅ Modello ottimizzato salvato in {compiled_path}")
    return compiled_path


def main():
    """Main function"""
    args = parse_args()
    
    # Convert torch dtype string to actual dtype
    torch_dtype = torch.float16 if args.torch_dtype == 'float16' else torch.float32
    
    print("🎯 Docker Pruna - Download e Compilazione Modelli")
    print("=" * 50)
    print(f"📋 Parametri:")
    print(f"   - Modello: {args.model_id}")
    print(f"   - Download Dir: {args.download_dir}")
    print(f"   - Compiled Dir: {args.compiled_dir}")
    print(f"   - Torch dtype: {args.torch_dtype}")
    print(f"   - Compilation Mode: {args.compilation_mode}")
    print(f"   - Force CPU: {args.force_cpu}")
    print(f"   - Device Override: {args.device}")
    print(f"   - Skip Download: {args.skip_download}")
    print(f"   - Skip Compile: {args.skip_compile}")
    print("=" * 50)
    
    model_path = None
    compiled_path = None
    
    try:
        # Step 1: Download model (if not skipped)
        if not args.skip_download:
            model_path = download_model(args.model_id, args.download_dir, torch_dtype)
        else:
            # Use existing model
            model_name = args.model_id.replace('/', '--')
            model_path = os.path.join(args.download_dir, model_name)
            if not os.path.exists(model_path):
                raise RuntimeError(f"❌ Modello non trovato in {model_path}. Rimuovi --skip-download per scaricarlo.")
            print(f"✅ Uso modello esistente: {model_path}")
        
        # Step 2: Compile model with Pruna (if not skipped)
        if not args.skip_compile:
            compiled_path = compile_model_with_pruna(
                model_path, 
                args.compiled_dir, 
                torch_dtype, 
                args.compilation_mode,
                args.force_cpu,
                args.device
            )
        else:
            print("⏭️  Compilazione saltata.")
        
        print("\n🎉 Processo completato con successo!")
        if model_path:
            print(f"📁 Modello scaricato: {model_path}")
        if compiled_path:
            print(f"🚀 Modello compilato: {compiled_path}")
            print(f"\n💡 Per usare il modello compilato, imposta:")
            print(f"   export PRUNA_COMPILED_DIR='{compiled_path}'")
        
    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
