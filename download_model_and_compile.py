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
try:
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    FluxPipeline = None
from pruna import SmashConfig, smash
from huggingface_hub import login


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download and compile diffusion models with Pruna')
    parser.add_argument('--model-id', 
                       default=os.environ.get('MODEL_DIFF', 'CompVis/stable-diffusion-v1-4'),
                       help='Hugging Face model ID to download (default: %(default)s)')
    parser.add_argument('--download-dir', 
                       default=os.environ.get('DOWNLOAD_DIR'),
                       help='Directory to download models (default: skip download if not specified)')
    parser.add_argument('--compiled-dir', 
                       default=os.environ.get('PRUNA_COMPILED_DIR'),
                       help='Directory to save compiled Pruna models (default: skip compilation if not specified)')
    parser.add_argument('--hf-token',
                       default=os.environ.get('HF_TOKEN'),
                       help='Hugging Face token for authentication')
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


def detect_model_type(model_id):
    """
    Detect the type of model based on model ID
    
    Args:
        model_id (str): Hugging Face model identifier
    
    Returns:
        str: Model type ('flux', 'stable-diffusion', 'generic')
    """
    model_id_lower = model_id.lower()
    
    if 'flux' in model_id_lower:
        return 'flux'
    elif any(keyword in model_id_lower for keyword in ['stable-diffusion', 'sd-', 'compvis']):
        return 'stable-diffusion'
    else:
        return 'generic'


def download_model(model_id, download_dir, torch_dtype=torch.float16, hf_token=None):
    """
    Download a diffusion model from Hugging Face
    
    Args:
        model_id (str): Hugging Face model identifier
        download_dir (str): Local directory to save the model
        torch_dtype: Torch data type for the model
        hf_token (str): Hugging Face token for authentication
    
    Returns:
        str: Path to the downloaded model directory
    """
    print(f"🔄 Scaricando modello '{model_id}' da Hugging Face...")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)

    print(f"📁 Directory di destinazione: {download_dir}")
    
    # Detect model type
    model_type = detect_model_type(model_id)
    print(f"🔍 Tipo di modello rilevato: {model_type}")
    
    # Authenticate with Hugging Face if token is provided
    if hf_token:
        print("🔐 Autenticazione con Hugging Face...")
        try:
            login(token=hf_token)
            print("✅ Autenticazione riuscita")
        except Exception as e:
            print(f"⚠️  Errore durante l'autenticazione: {e}")
            print("ℹ️  Continuando senza autenticazione...")
    
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
        # Configure ignore patterns for large files in repository root
        # FLUX models often have very large safetensors files in root that we want to skip
        ignore_patterns = None
        if model_type == 'flux':
            ignore_patterns = ["*.safetensors", "*.bin", "*.gguf"]
            print("🚫 Escludendo file safetensors/bin/gguf dalla root del repository per modelli FLUX")
        elif any(keyword in model_id.lower() for keyword in ["large", "xl"]):
            ignore_patterns = ["*.safetensors", "*.bin"]
            print("🚫 Escludendo file safetensors/bin dalla root del repository per modelli di grandi dimensioni")
        
        # Try to load based on model type
        pipeline = None
        
        if model_type == 'flux' and FLUX_AVAILABLE and FluxPipeline is not None:
            print("🌊 Tentativo di download con FluxPipeline...")
            try:
                pipeline = FluxPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    ignore_mismatched_sizes=True,
                    use_auth_token=hf_token if hf_token else None,
                    ignore_patterns=ignore_patterns
                )
            except Exception as flux_e:
                print(f"⚠️  FluxPipeline fallito: {flux_e}")
                pipeline = None
        
        if pipeline is None and model_type == 'stable-diffusion':
            print("🎨 Tentativo di download con StableDiffusionPipeline...")
            try:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    safety_checker=None,  # Optional: disable safety checker for faster loading
                    ignore_mismatched_sizes=True,  # Ignore size mismatches for robustness
                    use_auth_token=hf_token if hf_token else None,
                    ignore_patterns=ignore_patterns
                )
            except Exception as sd_e:
                print(f"⚠️  StableDiffusionPipeline fallito: {sd_e}")
                pipeline = None
        
        if pipeline is None:
            print("🔄 Fallback: tentativo con DiffusionPipeline generico...")
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                ignore_mismatched_sizes=True,
                use_auth_token=hf_token if hf_token else None,
                ignore_patterns=ignore_patterns
            )
            
    except Exception as e:
        raise RuntimeError(f"❌ Impossibile scaricare il modello {model_id}. Errore: {e}")
    
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
    
    # Detect model type from path or use generic detection
    model_id_from_path = os.path.basename(model_path).replace('--', '/')
    model_type = detect_model_type(model_id_from_path)
    print(f"🔍 Tipo di modello rilevato per compilazione: {model_type}")
    
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
    
    # Load the model for compilation based on detected type
    pipeline = None
    if os.path.isdir(model_path):
        try:
            # Try loading based on model type
            if model_type == 'flux' and FLUX_AVAILABLE and FluxPipeline is not None:
                print("🌊 Tentativo di caricamento con FluxPipeline...")
                try:
                    pipeline = FluxPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype
                    )
                except Exception as flux_e:
                    print(f"⚠️  FluxPipeline fallito: {flux_e}")
                    pipeline = None
            
            if pipeline is None and model_type == 'stable-diffusion':
                print("🎨 Tentativo di caricamento con StableDiffusionPipeline...")
                try:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        safety_checker=None
                    )
                except Exception as sd_e:
                    print(f"⚠️  StableDiffusionPipeline fallito: {sd_e}")
                    pipeline = None
            
            if pipeline is None:
                print("🔄 Fallback: caricamento con DiffusionPipeline generico...")
                pipeline = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype
                )
        except Exception as e:
            raise RuntimeError(f"❌ Errore nel caricamento del modello: {e}")
    else:
        raise RuntimeError(f"❌ Directory modello non trovata: {model_path}")
    
    # Configure Pruna optimization based on compilation mode, device compatibility, and model type
    smash_config = SmashConfig(device=device_name)
    
    # Special configuration for FLUX models
    if model_type == 'flux':
        print("🌊 Configurazione specifica per modelli FLUX")
        if compilation_mode == 'fast':
            print("🚀 Modalità VELOCE per FLUX: ottimizzazioni ultra-minimali")
            # FLUX models are very large and complex, use minimal optimizations
            if device_name != "mps":
                smash_config["quantizer"] = "half"
        elif compilation_mode == 'moderate':
            print("⚖️  Modalità MODERATA per FLUX: ottimizzazioni bilanciate")
            if device_name != "mps":
                smash_config["quantizer"] = "hqq_diffusers"
                smash_config["hqq_diffusers_weight_bits"] = 8
                smash_config["hqq_diffusers_group_size"] = 128  # Larger group size for FLUX
        else:  # normal
            print("🎯 Modalità NORMALE per FLUX: ottimizzazioni complete ma conservative")
            if device_name != "mps":
                smash_config["quantizer"] = "hqq_diffusers"
                smash_config["hqq_diffusers_weight_bits"] = 4
                smash_config["hqq_diffusers_group_size"] = 64
                if device_name == "cuda":
                    smash_config["hqq_diffusers_backend"] = "torchao_int4"
    else:
        # Standard configuration for Stable Diffusion and other models
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
    
    print(f"📊 Configurazione Pruna applicata per modalità: {compilation_mode} - Tipo modello: {model_type}")
    
    print("🚀 Avvio compilazione Pruna...")
    compiled = smash(pipeline, smash_config=smash_config)
    
    # Save compiled model
    compiled.save_pretrained(compiled_path)
    
    print(f"✅ Modello ottimizzato salvato in {compiled_path}")
    return compiled_path


def main():
    """Main function"""
    args = parse_args()
    
    # Auto-skip logic: if download-dir or compiled-dir are not specified, skip those steps
    auto_skip_download = args.download_dir is None
    auto_skip_compile = args.compiled_dir is None
    
    # Override skip flags if directories are not provided
    if auto_skip_download:
        args.skip_download = True
        print("⏭️  Download automaticamente saltato: --download-dir non specificato")
    
    if auto_skip_compile:
        args.skip_compile = True
        print("⏭️  Compilazione automaticamente saltata: --compiled-dir non specificato")
    
    # Convert torch dtype string to actual dtype
    torch_dtype = torch.float16 if args.torch_dtype == 'float16' else torch.float32
    
    print("=" * 50)
    print(f"📋 Parametri:")
    print(f"   - Modello: {args.model_id}")
    print(f"   - Download Dir: {args.download_dir or 'SALTATO'}")
    print(f"   - Compiled Dir: {args.compiled_dir or 'SALTATO'}")
    print(f"   - HF Token: {'***IMPOSTATO***' if args.hf_token else 'NON IMPOSTATO'}")
    print(f"   - Torch dtype: {args.torch_dtype}")
    print(f"   - Compilation Mode: {args.compilation_mode}")
    print(f"   - Force CPU: {args.force_cpu}")
    print(f"   - Device Override: {args.device}")
    print(f"   - Skip Download: {args.skip_download}")
    print(f"   - Skip Compile: {args.skip_compile}")
    print("=" * 50)
    
    # Check if both operations are skipped
    if args.skip_download and args.skip_compile:
        print("⚠️  Entrambe le operazioni (download e compilazione) sono saltate.")
        print("💡 Specifica almeno --download-dir o --compiled-dir per eseguire un'operazione.")
        return
    
    model_path = None
    compiled_path = None
    
    try:
        # Step 1: Download model (if not skipped)
        if not args.skip_download:
            model_path = download_model(args.model_id, args.download_dir, torch_dtype, args.hf_token)
        else:
            # Use existing model when download is skipped but compile is needed
            if not args.skip_compile:
                # Try to find existing model in default location or use model_id
                if args.download_dir:
                    model_name = args.model_id.replace('/', '--')
                    model_path = os.path.join(args.download_dir, model_name)
                    if not os.path.exists(model_path):
                        # Try alternative paths
                        potential_paths = [
                            os.path.join('./models', model_name),
                            args.model_id  # Use model_id directly as fallback
                        ]
                        for potential_path in potential_paths:
                            if os.path.exists(potential_path):
                                model_path = potential_path
                                break
                        
                        if not model_path or not os.path.exists(model_path):
                            raise RuntimeError(f"❌ Modello non trovato in {model_path}. Specifica --download-dir per scaricarlo.")
                else:
                    # If no download dir specified, try using model_id directly
                    model_path = args.model_id
                
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
