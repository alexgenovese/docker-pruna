import os

# Funzione pubblica per API: compila un modello dato un model_id
def compile_model(model_id, download_dir, compiled_dir, torch_dtype='float16', compilation_mode='normal', force_cpu=False, device_override=None):
    """
    Compila un modello già scaricato dato un model_id e salva nella cartella compiled_dir.
    """
    import torch as _torch
    model_name = model_id.replace('/', '--')
    model_path = os.path.join(download_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    dtype = _torch.float16 if str(torch_dtype) == 'float16' else _torch.float32
    return compile_model_with_pruna(
        model_path,
        compiled_dir,
        torch_dtype=dtype,
        compilation_mode=compilation_mode,
        force_cpu=force_cpu,
        device_override=device_override
    )
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
from lib.pruna_config import PrunaModelConfigurator

# Default values
DEFAULT_DOWNLOAD_DIR = './models'
DEFAULT_COMPILED_DIR = './compiled_models'
download_dir = Path(os.environ.get('DOWNLOAD_DIR', DEFAULT_DOWNLOAD_DIR))
compiled_dir = Path(os.environ.get('PRUNA_COMPILED_DIR', DEFAULT_COMPILED_DIR))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download and compile diffusion models with Pruna')
    parser.add_argument('--model-id', 
                       default=os.environ.get('MODEL_DIFF', 'CompVis/stable-diffusion-v1-4'),
                       help='Hugging Face model ID to download (default: %(default)s)')
    parser.add_argument('--download-dir',
                       default=None,
                       help=f'Directory to download models (default: {DEFAULT_DOWNLOAD_DIR})')
    parser.add_argument('--compiled-dir',
                       default=None,
                       help=f'Directory to save compiled models (default: {DEFAULT_COMPILED_DIR})')
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
                       default='moderate',
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
    Detect the type of model based on model ID - maintained for backward compatibility
    
    Args:
        model_id (str): Hugging Face model identifier
    
    Returns:
        str: Model type ('flux', 'stable-diffusion', 'generic')
    """
    configurator = PrunaModelConfigurator()
    return configurator.detect_model_type(model_id)


def check_model_exists(model_id, download_dir, compiled_dir):
    """
    Check if a model already exists in download_dir and/or compiled_dir
    
    Args:
        model_id (str): Hugging Face model identifier
        download_dir (str): Directory where models are downloaded
        compiled_dir (str): Directory where compiled models are stored
    
    Returns:
        tuple: (model_exists, compiled_exists, model_path, compiled_path)
    """
    # Convert model ID to folder name format
    model_name = model_id.replace('/', '--')
    
    # Check in download directory
    model_path = os.path.join(download_dir, model_name)
    model_exists = os.path.exists(model_path) and os.path.isdir(model_path) and os.listdir(model_path)
    
    # Check in compiled directory  
    compiled_path = os.path.join(compiled_dir, model_name)
    compiled_exists = os.path.exists(compiled_path) and os.path.isdir(compiled_path) and os.listdir(compiled_path)
    
    # Additional check for compiled model - verify it has required config files
    if compiled_exists:
        config_files = ['model_index.json', 'smash_config.json']
        compiled_exists = any(os.path.exists(os.path.join(compiled_path, f)) for f in config_files)
    
    return model_exists, compiled_exists, model_path, compiled_path


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



def compile_model_with_pruna(model_path, compiled_dir, torch_dtype=torch.float16, compilation_mode='normal', force_cpu=False, device_override=None):
    """
    Compile a diffusion model with Pruna optimization using the new configurator
    
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
    print(f"🔧 Compilazione modello con Pruna usando configurazione ottimizzata...")
    print(f"📂 Modello sorgente: {model_path}")
    print(f"📁 Directory compilazione: {compiled_dir}")
    
    # Detect model type from path or use generic detection
    model_id_from_path = os.path.basename(model_path).replace('--', '/')
    
    # Initialize Pruna configurator
    configurator = PrunaModelConfigurator()
    
    # Detect model type
    model_type = configurator.detect_model_type(model_id_from_path)
    print(f"🔍 Tipo di modello rilevato: {model_type}")
    
    # Detect device
    device_name = device_override if device_override else 'auto'
    if force_cpu:
        device_name = 'cpu'
    
    # Get model info and recommendations
    model_info = configurator.get_model_info(model_id_from_path, device_name)
    actual_device = model_info['device']
    compatibility = model_info['compatibility']
    
    print(f"💻 Dispositivo selezionato: {actual_device}")
    print(f"🔧 Modalità compilazione: {compilation_mode}")
    
    # Enhanced device diagnostics
    print(f"\n🔍 Diagnostica dispositivo:")
    try:
        import torch as _torch_diag
        print(f"   - PyTorch CUDA disponibile: {'✅' if _torch_diag.cuda.is_available() else '❌'}")
        if _torch_diag.cuda.is_available():
            print(f"   - Numero GPU: {_torch_diag.cuda.device_count()}")
            print(f"   - GPU attuale: {_torch_diag.cuda.current_device()}")
            print(f"   - Nome GPU: {_torch_diag.cuda.get_device_name(0)}")
            print(f"   - Memoria GPU totale: {_torch_diag.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   - Force CPU: {force_cpu}")
        print(f"   - Device override: {device_override}")
    except Exception as e:
        print(f"   - Errore diagnostica: {e}")
    
    # Print compatibility info
    print(f"\n✅ Compatibilità Pruna:")
    print(f"   - FORA Cacher: {'✅' if compatibility['fora_cacher'] else '❌'}")
    print(f"   - DeepCache: {'✅' if compatibility['deepcache'] else '❌'}")
    print(f"   - Factorizer: {'✅' if compatibility['factorizer'] else '❌'}")
    print(f"   - TorchCompile: {'✅' if compatibility['torch_compile'] else '❌'}")
    print(f"   - HQQ Quantizer: {'✅' if compatibility['hqq_quantizer'] else '❌'}")
    print(f"   - TorchAO Backend: {'✅' if compatibility['torchao_backend'] else '❌'}")
    
    # Get optimized Pruna configuration
    smash_config = configurator.get_smash_config(
        model_id_from_path, 
        compilation_mode, 
        actual_device, 
        force_cpu
    )
    
    print(f"📊 Configurazione Pruna ottimizzata:")
    # SmashConfig doesn't support iteration, so we'll show a summary
    print(f"   - Dispositivo: {actual_device}")
    print(f"   - Modalità: {compilation_mode}")
    print(f"   - Tipo modello: {model_type}")
    
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
            
            if pipeline is None and model_type.startswith('stable-diffusion'):
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
    
    try:
        print("🚀 Avvio compilazione Pruna con configurazione ottimizzata...")
        
        # Force device setup if CUDA is available but not being used
        if actual_device == 'cuda' and not force_cpu:
            print("🎯 Forcing CUDA device setup per Pruna...")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Ensure GPU 0 is visible
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.set_device(0)
                print(f"   - CUDA device impostato: {torch.cuda.current_device()}")
        
        compiled = smash(pipeline, smash_config=smash_config)
        
        # Save compiled model
        compiled.save_pretrained(compiled_path)
        
        print(f"✅ Modello ottimizzato salvato in {compiled_path}")
        return compiled_path
        
    except Exception as e:
        print(f"⚠️  Errore durante la compilazione Pruna: {e}")
        print("🔄 Salvataggio del modello base senza ottimizzazioni Pruna...")
        
        # Fallback: save the unoptimized model in compiled directory
        pipeline.save_pretrained(compiled_path)
        
        print(f"⚠️  Modello salvato senza ottimizzazioni in {compiled_path}")
        print("ℹ️  Il modello funzionerà ma senza le ottimizzazioni Pruna")
        return compiled_path


def main():
    """Main function"""
    args = parse_args()
    
    # Use default directories if not specified
    effective_download_dir = download_dir if hasattr(args, 'download_dir') and args.download_dir else DEFAULT_DOWNLOAD_DIR
    effective_compiled_dir = compiled_dir if hasattr(args, 'compiled_dir') and args.compiled_dir else DEFAULT_COMPILED_DIR
    
    # Convert torch dtype string to actual dtype
    torch_dtype = torch.float16 if args.torch_dtype == 'float16' else torch.float32
    
    print("=" * 50)
    print(f"📋 Parametri:")
    print(f"   - Modello: {args.model_id}")
    print(f"   - Download Dir: {effective_download_dir}")
    print(f"   - Compiled Dir: {effective_compiled_dir}")
    print(f"   - HF Token: {'***IMPOSTATO***' if args.hf_token else 'NON IMPOSTATO'}")
    print(f"   - Torch dtype: {args.torch_dtype}")
    print(f"   - Compilation Mode: {args.compilation_mode}")
    print(f"   - Force CPU: {args.force_cpu}")
    print(f"   - Device Override: {args.device}")
    print(f"   - Skip Download: {args.skip_download}")
    print(f"   - Skip Compile: {args.skip_compile}")
    print("=" * 50)
    
    # Check if model already exists
    model_exists, compiled_exists, model_path, compiled_path = check_model_exists(
        args.model_id, effective_download_dir, effective_compiled_dir
    )
    
    print(f"\n� Verifica esistenza modello:")
    print(f"   - Modello base: {'✅ PRESENTE' if model_exists else '❌ NON PRESENTE'} in {model_path}")
    print(f"   - Modello compilato: {'✅ PRESENTE' if compiled_exists else '❌ NON PRESENTE'} in {compiled_path}")
    
    # Auto-determine what needs to be done
    need_download = not model_exists and not args.skip_download
    need_compile = not compiled_exists and not args.skip_compile
    
    if compiled_exists and not args.skip_compile:
        print(f"✅ Modello compilato già disponibile, salto la compilazione")
        need_compile = False
    
    if model_exists and not need_compile and not args.skip_download:
        print(f"✅ Modello base già disponibile e compilazione non necessaria")
        need_download = False
    
    print(f"\n📋 Piano di esecuzione:")
    print(f"   - Download: {'✅ NECESSARIO' if need_download else '⏭️ SALTA'}")
    print(f"   - Compilazione: {'✅ NECESSARIO' if need_compile else '⏭️ SALTA'}")
    
    # Check if both operations are skipped
    if not need_download and not need_compile:
        print("\n🎉 Nessuna operazione necessaria - tutti i modelli sono già disponibili!")
        print(f"📁 Modello base: {model_path}")
        print(f"🚀 Modello compilato: {compiled_path}")
        print(f"\n💡 Per usare il modello compilato, imposta:")
        print(f"   export PRUNA_COMPILED_DIR='{compiled_path}'")
        return
    
    final_model_path = model_path if model_exists else None
    final_compiled_path = compiled_path if compiled_exists else None
    
    try:
        # Step 1: Download model (if needed)
        if need_download:
            print(f"\n🔄 Avvio download del modello...")
            final_model_path = download_model(args.model_id, effective_download_dir, torch_dtype, args.hf_token)
        elif model_exists:
            print(f"\n✅ Uso modello esistente: {model_path}")
            final_model_path = model_path
        
        # Step 2: Compile model with Pruna (if needed)
        if need_compile:
            if not final_model_path:
                raise RuntimeError("❌ Impossibile compilare: modello base non disponibile")
            
            print(f"\n🔧 Avvio compilazione del modello...")
            final_compiled_path = compile_model_with_pruna(
                final_model_path, 
                effective_compiled_dir, 
                torch_dtype, 
                args.compilation_mode,
                args.force_cpu,
                args.device
            )
        elif compiled_exists:
            print(f"\n✅ Uso modello compilato esistente: {compiled_path}")
            final_compiled_path = compiled_path
        
        print("\n🎉 Processo completato con successo!")
        if final_model_path:
            print(f"📁 Modello base: {final_model_path}")
        if final_compiled_path:
            print(f"🚀 Modello compilato: {final_compiled_path}")
            print(f"\n💡 Per usare il modello compilato, imposta:")
            print(f"   export PRUNA_COMPILED_DIR='{final_compiled_path}'")
        
    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione: {e}")
        sys.exit(1)

#
# Quando usare questo file? 
# Questo file può essere utilizzato per scaricare e compilare modelli per l'inferenza.
# 
# Funzionalità principali:
# - Verifica automaticamente se il modello esiste già nelle cartelle ./models e ./compiled_models
# - Se il modello non esiste, lo scarica automaticamente da Hugging Face
# - Se il modello compilato non esiste, lo compila automaticamente con Pruna
# - Se entrambi esistono già, salta entrambe le operazioni
# 
# Esempio di utilizzo:
# python download_model_and_compile.py --model-id runwayml/stable-diffusion-v1-5
# python download_model_and_compile.py --model-id CompVis/stable-diffusion-v1-4 --compilation-mode fast
# python download_model_and_compile.py --model-id <MODEL_ID> --download-dir <CUSTOM_DIR> --compiled-dir <CUSTOM_COMPILED_DIR>
# 
if __name__ == "__main__":
    main()
