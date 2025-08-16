#!/usr/bin/env python3
"""
Script migliorato per forzare l'uso di CUDA con Pruna.
"""

import os
import torch
import sys
from pathlib import Path

# Add the lib directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

def force_cuda_compilation(model_id="runwayml/stable-diffusion-v1-5", compilation_mode="fast"):
    """
    Forza la compilazione con CUDA, bypassando l'auto-detection di Pruna
    """
    try:
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
        from pruna import SmashConfig, smash
        
        print("🔥 FORZANDO COMPILAZIONE CUDA")
        print("=" * 50)
        
        # Verifica CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA non disponibile!")
        
        print(f"✅ CUDA disponibile: {torch.cuda.device_count()} GPU")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        
        # Forza device
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.cuda.set_device(0)
        print(f"✅ CUDA device impostato: {torch.cuda.current_device()}")
        
        # Path del modello
        model_name = model_id.replace('/', '--')
        model_path = f"./models/{model_name}"
        compiled_path = f"./compiled_models/{model_name}"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Modello non trovato: {model_path}")
        
        print(f"📂 Caricamento modello: {model_path}")
        
        # Carica modello
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        print("✅ Modello caricato")
        
        # Sposta pipeline su CUDA
        pipeline = pipeline.to('cuda')
        print("✅ Pipeline spostata su CUDA")
        
        # Configurazione Pruna forzata per CUDA
        print("🔧 Creazione configurazione Pruna forzata...")
        
        # Crea SmashConfig con device esplicito
        smash_config = SmashConfig(device='cuda')
        
        # Configurazione base per evitare problemi
        if compilation_mode == 'fast':
            # Configurazione minimale per velocità
            smash_config['quantizer'] = 'half'
            print("📊 Configurazione: quantizer=half")
        elif compilation_mode == 'moderate':
            # Configurazione bilanciata
            smash_config['compiler'] = 'torch_compile'
            smash_config['torch_compile_mode'] = 'default'
            smash_config['quantizer'] = 'half'
            print("📊 Configurazione: torch_compile + quantizer=half")
        else:  # normal
            # Configurazione completa
            smash_config['cacher'] = 'deepcache'
            smash_config['deepcache_interval'] = 3
            smash_config['compiler'] = 'torch_compile'
            smash_config['torch_compile_mode'] = 'default'
            smash_config['quantizer'] = 'hqq_diffusers'
            smash_config['hqq_diffusers_weight_bits'] = 8
            smash_config['hqq_diffusers_group_size'] = 64
            print("📊 Configurazione: deepcache + torch_compile + hqq")
        
        # Assicurati che la directory di destinazione esista
        os.makedirs(compiled_path, exist_ok=True)
        
        print("🚀 Avvio compilazione Pruna con CUDA FORZATO...")
        print("⏳ Questo può richiedere diversi minuti...")
        
        # Compilazione
        compiled = smash(pipeline, smash_config=smash_config)
        
        print("✅ Compilazione completata!")
        
        # Salva modello compilato
        compiled.save_pretrained(compiled_path)
        
        print(f"💾 Modello salvato in: {compiled_path}")
        print("🎉 Processo completato con successo!")
        
        return compiled_path
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Funzione principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Forza compilazione CUDA con Pruna')
    parser.add_argument('--model-id', default='runwayml/stable-diffusion-v1-5', help='Model ID')
    parser.add_argument('--mode', default='fast', choices=['fast', 'moderate', 'normal'], help='Compilation mode')
    
    args = parser.parse_args()
    
    print(f"🎯 Modello: {args.model_id}")
    print(f"🎯 Modalità: {args.mode}")
    
    result = force_cuda_compilation(args.model_id, args.mode)
    
    if result:
        print(f"\n✅ SUCCESSO!")
        print(f"📁 Modello compilato: {result}")
        print(f"\n💡 Per testare il modello compilato:")
        print(f"python3 server.py --compiled-model-path {result}")
    else:
        print(f"\n❌ FALLIMENTO!")
        sys.exit(1)

if __name__ == "__main__":
    main()
