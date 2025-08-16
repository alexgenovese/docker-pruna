#!/usr/bin/env python3
"""
Script per gestire la memoria GPU durante la compilazione Pruna
"""

import torch
import gc
import os

def clear_gpu_memory():
    """Pulisce completamente la memoria GPU"""
    if torch.cuda.is_available():
        print("🧹 Pulizia memoria GPU...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Forza la pulizia aggressiva
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
        
        # Mostra stato memoria
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
        print(f"   📊 Memoria GPU: {allocated:.1f}GB allocata, {reserved:.1f}GB riservata, {free:.1f}GB libera su {total:.1f}GB totali")
        
        return free > 2.0  # Ritorna True se abbiamo almeno 2GB liberi

def compile_with_memory_management(model_id="runwayml/stable-diffusion-v1-5", compilation_mode="fast"):
    """
    Compila con gestione ottimizzata della memoria
    """
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
    from pruna import SmashConfig, smash
    
    print("🚀 COMPILAZIONE CON GESTIONE MEMORIA OTTIMIZZATA")
    print("=" * 60)
    
    # Verifica CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA non disponibile!")
    
    # Pulizia iniziale
    if not clear_gpu_memory():
        print("⚠️  Memoria GPU insufficiente, potrebbe essere necessario riavviare il processo")
    
    # Imposta variabili per ottimizzare memoria
    os.environ.update({
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:512',
        'CUDA_VISIBLE_DEVICES': '0',
        'TORCH_CUDA_ARCH_LIST': '7.0;7.5;8.0;8.6'
    })
    
    # Path del modello
    model_name = model_id.replace('/', '--')
    model_path = f"./models/{model_name}"
    compiled_path = f"./compiled_models/{model_name}"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modello non trovato: {model_path}")
    
    print(f"📂 Caricamento modello: {model_path}")
    
    # Carica modello con dtype ottimizzato
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Usa sempre float16 per risparmiare memoria
        safety_checker=None,
        low_cpu_mem_usage=True,     # Carica direttamente su GPU
        device_map="auto"           # Distribuzione automatica
    )
    print("✅ Modello caricato")
    
    # Verifica memoria dopo caricamento
    clear_gpu_memory()
    
    # Configurazione Pruna ottimizzata per memoria
    print("🔧 Configurazione Pruna ottimizzata per memoria...")
    
    smash_config = SmashConfig(device='cuda')
    
    if compilation_mode == 'fast':
        # Configurazione ultra-leggera
        smash_config['quantizer'] = 'half'
        print("📊 Modalità FAST: solo quantizer half")
        
    elif compilation_mode == 'moderate':
        # Configurazione bilanciata - evita torch_compile che usa molta memoria
        smash_config['cacher'] = 'deepcache'
        smash_config['deepcache_interval'] = 4  # Intervallo alto per risparmiare memoria
        smash_config['quantizer'] = 'half'
        print("📊 Modalità MODERATE: deepcache + quantizer half")
        
    else:  # normal - configurazione più aggressiva ma attenta alla memoria
        smash_config['cacher'] = 'fora'
        smash_config['fora_interval'] = 4  # Intervallo alto
        smash_config['fora_start_step'] = 4
        smash_config['quantizer'] = 'hqq_diffusers'
        smash_config['hqq_diffusers_weight_bits'] = 8  # 8 bit invece di 4 per stabilità
        smash_config['hqq_diffusers_group_size'] = 128  # Group size grande per memoria
        # EVITA torch_compile che consuma molta memoria
        print("📊 Modalità NORMAL: fora + hqq (senza torch_compile)")
    
    # Assicurati che la directory esista
    os.makedirs(compiled_path, exist_ok=True)
    
    try:
        print("🚀 Avvio compilazione Pruna...")
        print("⏳ Monitoraggio memoria durante compilazione...")
        
        # Ultima pulizia prima della compilazione
        clear_gpu_memory()
        
        # Compilazione
        compiled = smash(pipeline, smash_config=smash_config)
        
        print("✅ Compilazione completata!")
        
        # Pulizia pipeline originale per liberare memoria
        del pipeline
        clear_gpu_memory()
        
        # Salva modello compilato
        compiled.save_pretrained(compiled_path)
        
        print(f"💾 Modello salvato in: {compiled_path}")
        return compiled_path
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ Errore memoria GPU: {e}")
        print("💡 Suggerimenti:")
        print("   1. Riavvia il processo per liberare tutta la memoria")
        print("   2. Usa modalità 'fast' per meno memoria")
        print("   3. Chiudi altre applicazioni che usano GPU")
        
        # Salva almeno il modello base
        print("🔄 Tentativo salvataggio modello base...")
        try:
            pipeline.save_pretrained(compiled_path)
            print(f"💾 Modello base salvato in: {compiled_path}")
            return compiled_path
        except:
            print("❌ Impossibile salvare anche il modello base")
            return None
            
    except Exception as e:
        print(f"❌ Errore generale: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Funzione principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compilazione con gestione memoria GPU')
    parser.add_argument('--model-id', default='runwayml/stable-diffusion-v1-5', help='Model ID')
    parser.add_argument('--mode', default='fast', choices=['fast', 'moderate', 'normal'], help='Compilation mode')
    
    args = parser.parse_args()
    
    print(f"🎯 Modello: {args.model_id}")
    print(f"🎯 Modalità: {args.mode}")
    
    result = compile_with_memory_management(args.model_id, args.mode)
    
    if result:
        print(f"\n✅ SUCCESSO!")
        print(f"📁 Modello: {result}")
    else:
        print(f"\n❌ FALLIMENTO!")
        print("💡 Prova a riavviare il processo e usare modalità 'fast'")

if __name__ == "__main__":
    main()
