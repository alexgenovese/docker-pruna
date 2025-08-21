#!/usr/bin/env python3
"""
Script di test per verificare il comportamento di Pruna con CUDA.
Questo script fornisce diagnostica dettagliata per capire perché Pruna compila per CPU.
"""

import torch
import os
import sys
from pathlib import Path

def test_cuda_availability():
    """Test della disponibilità e configurazione CUDA"""
    print("=" * 60)
    print("🔍 DIAGNOSTICA CUDA E PYTORCH")
    print("=" * 60)
    
    print(f"PyTorch versione: {torch.__version__}")
    print(f"CUDA disponibile: {'✅' if torch.cuda.is_available() else '❌'}")
    
    if torch.cuda.is_available():
        print(f"Numero GPU: {torch.cuda.device_count()}")
        print(f"GPU attuale: {torch.cuda.current_device()}")
        print(f"Nome GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Memoria allocata: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Memoria cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # Test allocazione memoria GPU
        try:
            test_tensor = torch.randn(1000, 1000, device='cuda')
            print(f"✅ Test allocazione GPU riuscito")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Test allocazione GPU fallito: {e}")
    else:
        print("❌ CUDA non disponibile")
        print("Possibili cause:")
        print("  - PyTorch installato senza supporto CUDA")
        print("  - Driver NVIDIA non installati o incompatibili")
        print("  - CUDA toolkit mancante")
    
    print(f"MPS disponibile: {'✅' if torch.backends.mps.is_available() else '❌'}")
    return torch.cuda.is_available()

def test_pruna_imports():
    """Test dell'importazione e configurazione Pruna"""
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSTICA PRUNA")
    print("=" * 60)
    
    try:
        from pruna import SmashConfig, smash
        print("✅ Pruna importato correttamente")
        
        # Test configurazione base
        try:
            config_cpu = SmashConfig(device='cpu')
            print("✅ SmashConfig CPU creato")
        except Exception as e:
            print(f"❌ Errore SmashConfig CPU: {e}")
        
        if torch.cuda.is_available():
            try:
                config_cuda = SmashConfig(device='cuda')
                print("✅ SmashConfig CUDA creato")
                return True
            except Exception as e:
                print(f"❌ Errore SmashConfig CUDA: {e}")
                print("⚠️  Pruna potrebbe non supportare CUDA correttamente")
                return False
        else:
            print("⏭️  CUDA non disponibile, test CUDA saltato")
            return False
            
    except ImportError as e:
        print(f"❌ Errore importazione Pruna: {e}")
        return False

def test_environment_variables():
    """Test delle variabili d'ambiente che potrebbero influenzare Pruna"""
    print("\n" + "=" * 60)
    print("🔍 VARIABILI D'AMBIENTE")
    print("=" * 60)
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER', 
        'CUDA_LAUNCH_BLOCKING',
        'TORCH_CUDA_ARCH_LIST',
        'PYTORCH_CUDA_ALLOC_CONF',
        'PRUNA_DEVICE',
        'HF_TOKEN'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'NON IMPOSTATA')
        print(f"{var}: {value}")

def test_simple_diffusion_model():
    """Test con un modello semplice per verificare Pruna"""
    print("\n" + "=" * 60)
    print("🔍 TEST MODELLO SEMPLICE")
    print("=" * 60)
    
    try:
        from diffusers import StableDiffusionPipeline
        from pruna import SmashConfig, smash
        from lib.pruna_config import PrunaModelConfigurator
        
        print("📦 Importazioni riuscite")
        
        # Test configuratore
        configurator = PrunaModelConfigurator()
        model_id = "runwayml/stable-diffusion-v1-5"
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_info = configurator.get_model_info(model_id, device)
        
        print(f"📊 Info modello:")
        print(f"   - Tipo: {model_info['model_type']}")
        print(f"   - Dispositivo: {model_info['device']}")
        print(f"   - Compatibilità: {model_info['compatibility']}")
        
        # Test configurazione Pruna
        smash_config = configurator.get_smash_config(model_id, 'fast', device, False)
        print(f"✅ Configurazione Pruna creata per {device}")
        
        # Verifica se il modello compilato esiste già
        compiled_path = f"./compiled_models/{model_id.replace('/', '--')}"
        if os.path.exists(compiled_path):
            print(f"✅ Modello compilato trovato: {compiled_path}")
            
            # Prova a caricare il modello compilato
            try:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    compiled_path,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                    safety_checker=None
                )
                print(f"✅ Modello compilato caricato correttamente")
                
                # Test spostamento su device
                if device == 'cuda':
                    pipeline = pipeline.to('cuda')
                    print(f"✅ Pipeline spostata su CUDA")
                
                return True
                
            except Exception as e:
                print(f"❌ Errore caricamento modello compilato: {e}")
                return False
        else:
            print(f"⚠️  Modello compilato non trovato: {compiled_path}")
            return False
            
    except ImportError as e:
        print(f"❌ Errore importazione: {e}")
        return False
    except Exception as e:
        print(f"❌ Errore generale: {e}")
        return False

def test_pruna_compilation():
    """Test di compilazione diretta con Pruna"""
    print("\n" + "=" * 60)
    print("🔍 TEST COMPILAZIONE PRUNA")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⏭️  CUDA non disponibile, test compilazione saltato")
        return False
    
    try:
        from diffusers import StableDiffusionPipeline
        from pruna import SmashConfig, smash
        
        # Test con modello esistente se disponibile
        model_path = "./models/runwayml--stable-diffusion-v1-5"
        if not os.path.exists(model_path):
            print(f"⚠️  Modello non trovato: {model_path}")
            print("⚠️  Esegui prima il download del modello")
            return False
        
        print(f"📂 Caricamento modello da: {model_path}")
        
        # Carica il modello
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        print("✅ Modello caricato")
        
        # Forza utilizzo CUDA
        print("🎯 Forzando utilizzo CUDA...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.cuda.set_device(0)
        
        # Configurazione Pruna con CUDA esplicita
        smash_config = SmashConfig(device='cuda')
        smash_config['compiler'] = 'torch_compile'
        smash_config['torch_compile_mode'] = 'default'
        
        print(f"🔧 Configurazione Pruna: device='cuda', compiler='torch_compile'")
        print(f"🚀 Avvio compilazione...")
        
        # Compilazione
        compiled = smash(pipeline, smash_config=smash_config)
        print("✅ Compilazione completata!")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante compilazione: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Funzione principale"""
    print("🔍 DIAGNOSTICA COMPLETA PRUNA E CUDA")
    print("Questo script verifica perché Pruna compila per CPU invece di GPU")
    print()
    
    # Test 1: CUDA
    cuda_available = test_cuda_availability()
    
    # Test 2: Pruna
    pruna_cuda_ok = test_pruna_imports()
    
    # Test 3: Environment
    test_environment_variables()
    
    # Test 4: Modello semplice
    model_test_ok = test_simple_diffusion_model()
    
    # Test 5: Compilazione diretta (solo se CUDA disponibile)
    compilation_ok = test_pruna_compilation() if cuda_available else False
    
    # Riassunto
    print("\n" + "=" * 60)
    print("📋 RIASSUNTO DIAGNOSTICA")
    print("=" * 60)
    print(f"CUDA disponibile: {'✅' if cuda_available else '❌'}")
    print(f"Pruna CUDA OK: {'✅' if pruna_cuda_ok else '❌'}")
    print(f"Test modello: {'✅' if model_test_ok else '❌'}")
    print(f"Test compilazione: {'✅' if compilation_ok else '❌'}")
    
    if not cuda_available:
        print("\n🔧 RACCOMANDAZIONI:")
        print("1. Verifica che PyTorch sia installato con supporto CUDA")
        print("2. Controlla i driver NVIDIA")
        print("3. Verifica CUDA toolkit")
    elif not pruna_cuda_ok:
        print("\n🔧 RACCOMANDAZIONI:")
        print("1. Pruna potrebbe non supportare la tua versione di CUDA/PyTorch")
        print("2. Prova a reinstallare Pruna")
        print("3. Controlla la compatibilità delle versioni")
    elif not compilation_ok:
        print("\n🔧 RACCOMANDAZIONI:")
        print("1. Prova a usare --device cuda esplicitamente")
        print("2. Usa --compilation-mode fast per test")
        print("3. Verifica che non ci siano conflitti di memoria")
    else:
        print("\n✅ Tutto sembra funzionare correttamente!")

if __name__ == "__main__":
    main()
