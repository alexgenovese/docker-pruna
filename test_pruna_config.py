#!/usr/bin/env python3
"""
Utility script per testare la classe PrunaModelConfigurator con diversi modelli.
"""

import sys
sys.path.append('.')
from lib.pruna_config import PrunaModelConfigurator

def test_model_configurations():
    """Test delle configurazioni per diversi tipi di modelli."""
    
    configurator = PrunaModelConfigurator()
    
    # Lista di modelli da testare
    test_models = [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-xl-base-1.0", 
        "stabilityai/stable-diffusion-3.5-large",
        "black-forest-labs/FLUX.1-dev",
        "Qwen/Qwen2-7B",
        "CompVis/stable-diffusion-v1-4"
    ]
    
    print("🧪 Test delle configurazioni Pruna per diversi modelli\n")
    print("=" * 80)
    
    for model_id in test_models:
        print(f"\n🔍 MODELLO: {model_id}")
        print("-" * 60)
        
        try:
            # Ottieni informazioni sul modello
            info = configurator.get_model_info(model_id)
            
            print(f"📊 Tipo modello: {info['model_type']}")
            print(f"💻 Dispositivo: {info['device']}")
            
            # Mostra compatibilità
            compatibility = info['compatibility']
            print(f"🔧 Compatibilità:")
            print(f"   ✅ FORA Cacher: {'SÌ' if compatibility['fora_cacher'] else 'NO'}")
            print(f"   ✅ DeepCache: {'SÌ' if compatibility['deepcache'] else 'NO'}")
            print(f"   ✅ Factorizer: {'SÌ' if compatibility['factorizer'] else 'NO'}")
            print(f"   ✅ TorchCompile: {'SÌ' if compatibility['torch_compile'] else 'NO'}")
            print(f"   ✅ HQQ Quantizer: {'SÌ' if compatibility['hqq_quantizer'] else 'NO'}")
            print(f"   ✅ TorchAO Backend: {'SÌ' if compatibility['torchao_backend'] else 'NO'}")
            
            # Test delle modalità di compilazione
            print(f"\n📋 Test modalità di compilazione:")
            for mode in ['fast', 'moderate', 'normal']:
                try:
                    config = configurator.get_smash_config(model_id, mode)
                    print(f"   ✅ {mode.upper()}: Configurazione creata con successo")
                except Exception as e:
                    print(f"   ❌ {mode.upper()}: Errore - {e}")
            
            # Mostra raccomandazioni
            print(f"\n💡 Raccomandazioni:")
            for mode, desc in info['recommended_modes'].items():
                print(f"   🔸 {mode.upper()}: {desc}")
                
        except Exception as e:
            print(f"❌ Errore nel test del modello: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 Test completato!")

def test_fora_compatibility():
    """Test specifico per la compatibilità FORA."""
    
    configurator = PrunaModelConfigurator()
    
    print("\n🔍 TEST SPECIFICO COMPATIBILITÀ FORA")
    print("=" * 50)
    
    # Modelli che dovrebbero essere compatibili con FORA
    compatible_models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "black-forest-labs/FLUX.1-dev"
    ]
    
    # Modelli che NON dovrebbero essere compatibili con FORA
    incompatible_models = [
        "runwayml/stable-diffusion-v1-5",
        "CompVis/stable-diffusion-v1-4"
    ]
    
    print("\n✅ Modelli che DOVREBBERO supportare FORA:")
    for model_id in compatible_models:
        info = configurator.get_model_info(model_id)
        fora_support = info['compatibility']['fora_cacher']
        status = "✅ SUPPORTATO" if fora_support else "❌ NON SUPPORTATO"
        print(f"   {model_id}: {status}")
    
    print("\n❌ Modelli che NON dovrebbero supportare FORA:")
    for model_id in incompatible_models:
        info = configurator.get_model_info(model_id)
        fora_support = info['compatibility']['fora_cacher']
        status = "✅ CORRETTAMENTE NON SUPPORTATO" if not fora_support else "❌ ERRORE: SUPPORTATO"
        print(f"   {model_id}: {status}")

if __name__ == "__main__":
    test_model_configurations()
    test_fora_compatibility()
