#!/usr/bin/env python3
"""
Script per verificare la versione di Pruna e diagnosticare problemi di CUDA
"""

def check_pruna_version():
    """Verifica versione e configurazione Pruna"""
    try:
        import pruna
        print(f"Pruna versione: {pruna.__version__}")
        
        # Prova a vedere se ci sono impostazioni globali che forzano CPU
        from pruna import SmashConfig
        
        # Test base
        config_test = SmashConfig()
        print(f"SmashConfig di default: {config_test}")
        
        # Test con device esplicito
        if torch.cuda.is_available():
            config_cuda = SmashConfig(device='cuda')
            print(f"SmashConfig CUDA: configurato per device CUDA")
        
    except Exception as e:
        print(f"Errore verifica Pruna: {e}")

def check_dependencies():
    """Verifica dipendenze critiche"""
    import torch
    import subprocess
    import sys
    
    print("🔍 VERIFICA DIPENDENZE")
    print("=" * 40)
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponibile: {torch.cuda.is_available()}")
    
    # Verifica installazione CUDA di sistema
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"NVCC: {result.stdout.split('release')[1].split(',')[0].strip()}")
        else:
            print("NVCC: Non trovato")
    except:
        print("NVCC: Non disponibile")
    
    # Lista pacchetti installati correlati
    try:
        import pkg_resources
        installed_packages = [d.project_name for d in pkg_resources.working_set]
        
        relevant_packages = [p for p in installed_packages if any(keyword in p.lower() for keyword in ['torch', 'cuda', 'pruna', 'diffusers'])]
        print(f"\nPacchetti rilevanti installati:")
        for pkg in sorted(relevant_packages):
            try:
                version = pkg_resources.get_distribution(pkg).version
                print(f"  {pkg}: {version}")
            except:
                print(f"  {pkg}: versione non determinabile")
                
    except Exception as e:
        print(f"Errore verifica pacchetti: {e}")

if __name__ == "__main__":
    import torch
    
    print("🔍 DIAGNOSTICA PRUNA E CUDA")
    print("=" * 50)
    
    check_dependencies()
    print()
    check_pruna_version()
    
    print("\n💡 SUGGERIMENTI:")
    print("1. Se Pruna compila sempre per CPU, prova 'force_cuda_compile.py'")
    print("2. Verifica che PyTorch e Pruna siano compatibili")
    print("3. Considera l'aggiornamento di Pruna se disponibile")
    print("4. Usa modalità 'fast' per test iniziali")
