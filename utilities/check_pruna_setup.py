#!/usr/bin/env python3
"""
Script to verify the Pruna version and diagnose CUDA issues
"""

def check_pruna_version():
    """Verify Pruna version and configuration"""
    try:
        import pruna
        print(f"Pruna version: {pruna.__version__}")
        
        # Try to see if there are global settings that force the CPU
        from pruna import SmashConfig
        
        # Basic test
        config_test = SmashConfig()
        print(f"Default SmashConfig: {config_test}")
        
        # Test with explicit device
        if torch.cuda.is_available():
            config_cuda = SmashConfig(device='cuda')
            print(f"SmashConfig CUDA: configured for CUDA device")
        
    except Exception as e:
        print(f"Error checking Pruna: {e}")

def check_dependencies():
    """Check critical dependencies"""
    import torch
    import subprocess
    import sys
    
    print("🔍 CHECK DEPENDENCIES")
    print("=" * 40)
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check system CUDA installation
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"NVCC: {result.stdout.split('release')[1].split(',')[0].strip()}")
        else:
            print("NVCC: Not found")
    except:
        print("NVCC: Not available")
    
    # List installed related packages
    try:
        import pkg_resources
        installed_packages = [d.project_name for d in pkg_resources.working_set]
        
        relevant_packages = [p for p in installed_packages if any(keyword in p.lower() for keyword in ['torch', 'cuda', 'pruna', 'diffusers'])]
        print(f"\nRelevant installed packages:")
        for pkg in sorted(relevant_packages):
            try:
                version = pkg_resources.get_distribution(pkg).version
                print(f"  {pkg}: {version}")
            except:
                print(f"  {pkg}: version not determinable")
                
    except Exception as e:
        print(f"Error checking packages: {e}")

if __name__ == "__main__":
    import torch
    
    print("🔍 PRUNA AND CUDA DIAGNOSTICS")
    print("=" * 50)
    
    check_dependencies()
    print()
    check_pruna_version()
    
    print("\n💡 SUGGESTIONS:")
    print("1. If Pruna always compiles for CPU, try 'force_cuda_compile.py'")
    print("2. Ensure PyTorch and Pruna are compatible")
    print("3. Consider updating Pruna if an update is available")
    print("4. Use 'fast' mode for initial quick tests")
