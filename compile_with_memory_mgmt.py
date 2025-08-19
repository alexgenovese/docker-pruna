#!/usr/bin/env python3
"""
Script to manage GPU memory during Pruna compilation
"""

import torch
import gc
import os

def clear_gpu_memory():
    """Fully clears GPU memory"""
    if torch.cuda.is_available():
        print("🧹 Clearing GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Force aggressive cleanup
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
        
        # Show memory status
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
        print(f"   📊 GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {free:.1f}GB free of {total:.1f}GB total")
        
        return free > 2.0  # Returns True if we have at least 2GB free

def compile_with_memory_management(model_id="runwayml/stable-diffusion-v1-5", compilation_mode="fast"):
    """
    Compile with optimized memory management
    """
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
    from pruna import SmashConfig, smash
    
    print("🚀 COMPILING WITH OPTIMIZED MEMORY MANAGEMENT")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available!")
    
    # Initial cleanup
    if not clear_gpu_memory():
        print("⚠️  GPU memory insufficient, you may need to restart the process")
    
    # Set environment variables to optimize memory
    os.environ.update({
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:512',
        'CUDA_VISIBLE_DEVICES': '0',
        'TORCH_CUDA_ARCH_LIST': '7.0;7.5;8.0;8.6'
    })
    
    # Model path
    model_name = model_id.replace('/', '--')
    model_path = f"./models/{model_name}"
    compiled_path = f"./compiled_models/{model_name}"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    
    print(f"📂 Loading model: {model_path}")
    
    # Load model with optimized dtype
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Usa sempre float16 per risparmiare memoria
        safety_checker=None,
        low_cpu_mem_usage=True,     # Carica direttamente su GPU
        device_map="auto"           # Distribuzione automatica
    )
    print("✅ Model loaded")
    
    # Check memory after loading
    clear_gpu_memory()
    
    # Pruna configuration optimized for memory
    print("🔧 Pruna configuration optimized for memory...")
    
    smash_config = SmashConfig(device='cuda')
    
    if compilation_mode == 'fast':
        # Ultra-light configuration
        smash_config['quantizer'] = 'half'
        print("📊 FAST mode: quantizer 'half' only")
        
    elif compilation_mode == 'moderate':
        # Balanced configuration - avoid torch_compile which uses a lot of memory
        smash_config['cacher'] = 'deepcache'
        smash_config['deepcache_interval'] = 4  # Intervallo alto per risparmiare memoria
        smash_config['quantizer'] = 'half'
        print("📊 MODERATE mode: deepcache + quantizer 'half'")
        
    else:  # normal - configurazione più aggressiva ma attenta alla memoria
        smash_config['cacher'] = 'fora'
        smash_config['fora_interval'] = 4  # High interval to save memory
        smash_config['fora_start_step'] = 4
        smash_config['quantizer'] = 'hqq_diffusers'
        smash_config['hqq_diffusers_weight_bits'] = 8  # 8 bits instead of 4 for stability
        smash_config['hqq_diffusers_group_size'] = 128  # Large group size for memory
        # Avoid torch_compile which consumes a lot of memory
        print("📊 NORMAL mode: fora + hqq (without torch_compile)")
    
    # Assicurati che la directory esista
    os.makedirs(compiled_path, exist_ok=True)
    
    try:
        print("🚀 Starting Pruna compilation...")
        print("⏳ Monitoring memory during compilation...")
        
        # Final cleanup before compilation
        clear_gpu_memory()
        
        # Compilation
        compiled = smash(pipeline, smash_config=smash_config)
        
        print("✅ Compilation completed!")
        
        # Clean up original pipeline to free memory
        del pipeline
        clear_gpu_memory()
        
        # Save compiled model
        compiled.save_pretrained(compiled_path)
        
        print(f"💾 Model saved to: {compiled_path}")
        return compiled_path
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU memory error: {e}")
        print("💡 Suggestions:")
        print("   1. Restart the process to free all memory")
        print("   2. Use 'fast' mode to use less memory")
        print("   3. Close other applications that use the GPU")
        
        # Attempt to at least save the base model
        print("🔄 Attempting to save base model...")
        try:
            pipeline.save_pretrained(compiled_path)
            print(f"💾 Base model saved to: {compiled_path}")
            return compiled_path
        except:
            print("❌ Unable to save base model either")
            return None
            
    except Exception as e:
        print(f"❌ General error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compilation with GPU memory management')
    parser.add_argument('--model-id', default='runwayml/stable-diffusion-v1-5', help='Model ID')
    parser.add_argument('--mode', default='fast', choices=['fast', 'moderate', 'normal'], help='Compilation mode')
    
    args = parser.parse_args()
    
    print(f"🎯 Model: {args.model_id}")
    print(f"🎯 Mode: {args.mode}")
    
    result = compile_with_memory_management(args.model_id, args.mode)
    
    if result:
        print(f"\n✅ SUCCESS!")
        print(f"📁 Model: {result}")
    else:
        print(f"\n❌ FAILURE!")
        print("💡 Try restarting the process and using 'fast' mode")

if __name__ == "__main__":
    main()
