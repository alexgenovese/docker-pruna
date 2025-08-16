"""
Configurazioni ottimizzate di Pruna per diversi tipi di modelli.
Gestisce automaticamente la compatibilità e le impostazioni ottimali per ogni modello.
"""

import torch
from pruna import SmashConfig
from typing import Dict, Any, Optional, Tuple
import re


class PrunaModelConfigurator:
    """
    Classe per gestire le configurazioni di Pruna specifiche per ogni tipo di modello.
    Supporta Stable Diffusion (1.5, XL, 3.5), FLUX, Qwen, e Wan con tre modalità di compilazione.
    """
    
    # Definizione dei pattern per il riconoscimento dei modelli
    MODEL_PATTERNS = {
        'stable-diffusion-1.5': [
            r'stable-diffusion-v1-\d+',
            r'runwayml.*stable-diffusion-v1',
            r'sd-v1',
            r'v1-\d+-pruned'
        ],
        'stable-diffusion-xl': [
            r'stable-diffusion-xl',
            r'sdxl',
            r'xl-base',
            r'xl-refiner'
        ],
        'stable-diffusion-3.5': [
            r'stable-diffusion-3',
            r'sd3',
            r'sd-3'
        ],
        'flux': [
            r'flux',
            r'black-forest-labs.*flux'
        ],
        'qwen': [
            r'qwen',
            r'qwen\d+',
            r'alibaba.*qwen'
        ],
        'wan': [
            r'wan',
            r'wanglab.*wan'
        ]
    }
    
    def __init__(self):
        """Inizializza il configuratore Pruna."""
        pass
    
    def detect_model_type(self, model_id: str) -> str:
        """
        Rileva il tipo di modello dall'ID.
        
        Args:
            model_id (str): ID del modello (es. "runwayml/stable-diffusion-v1-5")
            
        Returns:
            str: Tipo di modello rilevato
        """
        model_id_lower = model_id.lower()
        
        for model_type, patterns in self.MODEL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, model_id_lower):
                    return model_type
        
        # Fallback: prova a dedurre dal nome
        if 'stable' in model_id_lower and 'diffusion' in model_id_lower:
            if 'xl' in model_id_lower:
                return 'stable-diffusion-xl'
            elif '3' in model_id_lower:
                return 'stable-diffusion-3.5'
            else:
                return 'stable-diffusion-1.5'
        
        return 'generic'
    
    def get_device_compatibility(self, device: str, model_type: str) -> Dict[str, bool]:
        """
        Verifica la compatibilità delle funzionalità Pruna con il dispositivo e modello.
        
        Args:
            device (str): Dispositivo target ('cuda', 'mps', 'cpu')
            model_type (str): Tipo di modello
            
        Returns:
            Dict[str, bool]: Dizionario delle compatibilità
        """
        compatibility = {
            'fora_cacher': False,
            'deepcache': True,
            'factorizer': False,
            'torch_compile': True,
            'hqq_quantizer': True,
            'torchao_backend': False
        }
        
        if device == 'cuda':
            # CUDA ha la massima compatibilità
            compatibility.update({
                'fora_cacher': model_type in ['stable-diffusion-xl', 'flux'],  # FORA funziona solo con modelli più recenti
                'factorizer': True,
                'torchao_backend': True
            })
        elif device == 'cpu':
            # CPU ha buona compatibilità ma performance limitate
            compatibility.update({
                'fora_cacher': model_type in ['stable-diffusion-xl'],  # FORA limitato su CPU
                'factorizer': model_type not in ['flux', 'qwen', 'wan'],  # Factorizer problematico con modelli grandi su CPU
                'torchao_backend': False
            })
        elif device == 'mps':
            # MPS (Apple Silicon) ha compatibilità molto limitata - configurazione ultra-minimale
            compatibility.update({
                'fora_cacher': False,     # FORA non supportato su MPS
                'deepcache': False,       # DeepCache non supportato su MPS
                'factorizer': False,      # Factorizer non supportato su MPS
                'torch_compile': False,   # TorchCompile può essere problematico su MPS
                'hqq_quantizer': False,   # HQQ ha dipendenze problematiche su MPS
                'torchao_backend': False  # TorchAO non supportato su MPS
            })
        
        return compatibility
    
    def _get_stable_diffusion_15_config(self, mode: str, device: str, compatibility: Dict[str, bool]) -> Dict[str, Any]:
        """Configurazione per Stable Diffusion 1.5"""
        config = {}
        
        if mode == 'fast':
            # Configurazione veloce e leggera
            if compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 4  # Intervallo alto per velocità
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "half"
        
        elif mode == 'moderate':
            # Configurazione bilanciata
            if compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 3
            if compatibility['torch_compile']:
                config["compiler"] = "torch_compile"
                config["torch_compile_mode"] = "default"
            if compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 8
                config["hqq_diffusers_group_size"] = 64
        
        else:  # normal
            # Configurazione per qualità massima (evita FORA per SD 1.5)
            if compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 2
            if compatibility['factorizer']:
                config["factorizer"] = "qkv_diffusers"
            if compatibility['torch_compile']:
                config["compiler"] = "torch_compile"
                config["torch_compile_mode"] = "max-autotune"
            if compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 4
                config["hqq_diffusers_group_size"] = 32
                if compatibility['torchao_backend']:
                    config["hqq_diffusers_backend"] = "torchao_int4"
        
        return config
    
    def _get_stable_diffusion_xl_config(self, mode: str, device: str, compatibility: Dict[str, bool]) -> Dict[str, Any]:
        """Configurazione per Stable Diffusion XL"""
        config = {}
        
        if mode == 'fast':
            # SDXL è più grande, configurazione ultra-leggera
            if compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 5  # Intervallo ancora più alto
            if device != 'mps':
                config["quantizer"] = "half"
        
        elif mode == 'moderate':
            # Configurazione bilanciata per SDXL
            if compatibility['fora_cacher']:
                config["cacher"] = "fora"
                config["fora_interval"] = 3
                config["fora_start_step"] = 3
            elif compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 3
            if compatibility['torch_compile']:
                config["compiler"] = "torch_compile"
                config["torch_compile_mode"] = "default"
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 8
                config["hqq_diffusers_group_size"] = 128  # Group size più grande per SDXL
        
        else:  # normal
            # Configurazione completa per SDXL
            if compatibility['fora_cacher']:
                config["cacher"] = "fora"
                config["fora_interval"] = 2
                config["fora_start_step"] = 2
            elif compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 2
            if compatibility['factorizer']:
                config["factorizer"] = "qkv_diffusers"
            if compatibility['torch_compile']:
                config["compiler"] = "torch_compile"
                config["torch_compile_mode"] = "max-autotune"
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 4
                config["hqq_diffusers_group_size"] = 64
                if compatibility['torchao_backend']:
                    config["hqq_diffusers_backend"] = "torchao_int4"
        
        return config
    
    def _get_stable_diffusion_35_config(self, mode: str, device: str, compatibility: Dict[str, bool]) -> Dict[str, Any]:
        """Configurazione per Stable Diffusion 3.5"""
        config = {}
        
        if mode == 'fast':
            # SD 3.5 è ancora più avanzato, configurazione minimale
            if device != 'mps':
                config["quantizer"] = "half"
        
        elif mode == 'moderate':
            # Configurazione conservative per SD 3.5
            if compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 4
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 8
                config["hqq_diffusers_group_size"] = 128
        
        else:  # normal
            # Configurazione avanzata ma sicura per SD 3.5
            if compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 3
            if compatibility['torch_compile']:
                config["compiler"] = "torch_compile"
                config["torch_compile_mode"] = "default"  # Evita max-autotune per stabilità
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 6  # Compromesso tra qualità e stabilità
                config["hqq_diffusers_group_size"] = 64
                if compatibility['torchao_backend']:
                    config["hqq_diffusers_backend"] = "torchao_int4"
        
        return config
    
    def _get_flux_config(self, mode: str, device: str, compatibility: Dict[str, bool]) -> Dict[str, Any]:
        """Configurazione per modelli FLUX - ottimizzata per memoria GPU"""
        config = {}
        
        if mode == 'fast':
            # FLUX è enorme, configurazione ultra-minimale per memoria
            if device != 'mps':
                config["quantizer"] = "half"
        
        elif mode == 'moderate':
            # Configurazione bilanciata per FLUX - evita torch_compile per memoria
            if compatibility['fora_cacher']:
                config["cacher"] = "fora"
                config["fora_interval"] = 5  # Intervallo molto alto per memoria
                config["fora_start_step"] = 5
            elif compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 5
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 8  # 8 bit per stabilità
                config["hqq_diffusers_group_size"] = 256  # Group size molto grande per FLUX
        
        else:  # normal
            # Configurazione completa ma SENZA torch_compile per problemi di memoria
            if compatibility['fora_cacher']:
                config["cacher"] = "fora"
                config["fora_interval"] = 4
                config["fora_start_step"] = 4
            elif compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 4
            # RIMUOVO torch_compile per FLUX - consuma troppa memoria
            # if compatibility['torch_compile']:
            #     config["compiler"] = "torch_compile"
            #     config["torch_compile_mode"] = "default"
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 8  # 8 bit invece di 4 per memoria
                config["hqq_diffusers_group_size"] = 256  # Group size molto grande
                if compatibility['torchao_backend']:
                    config["hqq_diffusers_backend"] = "torchao_int4"
        
        return config
    
    def _get_qwen_config(self, mode: str, device: str, compatibility: Dict[str, bool]) -> Dict[str, Any]:
        """Configurazione per modelli Qwen (LLM)"""
        config = {}
        
        if mode == 'fast':
            # Configurazione veloce per LLM
            if device != 'mps':
                config["quantizer"] = "half"
        
        elif mode == 'moderate':
            # Configurazione bilanciata per LLM
            if compatibility['torch_compile']:
                config["compiler"] = "torch_compile"
                config["torch_compile_mode"] = "default"
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 8
                config["hqq_diffusers_group_size"] = 64
        
        else:  # normal
            # Configurazione completa per LLM
            if compatibility['torch_compile']:
                config["compiler"] = "torch_compile"
                config["torch_compile_mode"] = "max-autotune"
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 4
                config["hqq_diffusers_group_size"] = 32
                if compatibility['torchao_backend']:
                    config["hqq_diffusers_backend"] = "torchao_int4"
        
        return config
    
    def _get_wan_config(self, mode: str, device: str, compatibility: Dict[str, bool]) -> Dict[str, Any]:
        """Configurazione per modelli Wan"""
        config = {}
        
        if mode == 'fast':
            # Configurazione minimale
            if device != 'mps':
                config["quantizer"] = "half"
        
        elif mode == 'moderate':
            # Configurazione bilanciata
            if compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 3
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 8
                config["hqq_diffusers_group_size"] = 64
        
        else:  # normal
            # Configurazione completa
            if compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 2
            if compatibility['torch_compile']:
                config["compiler"] = "torch_compile"
                config["torch_compile_mode"] = "default"
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 4
                config["hqq_diffusers_group_size"] = 32
        
        return config
    
    def _get_generic_config(self, mode: str, device: str, compatibility: Dict[str, bool]) -> Dict[str, Any]:
        """Configurazione generica per modelli non riconosciuti"""
        config = {}
        
        if mode == 'fast':
            # Configurazione ultra-sicura
            if device != 'mps':
                config["quantizer"] = "half"
        
        elif mode == 'moderate':
            # Configurazione conservative
            if compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 4
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 8
                config["hqq_diffusers_group_size"] = 64
        
        else:  # normal
            # Configurazione completa ma sicura
            if compatibility['deepcache']:
                config["cacher"] = "deepcache"
                config["deepcache_interval"] = 3
            if compatibility['torch_compile']:
                config["compiler"] = "torch_compile"
                config["torch_compile_mode"] = "default"
            if device != 'mps' and compatibility['hqq_quantizer']:
                config["quantizer"] = "hqq_diffusers"
                config["hqq_diffusers_weight_bits"] = 6
                config["hqq_diffusers_group_size"] = 64
        
        return config
    
    def get_smash_config(self, model_id: str, compilation_mode: str = 'normal', 
                        device: str = 'auto', force_cpu: bool = False) -> SmashConfig:
        """
        Genera una configurazione SmashConfig ottimizzata per il modello specifico.
        
        Args:
            model_id (str): ID del modello
            compilation_mode (str): Modalità di compilazione ('fast', 'moderate', 'normal')
            device (str): Dispositivo target ('auto', 'cuda', 'mps', 'cpu')
            force_cpu (bool): Forza l'uso della CPU
            
        Returns:
            SmashConfig: Configurazione ottimizzata
        """
        # Rilevamento automatico del dispositivo
        if device == 'auto':
            if force_cpu:
                device = 'cpu'
            elif torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        # Rilevamento del tipo di modello
        model_type = self.detect_model_type(model_id)
        
        # Verifica compatibilità
        compatibility = self.get_device_compatibility(device, model_type)
        
        # Selezione della configurazione specifica
        config_methods = {
            'stable-diffusion-1.5': self._get_stable_diffusion_15_config,
            'stable-diffusion-xl': self._get_stable_diffusion_xl_config,
            'stable-diffusion-3.5': self._get_stable_diffusion_35_config,
            'flux': self._get_flux_config,
            'qwen': self._get_qwen_config,
            'wan': self._get_wan_config,
            'generic': self._get_generic_config
        }
        
        config_method = config_methods.get(model_type, self._get_generic_config)
        config_dict = config_method(compilation_mode, device, compatibility)
        
        # Creazione della configurazione Pruna
        smash_config = SmashConfig(device=device)
        
        # Applicazione delle configurazioni
        for key, value in config_dict.items():
            smash_config[key] = value
        
        return smash_config
    
    def get_model_info(self, model_id: str, device: str = 'auto') -> Dict[str, Any]:
        """
        Restituisce informazioni dettagliate sul modello e compatibilità.
        
        Args:
            model_id (str): ID del modello
            device (str): Dispositivo target
            
        Returns:
            Dict[str, Any]: Informazioni sul modello
        """
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        model_type = self.detect_model_type(model_id)
        compatibility = self.get_device_compatibility(device, model_type)
        
        return {
            'model_id': model_id,
            'model_type': model_type,
            'device': device,
            'compatibility': compatibility,
            'recommended_modes': self._get_recommended_modes(model_type, device, compatibility)
        }
    
    def _get_recommended_modes(self, model_type: str, device: str, compatibility: Dict[str, bool]) -> Dict[str, str]:
        """Restituisce le modalità raccomandate per il modello e dispositivo."""
        recommendations = {
            'fast': 'Raccomandato per test rapidi e prototipazione',
            'moderate': 'Raccomandato per uso generale e bilanciamento qualità/velocità',
            'normal': 'Raccomandato per produzione e massima qualità'
        }
        
        # Aggiustamenti basati su dispositivo
        if device == 'mps':
            recommendations.update({
                'fast': 'Configurazione minimale per MPS (Apple Silicon)',
                'moderate': 'Configurazione minimale per MPS (Apple Silicon)', 
                'normal': 'Configurazione minimale per MPS (Apple Silicon) - Pruna ha compatibilità limitata'
            })
        
        # Aggiustamenti basati su modello e dispositivo
        elif model_type in ['flux', 'stable-diffusion-3.5'] and device == 'cpu':
            recommendations['normal'] = 'Molto lento su CPU - considera moderate'
        
        if model_type == 'stable-diffusion-1.5' and not compatibility['fora_cacher']:
            recommendations['normal'] = 'Usa deepcache invece di fora per compatibilità'
        
        return recommendations


def create_pruna_configurator() -> PrunaModelConfigurator:
    """
    Factory function per creare un'istanza del configuratore Pruna.
    
    Returns:
        PrunaModelConfigurator: Istanza del configuratore
    """
    return PrunaModelConfigurator()


# Esempio di utilizzo
if __name__ == "__main__":
    configurator = create_pruna_configurator()
    
    # Test con diversi modelli
    test_models = [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-3.5-large",
        "black-forest-labs/FLUX.1-dev",
        "Qwen/Qwen2-7B",
        "unknown/model"
    ]
    
    for model_id in test_models:
        print(f"\n=== {model_id} ===")
        info = configurator.get_model_info(model_id)
        print(f"Tipo: {info['model_type']}")
        print(f"Dispositivo: {info['device']}")
        print(f"Compatibilità FORA: {info['compatibility']['fora_cacher']}")
        
        for mode in ['fast', 'moderate', 'normal']:
            config = configurator.get_smash_config(model_id, mode)
            print(f"{mode}: {config}")
