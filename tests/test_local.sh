# Test locale per il download e la compilazione di modelli

# read local environment variables
source .env

# load HF_TOKEN from local env
HF_TOKEN=${HF_TOKEN}

# Test con modalità moderate (più compatibile con MPS)
# python3 main.py --model-id "black-forest-labs/FLUX.1-Krea-dev" --download-dir "./models/flux-krea" --compiled-dir "./compiled_models/flux-krea" --compilation-mode fast --torch-dtype float16 --hf-token ""
python3 utilities/download_model_and_compile.py --model-id "stable-diffusion-v1-5/stable-diffusion-v1-5" --download-dir "./models/stable-diffusion-v1-5" --compiled-dir "./compiled_models/stable-diffusion-v1-5" --compilation-mode fast --torch-dtype float16 --hf-token "${HF_TOKEN}"

# Se anche moderate fallisce, prova con fast
# python3 main.py --model-id "CompVis/stable-diffusion-v1-4" --download-dir "./models/stable-diffusion-v1-4" --compiled-dir "./compiled_models/stable-diffusion-v1-4" --compilation-mode fast

# Se necessario, forza l'uso della CPU
# python3 main.py --model-id "CompVis/stable-diffusion-v1-4" --download-dir "./models/stable-diffusion-v1-4" --compiled-dir "./compiled_models/stable-diffusion-v1-4" --force-cpu --compilation-mode "moderate"