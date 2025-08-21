import os, torch
from pruna import PrunaModel

# Leggi la directory del modello compilato da variabile d'ambiente
MODEL_DIR = os.environ.get("PRUNA_COMPILED_DIR", "./compiled_models/runwayml--stable-diffusion-v1-5")
if not os.path.isdir(MODEL_DIR):
    raise RuntimeError(f"Directory modello non trovata: {MODEL_DIR}")

print(f"Carico modello compilato Pruna da: {MODEL_DIR}")

# Carica il modello ottimizzato Pruna
model = PrunaModel.from_pretrained(MODEL_DIR)

# Costruisci un prompt/SP semplice per test (adatta ai tuoi use-case, qui si assume testo)
PROMPT = "A close-up portrait of a young woman"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Eseguo inferenza di test...")

# Cambia la funzione di inferenza secondo il tipo di modello/pipeline
image = model(prompt=PROMPT, num_inference_steps=20, guidance_scale=3.5).images[0]

# Salva o mostra output
output_path = "./test_output.jpg"
image.save(output_path)
print(f"Immagine generata e salvata: {output_path}")
