# Base Docker Pruna for Faster inferences

## Cosa succede all’avvio container
Il modello sarà già ottimizzato in /compiled_models/flux_pruna/.

Nel workflow ComfyUI, il nodo Pruna leggerà da questa directory. Spesso è sufficiente configurare il path in uno yaml/settings o direttamente nello script del node ("pruna compiled path").

L’inferenza parte istantanea, senza compilazioni ripetute.

## Suggerimenti

Se vuoi integrare tutto nella pipeline ComfyUI, aggiungi le estensioni/plugin Pruna con COPY . e assicurati che nella tua UI si possa configurare il path /compiled_models/.

Sincronizza le versioni di torch/Pruna tra build e run.

Puoi dividere build & run in multistage Docker se necessario, ma in casi semplici lo script sopra funziona direttamente.

Così ottieni un container "ready-to-go": il modello viene scaricato, compilato da Pruna, e ComfyUI/Pruna Node reimpiega direttamente la compilazione per le inferenze.