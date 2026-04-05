from __future__ import annotations


def build_validation_prompt(evento_local: str, total_frames: int) -> str:
    return f"""
Eres un verificador visual de seguridad.
Recibirás {total_frames} imágenes consecutivas del mismo evento.

Debes analizar el conjunto completo y responder SOLO JSON válido.
No uses markdown.
No añadas texto adicional.

El evento preliminar detectado localmente es: "{evento_local}"

Devuelve exactamente este esquema:
{{
  "evento_detectado": "pelea|caído|aglomeración|ninguno",
  "confianza": 0,
  "confirmado": false,
  "best_frame_index": 0,
  "explicacion": "texto breve"
}}

Reglas:
- "confianza" entre 0 y 100
- "confirmado" = true solo si la evidencia visual es suficientemente sólida
- "best_frame_index" entre 0 y {total_frames - 1}
- Si no puedes confirmarlo, usa "ninguno"

Criterios:
- pelea: agresión física, forcejeo o interacción violenta visible
- caído: persona tendida, acostada o desplomada en cualquier superficie
- aglomeración: grupo numeroso concentrado en cercanía evidente
""".strip()
