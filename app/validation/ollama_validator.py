from __future__ import annotations

import requests

from app.config import AppConfig
from app.models import ValidationResult
from app.utils.files import image_to_base64
from app.utils.json_utils import safe_json_loads
from app.validation.base import BaseValidator
from app.validation.prompt_builder import build_validation_prompt


OLLAMA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "evento_detectado": {
            "type": "string",
            "enum": ["pelea", "caido", "aglomeracion", "ninguno"],
        },
        "confianza": {"type": "number"},
        "confirmado": {"type": "boolean"},
        "best_frame_index": {"type": "integer"},
        "explicacion": {"type": "string"},
    },
    "required": [
        "evento_detectado",
        "confianza",
        "confirmado",
        "best_frame_index",
        "explicacion",
    ],
}


class OllamaValidator(BaseValidator):
    def __init__(self, config: AppConfig):
        self.config = config

    def validate(self, image_paths: list[str], evento_local: str) -> ValidationResult:
        if not self.config.ollama_enabled:
            return ValidationResult(evento_local, 0.0, False, 0, "Ollama deshabilitado", "ollama")

        payload = {
            "model": self.config.ollama_model,
            "prompt": build_validation_prompt(evento_local, len(image_paths)),
            "images": [image_to_base64(path) for path in image_paths],
            "stream": False,
            "format": OLLAMA_JSON_SCHEMA,
            "options": {"temperature": self.config.ollama_temperature},
        }

        response = requests.post(
            self.config.ollama_url,
            json=payload,
            timeout=self.config.ai_timeout,
        )
        response.raise_for_status()
        data = response.json()
        parsed = safe_json_loads(data.get("response", ""))

        return ValidationResult(
            evento_detectado=parsed["evento_detectado"],
            confianza=float(parsed["confianza"]),
            confirmado=bool(parsed["confirmado"]),
            best_frame_index=int(parsed["best_frame_index"]),
            explicacion=parsed["explicacion"],
            provider="ollama",
        )
