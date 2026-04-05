from __future__ import annotations

import threading
import requests

from app.config import AppConfig
from app.models import ValidationResult
from app.utils.files import image_to_base64
from app.utils.json_utils import safe_json_loads
from app.validation.base import BaseValidator
from app.validation.prompt_builder import build_validation_prompt


class GeminiKeyRotator:
    def __init__(self, api_keys: list[str]):
        self.api_keys = [key.strip() for key in api_keys if key and key.strip()]
        if not self.api_keys:
            raise ValueError("No se proporcionaron GEMINI_API_KEYS válidas")
        self.index = 0
        self.lock = threading.Lock()

    def keys_in_order(self) -> list[str]:
        with self.lock:
            start = self.index
            return self.api_keys[start:] + self.api_keys[:start]

    def set_current(self, key: str) -> None:
        with self.lock:
            if key in self.api_keys:
                self.index = self.api_keys.index(key)

    def rotate(self) -> None:
        with self.lock:
            self.index = (self.index + 1) % len(self.api_keys)


class GeminiValidator(BaseValidator):
    def __init__(self, config: AppConfig):
        self.config = config
        self.rotator = GeminiKeyRotator(config.gemini_api_keys)

    @staticmethod
    def _is_quota_error(response_json: dict, status_code: int) -> bool:
        if status_code == 429:
            return True

        error = response_json.get("error", {}) if isinstance(response_json, dict) else {}
        message = str(error.get("message", "")).lower()
        status = str(error.get("status", "")).upper()

        markers = ["resource_exhausted", "quota", "rate limit", "too many requests", "exceeded"]
        return status == "RESOURCE_EXHAUSTED" or any(marker in message for marker in markers)

    def validate(self, image_paths: list[str], evento_local: str) -> ValidationResult:
        if not self.config.gemini_enabled:
            return ValidationResult(evento_local, 0.0, False, 0, "Gemini deshabilitado", "gemini")

        parts = [{"text": build_validation_prompt(evento_local, len(image_paths))}]
        for path in image_paths:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_to_base64(path),
                    }
                }
            )

        payload = {"contents": [{"parts": parts}]}
        last_error: Exception | None = None

        for key in self.rotator.keys_in_order():
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.config.gemini_model}:generateContent?key={key}"
            )
            try:
                response = requests.post(url, json=payload, timeout=self.config.gemini_timeout)
                if response.ok:
                    data = response.json()
                    response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = safe_json_loads(response_text)
                    self.rotator.set_current(key)
                    return ValidationResult(
                        evento_detectado=parsed["evento_detectado"],
                        confianza=float(parsed["confianza"]),
                        confirmado=bool(parsed["confirmado"]),
                        best_frame_index=int(parsed["best_frame_index"]),
                        explicacion=parsed["explicacion"],
                        provider="gemini",
                    )

                try:
                    data = response.json()
                except Exception:
                    data = {"error": {"message": response.text}}

                if self._is_quota_error(data, response.status_code):
                    print("[GEMINI] Key en límite/cuota. Rotando a la siguiente...")
                    self.rotator.rotate()
                    last_error = RuntimeError("Key de Gemini en límite")
                    continue

                raise RuntimeError(f"Error Gemini no recuperable: {data}")
            except requests.RequestException as exc:
                last_error = exc
                continue

        raise RuntimeError(f"Todas las keys de Gemini fallaron. Último error: {last_error}")
