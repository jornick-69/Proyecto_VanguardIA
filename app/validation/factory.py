from __future__ import annotations

from app.config import AppConfig
from app.validation.base import BaseValidator
from app.validation.gemini_validator import GeminiValidator
from app.validation.ollama_validator import OllamaValidator


class ValidatorFactory:
    @staticmethod
    def create(config: AppConfig) -> BaseValidator:
        provider = config.validation_provider.lower()
        if provider == "ollama":
            return OllamaValidator(config)
        if provider == "gemini":
            return GeminiValidator(config)
        raise ValueError(f"Proveedor de validación no soportado: {provider}")
