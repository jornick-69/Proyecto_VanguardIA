from __future__ import annotations

from abc import ABC, abstractmethod
from app.models import ValidationResult


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, image_paths: list[str], evento_local: str) -> ValidationResult:
        raise NotImplementedError
