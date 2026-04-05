from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set
import time


@dataclass
class FrameAssembly:
    frame_id: int
    started_at: float = field(default_factory=time.time)
    parts: Dict[int, bytes] = field(default_factory=dict)
    expected_next_seq: int = 1
    last_seq: Optional[int] = None
    closed: bool = False
    invalid: bool = False


@dataclass(frozen=True)
class EventSignature:
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class DetectedEvent:
    tipo: str
    bbox: tuple[float, float, float, float]
    track_ids: Set[int]


@dataclass
class ValidationResult:
    evento_detectado: str
    confianza: float
    confirmado: bool
    best_frame_index: int
    explicacion: str
    provider: str
