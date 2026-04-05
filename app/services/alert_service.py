from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from app.config import AppConfig
from app.detection.event_memory import EventMemory
from app.models import EventSignature
from app.notification.telegram_notifier import TelegramNotifier
from app.services.burst_capture_service import BurstCaptureService
from app.validation.base import BaseValidator


@dataclass
class AlertItem:
    evento_local: str
    burst_paths: list[str]
    signature: EventSignature
    track_ids: set[int]


class AlertService:
    def __init__(
        self,
        config: AppConfig,
        validator: BaseValidator,
        notifier: TelegramNotifier,
        event_memory: EventMemory,
        burst_capture: BurstCaptureService,
    ):
        self.config = config
        self.validator = validator
        self.notifier = notifier
        self.event_memory = event_memory
        self.burst_capture = burst_capture
        self.alert_queue: queue.Queue[AlertItem] = queue.Queue(maxsize=50)
        self.running = True

    def stop(self) -> None:
        self.running = False

    def start_worker(self) -> None:
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def enqueue(self, evento_local: str, annotated_frame, signature: EventSignature, track_ids: set[int]) -> None:
        burst_paths = self.burst_capture.capture(annotated_frame, evento_local)
        try:
            self.alert_queue.put_nowait(AlertItem(evento_local, burst_paths, signature, set(track_ids)))
            print(f"[ALERT_QUEUE] Evento encolado: {evento_local}")
        except queue.Full:
            print("[ALERT_QUEUE] Cola llena, se descartó una alerta")

    def _worker_loop(self) -> None:
        while self.running:
            try:
                item = self.alert_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                result = self.validator.validate(item.burst_paths, item.evento_local)
                best_index = max(0, min(result.best_frame_index, len(item.burst_paths) - 1))
                best_image = item.burst_paths[best_index]

                print(f"[VALIDATION] Resultado: {result}")

                if not (
                    result.confirmado
                    and result.confianza >= self.config.ai_threshold_confirmacion
                    and result.evento_detectado == item.evento_local
                ):
                    print("[VALIDATION] Evento descartado por validación")
                    continue

                if self.event_memory.is_same_confirmed_event(item.evento_local, item.signature, item.track_ids):
                    print(f"[EVENT] '{item.evento_local}' confirmado pero es la misma situación. No se alerta.")
                    continue

                if self.event_memory.evento_en_cooldown(item.evento_local):
                    print(f"[EVENT] '{item.evento_local}' en cooldown por tipo. No se alerta.")
                    self.event_memory.register_confirmed_event(item.evento_local, item.signature, item.track_ids)
                    continue

                self.notifier.send_alert(best_image, item.evento_local, result)
                self.event_memory.marcar_alerta_enviada(item.evento_local)
                self.event_memory.register_confirmed_event(item.evento_local, item.signature, item.track_ids)
            except Exception as exc:
                print(f"[VALIDATION/TELEGRAM] Error procesando alerta: {exc}")
            finally:
                self.alert_queue.task_done()
