from __future__ import annotations

import queue
import threading

import cv2
import numpy as np
from PIL import ImageFile

from app.config import AppConfig, load_config
from app.detection.detector import PersonPoseDetector
from app.detection.event_analyzer import EventAnalyzer
from app.detection.event_memory import EventMemory
from app.drone.video_source import VideoSource
from app.notification.telegram_notifier import TelegramNotifier
from app.services.alert_service import AlertService
from app.services.burst_capture_service import BurstCaptureService
from app.validation.factory import ValidatorFactory

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VanguardApplication:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.video_source = VideoSource(self.config, self.frame_queue)
        self.detector = PersonPoseDetector(self.config)
        self.event_analyzer = EventAnalyzer(self.config)
        self.event_memory = EventMemory(self.config)
        self.validator = ValidatorFactory.create(self.config)
        self.notifier = TelegramNotifier(self.config)
        self.burst_capture = BurstCaptureService(self.config, self.video_source)
        self.alert_service = AlertService(
            self.config,
            self.validator,
            self.notifier,
            self.event_memory,
            self.burst_capture,
        )
        self.running = True

        self.last_save_times = {
            "pelea": 0.0,
            "golpe": 0.0,
            "caido": 0.0,
            "aglomeracion": 0.0,
        }

    def _adjust_to_window(self, frame):
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(self.config.window_name)
            if win_w <= 0 or win_h <= 0:
                return frame

            h, w = frame.shape[:2]
            scale = min(win_w / w, win_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame

    def _should_emit(self, event_type: str, now_ts: float) -> bool:
        interval_map = {
            "pelea": self.config.save_interval_pelea,
            "caido": self.config.save_interval_caido,
            "aglomeracion": self.config.save_interval_aglomeracion,
        }
        return (now_ts - self.last_save_times.get(event_type, 0.0)) > interval_map[event_type]

    def _mark_emitted(self, event_type: str, now_ts: float) -> None:
        self.last_save_times[event_type] = now_ts

    def _draw_event(self, annotated, event):
        x1, y1, x2, y2 = map(int, event.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
        ids_text = ",".join(map(str, sorted(list(event.track_ids)))) if event.track_ids else "sin-id"

        if event.tipo == "pelea":
            cv2.putText(annotated, f"PELEA [{ids_text}]", (x1, max(20, y1 - 10)), 0, 0.8, (0, 0, 255), 2)
        elif event.tipo == "caido":
            cv2.putText(annotated, f"PERSONA CAIDA [{ids_text}]", (x1, max(20, y1 - 10)), 0, 0.8, (255, 0, 255), 2)
        elif event.tipo == "aglomeracion":
            cv2.putText(annotated, "AGLOMERACION", (x1, max(20, y1 - 10)), 0, 0.8, (0, 255, 0), 2)

    def run(self) -> None:
        print(f"Iniciando VanguardIA | Modo: {'DRON' if self.config.modo_dron else 'WEBCAM'}")
        threading.Thread(target=self.video_source.run, daemon=True).start()
        self.alert_service.start_worker()

        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.config.window_name, 960, 720)

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except Exception:
                continue

            results = self.detector.infer(frame)
            for result in results:
                annotated = result.plot()

                if result.boxes is None or result.keypoints is None:
                    cv2.imshow(self.config.window_name, self._adjust_to_window(annotated))
                    continue

                coords, kpts, track_ids = self.detector.extract_arrays(result)
                events = self.event_analyzer.detect_events(coords, kpts, track_ids)
                now_ts = __import__("time").time()

                for event in events:
                    self._draw_event(annotated, event)
                    signature = self.event_memory.bbox_to_signature(event.bbox, annotated.shape)
                    if self._should_emit(event.tipo, now_ts):
                        self.alert_service.enqueue(event.tipo, annotated, signature, event.track_ids)
                        self._mark_emitted(event.tipo, now_ts)

                cv2.imshow(self.config.window_name, self._adjust_to_window(annotated))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break

        self.video_source.stop()
        self.alert_service.stop()
        cv2.destroyAllWindows()
