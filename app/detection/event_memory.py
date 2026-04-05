from __future__ import annotations

import math
import time
from app.config import AppConfig
from app.models import EventSignature


class EventMemory:
    def __init__(self, config: AppConfig):
        self.config = config
        self.last_alert_sent = {"pelea": 0.0, "caído": 0.0, "aglomeración": 0.0}
        self.active_events = {"pelea": [], "caído": [], "aglomeración": []}

    @staticmethod
    def bbox_to_signature(bbox, frame_shape) -> EventSignature:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return EventSignature(cx=cx / w, cy=cy / h, w=bw / w, h=bh / h)

    def evento_en_cooldown(self, event_type: str) -> bool:
        cooldown = self.config.cooldowns.get(event_type, 30)
        return (time.time() - self.last_alert_sent.get(event_type, 0)) < cooldown

    def marcar_alerta_enviada(self, event_type: str) -> None:
        self.last_alert_sent[event_type] = time.time()

    def cleanup(self) -> None:
        now = time.time()
        for event_type in self.active_events:
            self.active_events[event_type] = [
                event for event in self.active_events[event_type]
                if now - event["last_seen"] <= self.config.event_memory_seconds
            ]

    def same_ids(self, ids1: set[int], ids2: set[int]) -> bool:
        if not ids1 or not ids2:
            return False
        return len(ids1.intersection(ids2)) >= self.config.event_id_overlap_min

    def same_signature(self, s1: EventSignature, s2: EventSignature) -> bool:
        dx = s1.cx - s2.cx
        dy = s1.cy - s2.cy
        dist = math.sqrt(dx * dx + dy * dy)
        size_delta_w = abs(s1.w - s2.w) / max(s2.w, 1e-6)
        size_delta_h = abs(s1.h - s2.h) / max(s2.h, 1e-6)
        return (
            dist <= self.config.event_centroid_distance
            and size_delta_w <= self.config.event_size_delta
            and size_delta_h <= self.config.event_size_delta
        )

    def is_same_confirmed_event(self, event_type: str, signature: EventSignature, track_ids: set[int]) -> bool:
        self.cleanup()
        for event in self.active_events[event_type]:
            if self.same_ids(track_ids, event["track_ids"]):
                event["last_seen"] = time.time()
                event["signature"] = signature
                event["track_ids"] = set(track_ids)
                return True
            if self.same_signature(signature, event["signature"]):
                event["last_seen"] = time.time()
                event["signature"] = signature
                event["track_ids"] = set(track_ids)
                return True
        return False

    def register_confirmed_event(self, event_type: str, signature: EventSignature, track_ids: set[int]) -> None:
        self.cleanup()
        self.active_events[event_type].append(
            {"signature": signature, "track_ids": set(track_ids), "last_seen": time.time()}
        )
