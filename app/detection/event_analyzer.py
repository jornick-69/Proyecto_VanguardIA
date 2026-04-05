from __future__ import annotations

from typing import Set
import numpy as np

from app.config import AppConfig
from app.models import DetectedEvent


class EventAnalyzer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.historial_manos: dict[int, dict[str, list[np.ndarray]]] = {}

    @staticmethod
    def distancia(p1, p2):
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def punto_valido(p) -> bool:
        return p[0] > 0 and p[1] > 0

    def calcular_velocidad(self, pid: int, hand_id: str, punto) -> float:
        if pid not in self.historial_manos:
            self.historial_manos[pid] = {}
        if hand_id not in self.historial_manos[pid]:
            self.historial_manos[pid][hand_id] = []

        history = self.historial_manos[pid][hand_id]
        history.append(punto)
        if len(history) > self.config.frames_memoria:
            history.pop(0)

        if len(history) >= 2:
            return float(self.distancia(np.array(history[-2]), np.array(history[-1])))
        return 0.0

    def detect_events(self, coords, kpts, track_ids) -> list[DetectedEvent]:
        events: list[DetectedEvent] = []

        if len(coords) >= self.config.umbral_aglomeracion:
            x1 = np.min(coords[:, 0])
            y1 = np.min(coords[:, 1])
            x2 = np.max(coords[:, 2])
            y2 = np.max(coords[:, 3])
            valid_ids = {int(tid) for tid in track_ids if int(tid) >= 0}
            events.append(DetectedEvent("aglomeracion", (float(x1), float(y1), float(x2), float(y2)), valid_ids))

        for i in range(len(coords)):
            cabeza = kpts[i][self.config.kp_cabeza]
            cadera = (kpts[i][self.config.kp_cadera_izq] + kpts[i][self.config.kp_cadera_der]) / 2

            if self.punto_valido(cabeza) and self.punto_valido(cadera):
                dy = abs(cabeza[1] - cadera[1])
                dx = abs(cabeza[0] - cadera[0])
                if dx > dy * 1.5:
                    x1, y1, x2, y2 = coords[i]
                    tid = int(track_ids[i]) if i < len(track_ids) else -1
                    events.append(DetectedEvent("caido", (float(x1), float(y1), float(x2), float(y2)), {tid} if tid >= 0 else set()))

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                cabeza_i = kpts[i][self.config.kp_cabeza]
                cabeza_j = kpts[j][self.config.kp_cabeza]

                manos_i = [
                    (kpts[i][self.config.kp_muneca_izq], "izq"),
                    (kpts[i][self.config.kp_muneca_der], "der"),
                ]
                manos_j = [
                    (kpts[j][self.config.kp_muneca_izq], "izq"),
                    (kpts[j][self.config.kp_muneca_der], "der"),
                ]

                pelea_detectada = False
                for mano, side in manos_i:
                    if self.punto_valido(mano) and self.punto_valido(cabeza_j):
                        vel = self.calcular_velocidad(i, side, mano)
                        dist = self.distancia(mano, cabeza_j)
                        if vel > self.config.velocidad_min_golpe and dist < self.config.distancia_impacto:
                            pelea_detectada = True
                        elif dist < self.config.distancia_impacto:
                            pelea_detectada = True

                for mano, side in manos_j:
                    if self.punto_valido(mano) and self.punto_valido(cabeza_i):
                        vel = self.calcular_velocidad(j, side, mano)
                        dist = self.distancia(mano, cabeza_i)
                        if vel > self.config.velocidad_min_golpe and dist < self.config.distancia_impacto:
                            pelea_detectada = True
                        elif dist < self.config.distancia_impacto:
                            pelea_detectada = True

                if pelea_detectada:
                    x1 = min(coords[i][0], coords[j][0])
                    y1 = min(coords[i][1], coords[j][1])
                    x2 = max(coords[i][2], coords[j][2])
                    y2 = max(coords[i][3], coords[j][3])
                    ids: Set[int] = set()
                    tid_i = int(track_ids[i]) if i < len(track_ids) else -1
                    tid_j = int(track_ids[j]) if j < len(track_ids) else -1
                    if tid_i >= 0:
                        ids.add(tid_i)
                    if tid_j >= 0:
                        ids.add(tid_j)
                    events.append(DetectedEvent("pelea", (float(x1), float(y1), float(x2), float(y2)), ids))

        return events
