import time
import socket
import threading
import os
import queue
import json
import base64
import math
import requests
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Set

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
MODO_DRON = False
MODEL_PATH = "modelo/yolo11n-pose.pt"
WINDOW_NAME = "UCE Van-GuardIA"

# ============================================================
# DETECCIÓN LOCAL
# ============================================================
DISTANCIA_IMPACTO = 100
VELOCIDAD_MIN_GOLPE = 60
FRAMES_MEMORIA = 5
UMBRAL_AGLOMERACION = 4

SAVE_INTERVAL_PELEA = 6
SAVE_INTERVAL_GOLPE = 2
SAVE_INTERVAL_CAIDO = 6
SAVE_INTERVAL_AGLOMERACION = 10

KP_CABEZA = 0
KP_CADERA_IZQ = 11
KP_CADERA_DER = 12
KP_MUÑECA_IZQ = 9
KP_MUÑECA_DER = 10

# ============================================================
# TRACKING
# ============================================================
USE_TRACKING = True
TRACKER_CONFIG = "bytetrack.yaml"   # o "botsort.yaml"

# ============================================================
# SEGUNDA VALIDACIÓN
# ============================================================
VALIDATION_PROVIDER = "gemini"   # "ollama" o "gemini"
AI_THRESHOLD_CONFIRMACION = 80.0
AI_TIMEOUT = 60

# ---- Ollama ----
OLLAMA_ENABLED = True
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "moondream:1.8b-v2-q3_K_M"
OLLAMA_TEMPERATURE = 0

# ---- Gemini con rotación de keys ----
GEMINI_ENABLED = True
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEYS = [
    "AIzaSyC7MUPzIm4nw7YqJD3-r7PwsjJPXbZmsMk",
    "AIzaSyAr_oYh61MvIHgHUyqrY5oi2YxnHrrjLds",
    "AIzaSyDtdMJcnmu0xuaE7yVkUUCIUBMC2yYkb5I",
    "AIzaSyDIBVqxx3iCjTaGodoOoa_sHuDaHMlgzrk",
    "AIzaSyAEeiUHlQK_5SPQgGB83AmGFjsF5XzwOF4",
    "AIzaSyDkUd1fiFbGQIA2rs8eijUH6wjjT0z-GME",
    "AIzaSyAWw8TjNg3o1_F2eKH8-NAmpKP7qWA8OpU",
    "AIzaSyB01OTqmB5D4F69_rJoJPeTrufXsCHW1Aw"
]
GEMINI_TIMEOUT = 35

# ============================================================
# RÁFAGA
# ============================================================
BURST_FRAME_COUNT = 3
BURST_FRAME_GAP_MS = 180

# ============================================================
# ANTISPAM / EVENTOS CONFIRMADOS
# ============================================================
EVENT_MEMORY_SECONDS = 35
EVENT_CENTROID_DISTANCE = 0.18
EVENT_SIZE_DELTA = 0.40
EVENT_ID_OVERLAP_MIN = 1   # si comparte al menos 1 track id relevante, suele ser la misma situación

ALERTA_COOLDOWN_SEGUNDOS = {
    "pelea": 20,
    "caido": 30,
    "aglomeracion": 40
}

# ============================================================
# TELEGRAM
# ============================================================
TELEGRAM_ENABLED = True
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8321811557:AAGOFAiPDSskZsYS1JVebHFgGeoAK7c_Jxc")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003715469757")

# ============================================================
# DRON CONFIG
# ============================================================
DRONE_IP = "192.168.28.1"
DRONE_CMD_PORT = 7080
LOCAL_VIDEO_PORT = 7070

START_CMD = bytes.fromhex("cc 5a 01 82 02 36 b7")
STOP_CMD = bytes.fromhex("cc 5a 01 82 02 37 b6")

SOCKET_RCVBUF = 8 * 1024 * 1024

print(f"Iniciando VanguardIA UCE | Modo: {'DRON' if MODO_DRON else 'WEBCAM'}")

import cv2
import numpy as np
from PIL import ImageFile
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# DECODER
# ============================================================
@dataclass
class FrameAssembly:
    frame_id: int
    started_at: float = field(default_factory=time.time)
    parts: Dict[int, bytes] = field(default_factory=dict)
    expected_next_seq: int = 1
    last_seq: Optional[int] = None
    closed: bool = False
    invalid: bool = False


def deobfuscate_packet(packet: bytes, packet_len: int) -> bytes:
    if packet_len < 9:
        return packet[:packet_len]

    data = bytearray(packet[:packet_len])
    b0 = data[0] & 0xFF
    b2 = data[2] & 0xFF

    denom = packet_len - 8
    if denom <= 0:
        return bytes(data)

    idx = (((b0 * b2) + 10) * 6666) % denom
    target = idx + 6

    if 0 <= target < packet_len:
        data[target] ^= 0xFF

    return bytes(data)


class RobustDroneDecoder:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames: Dict[int, FrameAssembly] = {}

    def process_packet(self, raw_packet: bytes, packet_len: int):
        if packet_len < 5:
            return None

        packet = deobfuscate_packet(raw_packet, packet_len)

        frame_id = packet[0]
        flag = packet[1]
        seq = int.from_bytes(packet[2:4], "little")
        payload = packet[4:packet_len]

        with self.lock:
            frame = self.frames.get(frame_id)

            if seq == 1:
                frame = FrameAssembly(frame_id=frame_id)
                self.frames[frame_id] = frame

            if frame is None or frame.invalid:
                return None

            if seq != frame.expected_next_seq:
                frame.invalid = True
                return None

            frame.parts[seq] = payload
            frame.expected_next_seq = seq + 1

            if flag == 0x01:
                frame.last_seq = seq
                frame.closed = True

            if not frame.closed:
                return None

            data = b"".join(frame.parts[i] for i in range(1, frame.last_seq + 1))
            del self.frames[frame_id]

        soi = data.find(b"\xff\xd8")
        eoi = data.rfind(b"\xff\xd9")

        if soi == -1 or eoi == -1:
            return None

        jpeg = data[soi:eoi + 2]
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

        return cv2.rotate(img, cv2.ROTATE_180)


# ============================================================
# UTILIDADES
# ============================================================
def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def safe_json_loads(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


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
  "evento_detectado": "pelea|caido|aglomeracion|ninguno",
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
- caido: persona tendida, acostada o desplomada en cualquier superficie
- aglomeracion: grupo numeroso concentrado en cercanía evidente
""".strip()


OLLAMA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "evento_detectado": {
            "type": "string",
            "enum": ["pelea", "caido", "aglomeracion", "ninguno"]
        },
        "confianza": {"type": "number"},
        "confirmado": {"type": "boolean"},
        "best_frame_index": {"type": "integer"},
        "explicacion": {"type": "string"}
    },
    "required": [
        "evento_detectado",
        "confianza",
        "confirmado",
        "best_frame_index",
        "explicacion"
    ]
}


# ============================================================
# VALIDACIÓN OLLAMA
# ============================================================
def validate_with_ollama(image_paths: List[str], evento_local: str) -> dict:
    if not OLLAMA_ENABLED:
        return {
            "evento_detectado": evento_local,
            "confianza": 0,
            "confirmado": False,
            "best_frame_index": 0,
            "explicacion": "Ollama deshabilitado",
            "provider": "ollama"
        }

    images_b64 = [image_to_base64(p) for p in image_paths]

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": build_validation_prompt(evento_local, len(image_paths)),
        "images": images_b64,
        "stream": False,
        "format": OLLAMA_JSON_SCHEMA,
        "options": {
            "temperature": OLLAMA_TEMPERATURE
        }
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=AI_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()
    response_text = data.get("response", "").strip()

    if not response_text:
        raise ValueError(f"Ollama no devolvió respuesta válida: {data}")

    parsed = safe_json_loads(response_text)
    parsed["provider"] = "ollama"
    return parsed


# ============================================================
# VALIDACIÓN GEMINI CON ROTACIÓN DE KEYS
# ============================================================
class GeminiKeyRotator:
    def __init__(self, api_keys: List[str]):
        self.api_keys = [k.strip() for k in api_keys if k and k.strip()]
        if not self.api_keys:
            raise ValueError("No se proporcionaron GEMINI_API_KEYS válidas")
        self.index = 0
        self.lock = threading.Lock()

    def current_key(self) -> str:
        with self.lock:
            return self.api_keys[self.index]

    def rotate(self):
        with self.lock:
            self.index = (self.index + 1) % len(self.api_keys)

    def keys_in_order(self) -> List[str]:
        with self.lock:
            start = self.index
            return self.api_keys[start:] + self.api_keys[:start]


GEMINI_ROTATOR = GeminiKeyRotator(GEMINI_API_KEYS)


def is_gemini_quota_error(response_json: dict, status_code: int) -> bool:
    if status_code == 429:
        return True

    error = response_json.get("error", {}) if isinstance(response_json, dict) else {}
    message = str(error.get("message", "")).lower()
    status = str(error.get("status", "")).upper()

    quota_markers = [
        "resource_exhausted",
        "quota",
        "rate limit",
        "too many requests",
        "exceeded"
    ]

    return status == "RESOURCE_EXHAUSTED" or any(marker in message for marker in quota_markers)


def validate_with_gemini(image_paths: List[str], evento_local: str) -> dict:
    if not GEMINI_ENABLED:
        return {
            "evento_detectado": evento_local,
            "confianza": 0,
            "confirmado": False,
            "best_frame_index": 0,
            "explicacion": "Gemini deshabilitado",
            "provider": "gemini"
        }

    prompt = build_validation_prompt(evento_local, len(image_paths))

    parts = [{"text": prompt}]
    for path in image_paths:
        b64 = image_to_base64(path)
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64
            }
        })

    payload = {
        "contents": [
            {
                "parts": parts
            }
        ]
    }

    last_error = None
    keys_to_try = GEMINI_ROTATOR.keys_in_order()

    for key in keys_to_try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={key}"
        )
        headers = {"Content-Type": "application/json"}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT)

            if resp.ok:
                data = resp.json()
                try:
                    response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    raise ValueError(f"No se pudo extraer respuesta de Gemini: {data}")

                parsed = safe_json_loads(response_text)
                parsed["provider"] = "gemini"
                parsed["gemini_key_used"] = key[-6:] if len(key) >= 6 else "masked"

                # Si esta key funcionó, la dejamos como actual
                with GEMINI_ROTATOR.lock:
                    if key in GEMINI_ROTATOR.api_keys:
                        GEMINI_ROTATOR.index = GEMINI_ROTATOR.api_keys.index(key)

                return parsed

            else:
                try:
                    data = resp.json()
                except Exception:
                    data = {"error": {"message": resp.text}}

                if is_gemini_quota_error(data, resp.status_code):
                    print(f"[GEMINI] Key en límite/cuota. Rotando a la siguiente...")
                    GEMINI_ROTATOR.rotate()
                    last_error = RuntimeError(f"Límite/cuota con key actual: {data}")
                    continue

                raise RuntimeError(f"Error Gemini no recuperable: status={resp.status_code}, body={data}")

        except requests.RequestException as e:
            last_error = e
            continue

    raise RuntimeError(f"Todas las API keys de Gemini fallaron o llegaron a límite. Último error: {last_error}")


def validate_event_with_provider(image_paths: List[str], evento_local: str) -> dict:
    provider = VALIDATION_PROVIDER.lower()

    if provider == "ollama":
        return validate_with_ollama(image_paths, evento_local)
    elif provider == "gemini":
        return validate_with_gemini(image_paths, evento_local)
    else:
        raise ValueError(f"Proveedor de validación no soportado: {VALIDATION_PROVIDER}")


# ============================================================
# TELEGRAM
# ============================================================
def send_telegram_alert(image_path: str, evento_local: str, ai_result: dict):
    if not TELEGRAM_ENABLED:
        print("[TELEGRAM] Deshabilitado")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    caption = (
        f"🚨 <b>ALERTA VALIDADA</b>\n"
        f"<b>Evento local:</b> {evento_local}\n"
        f"<b>Evento confirmado:</b> {ai_result.get('evento_detectado')}\n"
        f"<b>Confianza:</b> {ai_result.get('confianza')}%\n"
        #f"<b>Proveedor:</b> {ai_result.get('provider')}\n"
        f"<b>Proveedor:</b> Van-GuardIA\n"
        f"<b>Detalle:</b> {ai_result.get('explicacion')}"
    )

    with open(image_path, "rb") as photo:
        files = {"photo": photo}
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML"
        }
        resp = requests.post(url, data=data, files=files, timeout=30)
        resp.raise_for_status()

    print(f"[TELEGRAM] Alerta enviada: {image_path}")


# ============================================================
# APP
# ============================================================
class DroneVanguardIA:
    def __init__(self):
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)
        self.alert_queue = queue.Queue(maxsize=50)

        self.last_save_pelea = 0
        self.last_save_golpe = 0
        self.last_save_caido = 0
        self.last_save_aglomeracion = 0

        self.historial_manos = {}
        self.last_frame = None
        self.last_frame_lock = threading.Lock()

        self.last_alert_sent = {
            "pelea": 0,
            "caido": 0,
            "aglomeracion": 0
        }

        # eventos confirmados recientes
        self.active_events = {
            "pelea": [],
            "caido": [],
            "aglomeracion": []
        }

        self.output_dir = "capturas"
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = YOLO(MODEL_PATH)

        self.cap = None
        if not MODO_DRON:
            self.cap = cv2.VideoCapture(0)

    # =========================
    def distancia(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def punto_valido(self, p):
        return p[0] > 0 and p[1] > 0

    def calcular_velocidad(self, pid, mid, punto):
        if pid not in self.historial_manos:
            self.historial_manos[pid] = {}

        if mid not in self.historial_manos[pid]:
            self.historial_manos[pid][mid] = []

        hist = self.historial_manos[pid][mid]
        hist.append(punto)

        if len(hist) > FRAMES_MEMORIA:
            hist.pop(0)

        if len(hist) >= 2:
            return self.distancia(np.array(hist[-2]), np.array(hist[-1]))

        return 0

    # =========================
    def bbox_to_signature(self, bbox, frame_shape):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return {
            "cx": cx / w,
            "cy": cy / h,
            "w": bw / w,
            "h": bh / h
        }

    def cleanup_active_events(self):
        now = time.time()
        for event_type in self.active_events:
            self.active_events[event_type] = [
                e for e in self.active_events[event_type]
                if now - e["last_seen"] <= EVENT_MEMORY_SECONDS
            ]

    def same_event_signature(self, sig1, sig2):
        dx = sig1["cx"] - sig2["cx"]
        dy = sig1["cy"] - sig2["cy"]
        dist = math.sqrt(dx * dx + dy * dy)

        size_delta_w = abs(sig1["w"] - sig2["w"]) / max(sig2["w"], 1e-6)
        size_delta_h = abs(sig1["h"] - sig2["h"]) / max(sig2["h"], 1e-6)

        return (
            dist <= EVENT_CENTROID_DISTANCE and
            size_delta_w <= EVENT_SIZE_DELTA and
            size_delta_h <= EVENT_SIZE_DELTA
        )

    def same_ids(self, ids1: Set[int], ids2: Set[int]) -> bool:
        if not ids1 or not ids2:
            return False
        overlap = len(ids1.intersection(ids2))
        return overlap >= EVENT_ID_OVERLAP_MIN

    def is_same_confirmed_event(self, event_type: str, signature: dict, track_ids: Set[int]) -> bool:
        self.cleanup_active_events()

        for evt in self.active_events[event_type]:
            # mismo grupo/persona seguido
            if self.same_ids(track_ids, evt["track_ids"]):
                evt["last_seen"] = time.time()
                evt["signature"] = signature
                evt["track_ids"] = set(track_ids)
                return True

            # fallback espacial
            if self.same_event_signature(signature, evt["signature"]):
                evt["last_seen"] = time.time()
                evt["signature"] = signature
                evt["track_ids"] = set(track_ids)
                return True

        return False

    def register_confirmed_event(self, event_type: str, signature: dict, track_ids: Set[int]):
        self.cleanup_active_events()
        self.active_events[event_type].append({
            "signature": signature,
            "track_ids": set(track_ids),
            "last_seen": time.time()
        })

    # =========================
    def detectar_eventos(self, coords, kpts, track_ids):
        """
        Devuelve lista de eventos con:
        {
          "tipo": str,
          "bbox": (x1,y1,x2,y2),
          "track_ids": set(int)
        }
        """
        eventos = []

        # Aglomeración
        if len(coords) >= UMBRAL_AGLOMERACION:
            x1 = np.min(coords[:, 0])
            y1 = np.min(coords[:, 1])
            x2 = np.max(coords[:, 2])
            y2 = np.max(coords[:, 3])
            valid_ids = {int(tid) for tid in track_ids if int(tid) >= 0}

            eventos.append({
                "tipo": "aglomeracion",
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "track_ids": valid_ids
            })

        # Caído
        for i in range(len(coords)):
            cabeza = kpts[i][KP_CABEZA]
            cadera = (kpts[i][KP_CADERA_IZQ] + kpts[i][KP_CADERA_DER]) / 2

            if self.punto_valido(cabeza) and self.punto_valido(cadera):
                dy = abs(cabeza[1] - cadera[1])
                dx = abs(cabeza[0] - cadera[0])

                if dx > dy * 1.5:
                    x1, y1, x2, y2 = coords[i]
                    tid = int(track_ids[i]) if i < len(track_ids) else -1
                    eventos.append({
                        "tipo": "caido",
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "track_ids": {tid} if tid >= 0 else set()
                    })

        # Pelea
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                cabeza_i = kpts[i][KP_CABEZA]
                cabeza_j = kpts[j][KP_CABEZA]

                manos_i = [
                    (kpts[i][KP_MUÑECA_IZQ], "izq"),
                    (kpts[i][KP_MUÑECA_DER], "der")
                ]
                manos_j = [
                    (kpts[j][KP_MUÑECA_IZQ], "izq"),
                    (kpts[j][KP_MUÑECA_DER], "der")
                ]

                pelea_detectada = False

                for mano, tipo in manos_i:
                    if self.punto_valido(mano) and self.punto_valido(cabeza_j):
                        vel = self.calcular_velocidad(i, tipo, mano)
                        dist = self.distancia(mano, cabeza_j)
                        if vel > VELOCIDAD_MIN_GOLPE and dist < DISTANCIA_IMPACTO:
                            pelea_detectada = True
                        elif dist < DISTANCIA_IMPACTO:
                            pelea_detectada = True

                for mano, tipo in manos_j:
                    if self.punto_valido(mano) and self.punto_valido(cabeza_i):
                        vel = self.calcular_velocidad(j, tipo, mano)
                        dist = self.distancia(mano, cabeza_i)
                        if vel > VELOCIDAD_MIN_GOLPE and dist < DISTANCIA_IMPACTO:
                            pelea_detectada = True
                        elif dist < DISTANCIA_IMPACTO:
                            pelea_detectada = True

                if pelea_detectada:
                    x1 = min(coords[i][0], coords[j][0])
                    y1 = min(coords[i][1], coords[j][1])
                    x2 = max(coords[i][2], coords[j][2])
                    y2 = max(coords[i][3], coords[j][3])

                    ids = set()
                    tid_i = int(track_ids[i]) if i < len(track_ids) else -1
                    tid_j = int(track_ids[j]) if j < len(track_ids) else -1
                    if tid_i >= 0:
                        ids.add(tid_i)
                    if tid_j >= 0:
                        ids.add(tid_j)

                    eventos.append({
                        "tipo": "pelea",
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "track_ids": ids
                    })

        return eventos

    # =========================
    def ajustar_a_ventana(self, frame):
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            if win_w <= 0 or win_h <= 0:
                return frame

            h, w = frame.shape[:2]
            scale = min(win_w / w, win_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame

    def evento_en_cooldown(self, evento: str) -> bool:
        ahora = time.time()
        cooldown = ALERTA_COOLDOWN_SEGUNDOS.get(evento, 30)
        return (ahora - self.last_alert_sent.get(evento, 0)) < cooldown

    def marcar_alerta_enviada(self, evento: str):
        self.last_alert_sent[evento] = time.time()

    # =========================
    def capture_burst_frames(self, annotated_frame, evento_local: str) -> List[str]:
        burst_paths = []
        timestamp_base = int(time.time() * 1000)

        first_path = os.path.join(self.output_dir, f"{evento_local}_{timestamp_base}_0.jpg")
        cv2.imwrite(first_path, annotated_frame)
        burst_paths.append(first_path)

        for i in range(1, BURST_FRAME_COUNT):
            time.sleep(BURST_FRAME_GAP_MS / 1000.0)
            with self.last_frame_lock:
                if self.last_frame is None:
                    frame_copy = annotated_frame.copy()
                else:
                    frame_copy = self.last_frame.copy()

            path = os.path.join(self.output_dir, f"{evento_local}_{timestamp_base}_{i}.jpg")
            cv2.imwrite(path, frame_copy)
            burst_paths.append(path)

        return burst_paths

    def enqueue_alert_validation(self, evento_local: str, burst_paths: List[str], signature: dict, track_ids: Set[int]):
        try:
            self.alert_queue.put_nowait({
                "evento_local": evento_local,
                "burst_paths": burst_paths,
                "signature": signature,
                "track_ids": set(track_ids),
                "timestamp": time.time()
            })
            print(f"[ALERT_QUEUE] Evento encolado: {evento_local}")
        except queue.Full:
            print("[ALERT_QUEUE] Cola llena, se descartó una alerta")

    # =========================
    def alert_worker(self):
        while self.running:
            try:
                item = self.alert_queue.get(timeout=1)
            except queue.Empty:
                continue

            evento_local = item["evento_local"]
            burst_paths = item["burst_paths"]
            signature = item["signature"]
            track_ids = item["track_ids"]

            try:
                ai_result = validate_event_with_provider(burst_paths, evento_local)

                confianza = float(ai_result.get("confianza", 0))
                confirmado = bool(ai_result.get("confirmado", False))
                evento_detectado = ai_result.get("evento_detectado", "ninguno")
                best_frame_index = int(ai_result.get("best_frame_index", 0))

                if best_frame_index < 0 or best_frame_index >= len(burst_paths):
                    best_frame_index = 0

                best_image = burst_paths[best_frame_index]

                print(f"[VALIDATION] Resultado: {ai_result}")

                if not (
                    confirmado and
                    confianza >= AI_THRESHOLD_CONFIRMACION and
                    evento_detectado == evento_local
                ):
                    print("[VALIDATION] Evento descartado por validación.")
                    continue

                if self.is_same_confirmed_event(evento_local, signature, track_ids):
                    print(f"[EVENT] '{evento_local}' confirmado pero es la misma situación. No se alerta.")
                    continue

                if self.evento_en_cooldown(evento_local):
                    print(f"[EVENT] '{evento_local}' en cooldown por tipo. No se alerta.")
                    self.register_confirmed_event(evento_local, signature, track_ids)
                    continue

                send_telegram_alert(best_image, evento_local, ai_result)
                self.marcar_alerta_enviada(evento_local)
                self.register_confirmed_event(evento_local, signature, track_ids)

            except Exception as e:
                print(f"[VALIDATION/TELEGRAM] Error procesando alerta: {e}")
            finally:
                self.alert_queue.task_done()

    # =========================
    def video_receiver(self):
        if MODO_DRON:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", LOCAL_VIDEO_PORT))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)

            decoder = RobustDroneDecoder()
            sock.sendto(START_CMD, (DRONE_IP, DRONE_CMD_PORT))

            while self.running:
                try:
                    packet, addr = sock.recvfrom(2048)
                    img = decoder.process_packet(packet, len(packet))

                    if img is not None:
                        with self.last_frame_lock:
                            self.last_frame = img.copy()

                        if self.frame_queue.full():
                            self.frame_queue.get()
                        self.frame_queue.put(img)
                except Exception:
                    continue
        else:
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    with self.last_frame_lock:
                        self.last_frame = frame.copy()

                    if self.frame_queue.full():
                        self.frame_queue.get()
                    self.frame_queue.put(frame)

    # =========================
    def generar_alerta_con_rafaga(self, nombre_evento: str, annotated, signature: dict, track_ids: Set[int]):
        burst_paths = self.capture_burst_frames(annotated, nombre_evento)
        self.enqueue_alert_validation(nombre_evento, burst_paths, signature, track_ids)

    # =========================
    def run(self):
        threading.Thread(target=self.video_receiver, daemon=True).start()
        threading.Thread(target=self.alert_worker, daemon=True).start()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except Exception:
                continue

            if USE_TRACKING:
                results = self.model.track(
                    frame,
                    conf=0.5,
                    imgsz=256,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    verbose=False
                )
            else:
                results = self.model.predict(
                    frame,
                    conf=0.5,
                    imgsz=256,
                    verbose=False
                )

            for r in results:
                annotated = r.plot()

                if r.boxes is None or r.keypoints is None:
                    frame_mostrar = self.ajustar_a_ventana(annotated)
                    cv2.imshow(WINDOW_NAME, frame_mostrar)
                    continue

                coords = r.boxes.xyxy.cpu().numpy()
                kpts = r.keypoints.xy.cpu().numpy()

                # track IDs
                if hasattr(r.boxes, "id") and r.boxes.id is not None:
                    track_ids = r.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = np.full((len(coords),), -1, dtype=int)

                eventos = self.detectar_eventos(coords, kpts, track_ids)
                t = time.time()

                for evento in eventos:
                    tipo = evento["tipo"]
                    bbox = evento["bbox"]
                    ids = evento["track_ids"]
                    signature = self.bbox_to_signature(bbox, annotated.shape)

                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)

                    ids_text = ",".join(map(str, sorted(list(ids)))) if ids else "sin-id"

                    if tipo == "pelea":
                        cv2.putText(annotated, f"PELEA [{ids_text}]", (x1, max(20, y1 - 10)), 0, 0.8, (0, 0, 255), 2)

                        if t - self.last_save_pelea > SAVE_INTERVAL_PELEA:
                            self.generar_alerta_con_rafaga("pelea", annotated, signature, ids)
                            self.last_save_pelea = t

                    elif tipo == "caido":
                        cv2.putText(annotated, f"PERSONA CAIDA [{ids_text}]", (x1, max(20, y1 - 10)), 0, 0.8, (255, 0, 255), 2)

                        if t - self.last_save_caido > SAVE_INTERVAL_CAIDO:
                            self.generar_alerta_con_rafaga("caido", annotated, signature, ids)
                            self.last_save_caido = t

                    elif tipo == "aglomeracion":
                        cv2.putText(annotated, "AGLOMERACION", (x1, max(20, y1 - 10)), 0, 0.8, (0, 255, 0), 2)

                        if t - self.last_save_aglomeracion > SAVE_INTERVAL_AGLOMERACION:
                            self.generar_alerta_con_rafaga("aglomeracion", annotated, signature, ids)
                            self.last_save_aglomeracion = t

                frame_mostrar = self.ajustar_a_ventana(annotated)
                cv2.imshow(WINDOW_NAME, frame_mostrar)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DroneVanguardIA()
    app.run()