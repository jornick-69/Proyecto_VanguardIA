import time
import socket
import threading
import os
import io
import queue
from dataclasses import dataclass, field
from typing import Dict, Optional

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODO_DRON = True
MODEL_PATH = "Modelo/yolo11n-pose.pt"

# ---- DISTANCIAS ----
DISTANCIA_IMPACTO = 100

# ---- VELOCIDAD ----
VELOCIDAD_MIN_GOLPE = 60
FRAMES_MEMORIA = 5

# ---- AGLOMERACIÓN ----
UMBRAL_AGLOMERACION = 4
SAVE_INTERVAL_AGLOMERACION = 15

# ---- TIEMPOS ----
SAVE_INTERVAL_PELEA = 10
SAVE_INTERVAL_GOLPE = 2
SAVE_INTERVAL_CAIDO = 8

# ---- KEYPOINTS ----
KP_CABEZA = 0
KP_HOMBRO_IZQ = 5
KP_HOMBRO_DER = 6
KP_CADERA_IZQ = 11
KP_CADERA_DER = 12
KP_MUÑECA_IZQ = 9
KP_MUÑECA_DER = 10

WINDOW_NAME = "VANGUARDIA UCE"

print(f"Iniciando Vanguardia UCE | Modo: {'DRON' if MODO_DRON else 'WEBCAM'}")

import cv2
import numpy as np
from PIL import Image, ImageFile
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# DRON CONFIG
# ============================================================
DRONE_IP = "192.168.28.1"
DRONE_CMD_PORT = 7080
LOCAL_VIDEO_PORT = 7070

START_CMD = bytes.fromhex("cc 5a 01 82 02 36 b7")
STOP_CMD = bytes.fromhex("cc 5a 01 82 02 37 b6")

SOCKET_RCVBUF = 8 * 1024 * 1024


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
# APP
# ============================================================
class DroneVanguardIA:
    def __init__(self):
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)

        self.last_save_pelea = 0
        self.last_save_golpe = 0
        self.last_save_caido = 0
        self.last_save_aglomeracion = 0

        self.historial_manos = {}

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
    def detectar_eventos(self, coords, kpts):
        eventos = []

        # 🔵 AGLOMERACIÓN
        if len(coords) >= UMBRAL_AGLOMERACION:
            eventos.append("aglomeracion")

        for i in range(len(coords)):
            # 🟣 PERSONA CAÍDA
            cabeza = kpts[i][KP_CABEZA]
            cadera = (kpts[i][KP_CADERA_IZQ] + kpts[i][KP_CADERA_DER]) / 2

            if self.punto_valido(cabeza) and self.punto_valido(cadera):
                dy = abs(cabeza[1] - cadera[1])
                dx = abs(cabeza[0] - cadera[0])

                if dx > dy * 1.5:
                    eventos.append("caido")

        # 🔴 INTERACCIONES ENTRE PERSONAS
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

                for mano, tipo in manos_i:
                    if self.punto_valido(mano) and self.punto_valido(cabeza_j):
                        vel = self.calcular_velocidad(i, tipo, mano)
                        dist = self.distancia(mano, cabeza_j)

                        if vel > VELOCIDAD_MIN_GOLPE and dist < DISTANCIA_IMPACTO:
                            eventos.append("golpe")
                        elif dist < DISTANCIA_IMPACTO:
                            eventos.append("pelea")

                for mano, tipo in manos_j:
                    if self.punto_valido(mano) and self.punto_valido(cabeza_i):
                        vel = self.calcular_velocidad(j, tipo, mano)
                        dist = self.distancia(mano, cabeza_i)

                        if vel > VELOCIDAD_MIN_GOLPE and dist < DISTANCIA_IMPACTO:
                            eventos.append("golpe")
                        elif dist < DISTANCIA_IMPACTO:
                            eventos.append("pelea")

        return list(set(eventos))

    # =========================
    def ajustar_a_ventana(self, frame):
        """
        Redimensiona el frame al tamaño actual de la ventana,
        manteniendo proporción.
        """
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)

            if win_w <= 0 or win_h <= 0:
                return frame

            h, w = frame.shape[:2]

            scale = min(win_w / w, win_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized
        except:
            return frame

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
                        if self.frame_queue.full():
                            self.frame_queue.get()
                        self.frame_queue.put(img)
                except:
                    continue
        else:
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    if self.frame_queue.full():
                        self.frame_queue.get()
                    self.frame_queue.put(frame)

    # =========================
    def run(self):
        threading.Thread(target=self.video_receiver, daemon=True).start()

        # Crear ventana redimensionable
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except:
                continue

            results = self.model.predict(frame, conf=0.5, imgsz=256, verbose=False)

            for r in results:
                annotated = r.plot()

                if r.boxes is not None and r.keypoints is not None:
                    coords = r.boxes.xyxy.cpu().numpy()
                    kpts = r.keypoints.xy.cpu().numpy()

                    eventos = self.detectar_eventos(coords, kpts)
                    t = time.time()

                    if "golpe" in eventos:
                        cv2.putText(annotated, "PELEA", (50, 50), 0, 1, (0, 0, 255), 3)

                        if t - self.last_save_golpe > SAVE_INTERVAL_GOLPE:
                            cv2.imwrite(f"{self.output_dir}/golpe_{int(t)}.jpg", annotated)
                            self.last_save_golpe = t

                    if "pelea" in eventos:
                        cv2.putText(annotated, "PELEA", (50, 100), 0, 1, (0, 255, 255), 2)

                        if t - self.last_save_pelea > SAVE_INTERVAL_PELEA:
                            cv2.imwrite(f"{self.output_dir}/pelea_{int(t)}.jpg", annotated)
                            self.last_save_pelea = t

                    if "caido" in eventos:
                        cv2.putText(annotated, "PERSONA CAIDA", (50, 150), 0, 1, (255, 0, 255), 2)

                        if t - self.last_save_caido > SAVE_INTERVAL_CAIDO:
                            cv2.imwrite(f"{self.output_dir}/caido_{int(t)}.jpg", annotated)
                            self.last_save_caido = t

                    if "aglomeracion" in eventos:
                        cv2.putText(annotated, "AGLOMERACION", (50, 200), 0, 1, (0, 255, 0), 2)

                        if t - self.last_save_aglomeracion > SAVE_INTERVAL_AGLOMERACION:
                            cv2.imwrite(f"{self.output_dir}/aglomeracion_{int(t)}.jpg", annotated)
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