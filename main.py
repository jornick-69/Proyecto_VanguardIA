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
DISTANCIA_PERSONAS = 250
DISTANCIA_IMPACTO = 100

# ---- VELOCIDAD ----
VELOCIDAD_MIN_GOLPE = 60
FRAMES_MEMORIA = 5

# ---- TIEMPOS ----
SAVE_INTERVAL_PELEA = 10
SAVE_INTERVAL_GOLPE = 2

# ---- KEYPOINTS ----
KP_CABEZA = 0
KP_MUÑECA_IZQ = 9
KP_MUÑECA_DER = 10

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

    def process_packet(self, raw_packet: bytes, packet_len: int) -> Optional[np.ndarray]:
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

        if img is None:
            try:
                pil_img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                return None

        return cv2.rotate(img, cv2.ROTATE_180)


# ============================================================
# APP PRINCIPAL
# ============================================================
class DroneVanguardIA:
    def __init__(self):
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)
        self.window_name = "VANGUARDIA UCE"

        # Guardado
        self.last_save_pelea = 0
        self.last_save_golpe = 0

        self.output_dir = "capturas"
        os.makedirs(self.output_dir, exist_ok=True)

        self.historial_manos = {}

        print("Cargando modelo YOLO...")
        self.model = YOLO(MODEL_PATH)

        self.cap = None
        if not MODO_DRON:
            self.cap = cv2.VideoCapture(0)

    # ========================================================
    # UTILIDADES
    # ========================================================
    def distancia(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def punto_valido(self, p):
        return p[0] > 0 and p[1] > 0

    def calcular_velocidad(self, persona_id, mano_id, punto):
        if persona_id not in self.historial_manos:
            self.historial_manos[persona_id] = {}

        if mano_id not in self.historial_manos[persona_id]:
            self.historial_manos[persona_id][mano_id] = []

        historial = self.historial_manos[persona_id][mano_id]
        historial.append(punto)

        if len(historial) > FRAMES_MEMORIA:
            historial.pop(0)

        if len(historial) >= 2:
            return self.distancia(np.array(historial[-2]), np.array(historial[-1]))

        return 0

    # ========================================================
    # DETECTOR INTELIGENTE
    # ========================================================
    def detectar_eventos(self, coords, kpts):
        num_personas = len(coords)

        if num_personas < 2:
            return "none"

        for i in range(num_personas):
            for j in range(i + 1, num_personas):

                cabeza_i = kpts[i][KP_CABEZA]
                cabeza_j = kpts[j][KP_CABEZA]

                manos_i = [(kpts[i][KP_MUÑECA_IZQ], "izq"),
                           (kpts[i][KP_MUÑECA_DER], "der")]

                manos_j = [(kpts[j][KP_MUÑECA_IZQ], "izq"),
                           (kpts[j][KP_MUÑECA_DER], "der")]

                # 🔴 GOLPE
                for mano, tipo in manos_i:
                    if self.punto_valido(mano) and self.punto_valido(cabeza_j):
                        vel = self.calcular_velocidad(i, tipo, mano)
                        dist = self.distancia(mano, cabeza_j)

                        if vel > VELOCIDAD_MIN_GOLPE and dist < DISTANCIA_IMPACTO:
                            return "golpe"

                for mano, tipo in manos_j:
                    if self.punto_valido(mano) and self.punto_valido(cabeza_i):
                        vel = self.calcular_velocidad(j, tipo, mano)
                        dist = self.distancia(mano, cabeza_i)

                        if vel > VELOCIDAD_MIN_GOLPE and dist < DISTANCIA_IMPACTO:
                            return "golpe"

                # 🟡 PELEA
                for mano, _ in manos_i:
                    if self.punto_valido(mano) and self.punto_valido(cabeza_j):
                        if self.distancia(mano, cabeza_j) < DISTANCIA_IMPACTO:
                            return "pelea"

                for mano, _ in manos_j:
                    if self.punto_valido(mano) and self.punto_valido(cabeza_i):
                        if self.distancia(mano, cabeza_i) < DISTANCIA_IMPACTO:
                            return "pelea"

        return "none"

    # ========================================================
    # CAPTURA
    # ========================================================
    def video_receiver(self):

        if MODO_DRON:
            print("📡 Modo DRON activo")

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
            sock.bind(("0.0.0.0", LOCAL_VIDEO_PORT))

            decoder = RobustDroneDecoder()
            sock.sendto(START_CMD, (DRONE_IP, DRONE_CMD_PORT))

            while self.running:
                try:
                    packet, addr = sock.recvfrom(2048)
                    ip, port = addr

                    if ip != DRONE_IP or port != 7070:
                        continue

                    img = decoder.process_packet(packet, len(packet))

                    if img is not None:
                        try:
                            self.frame_queue.put_nowait(img)
                        except queue.Full:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(img)

                except:
                    continue

            sock.sendto(STOP_CMD, (DRONE_IP, DRONE_CMD_PORT))
            sock.close()

        else:
            print("📷 Webcam activa")

            while self.running:
                ret, frame = self.cap.read()

                if ret:
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)

    # ========================================================
    # MAIN LOOP
    # ========================================================
    def run(self):
        threading.Thread(target=self.video_receiver, daemon=True).start()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except:
                continue

            results = self.model.predict(frame, conf=0.5, imgsz=256, verbose=False)

            for r in results:
                annotated = r.plot()
                evento = "none"

                if r.boxes is not None and r.keypoints is not None:
                    coords = r.boxes.xyxy.cpu().numpy()
                    kpts = r.keypoints.xy.cpu().numpy()

                    evento = self.detectar_eventos(coords, kpts)

                t = time.time()

                if evento == "golpe":
                    cv2.putText(annotated, "GOLPE DETECTADO",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)

                    if t - self.last_save_golpe >= SAVE_INTERVAL_GOLPE:
                        filename = os.path.join(self.output_dir, f"golpe_{int(t)}.jpg")
                        cv2.imwrite(filename, annotated)
                        print(f"🚨 GOLPE guardado: {filename}")
                        self.last_save_golpe = t

                elif evento == "pelea":
                    cv2.putText(annotated, "POSIBLE PELEA",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 2)

                    if t - self.last_save_pelea >= SAVE_INTERVAL_PELEA:
                        filename = os.path.join(self.output_dir, f"pelea_{int(t)}.jpg")
                        cv2.imwrite(filename, annotated)
                        print(f"⚠️ Pelea guardada: {filename}")
                        self.last_save_pelea = t

                cv2.imshow(self.window_name, annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app = DroneVanguardIA()
    app.run()