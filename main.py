import time
import socket
import threading
import os
import io
import sys
import queue
from dataclasses import dataclass, field
from typing import Dict, Optional

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODO_DRON = False  # True = Drone | False = Webcam
#MODEL_PATH = r"C:\Users\elvis\Downloads\yolo11n-pose.pt"
MODEL_PATH = "Modelo/yolo11n-pose.pt"

print(f"Iniciando Vanguardia UCE | Modo: {'DRON' if MODO_DRON else 'WEBCAM'}")

import cv2
import numpy as np
from PIL import Image, ImageFile

os.environ["YOLO_VERBOSE"] = "False"
os.environ["YOLO_OFFLINE"] = "True"

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
# DECODER (EL QUE SÍ FUNCIONA)
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

        if seq <= 0 or seq > 200:
            return None

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

        print("Cargando modelo YOLO...")
        self.model = YOLO(MODEL_PATH)

        if not MODO_DRON:
            self.cap = cv2.VideoCapture(0)

    # ========================================================
    # HILO CAPTURA
    # ========================================================
    def video_receiver(self):
        if MODO_DRON:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
            sock.bind(("0.0.0.0", LOCAL_VIDEO_PORT))

            decoder = RobustDroneDecoder()

            sock.sendto(START_CMD, (DRONE_IP, DRONE_CMD_PORT))
            print("📡 Dron conectado")

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
    # MAIN LOOP (IA + DISPLAY)
    # ========================================================
    def run(self):
        threading.Thread(target=self.video_receiver, daemon=True).start()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 540)

        print("Presiona 'q' para salir")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except:
                continue

            # IA
            results = self.model.predict(frame, conf=0.5, imgsz=256, verbose=False)

            for r in results:
                annotated = r.plot()

                count = len(r.boxes)
                if count >= 2:
                    cv2.putText(annotated, "ALERTA CONFLICTO",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

                cv2.imshow(self.window_name, annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        if not MODO_DRON:
            self.cap.release()

        cv2.destroyAllWindows()
        print("Finalizado")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app = DroneVanguardIA()
    app.run()