import socket
import time
import threading
import queue
import io
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# CONFIGURACIÓN
# ============================================================
DRONE_IP = "192.168.28.1"
DRONE_CMD_PORT = 7080
LOCAL_VIDEO_PORT = 7070

START_CAMERA_CMD = bytes.fromhex("cc 5a 01 82 02 36 b7")
STOP_CAMERA_CMD = bytes.fromhex("cc 5a 01 82 02 37 b6")

SOCKET_RCVBUF = 8 * 1024 * 1024
FRAME_TIMEOUT_SECONDS = 0.20
SHOW_EVERY_N_FRAMES = 2

MIN_JPEG_SIZE = 8000
MAX_JPEG_SIZE = 120000

ROTATE_180 = True
ROTATE_90_CLOCKWISE = False
FLIP_HORIZONTAL = False

# ============================================================
# ESTRUCTURAS---------
# ============================================================
# ESTRUCTURAS
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


# ============================================================
# DESOFUSCACIÓN DEL PAQUETE
# ============================================================
def deobfuscate_packet(packet: bytes, packet_len: int) -> bytes:
    """
    Basado en la lógica observada en s2.l:
    data[index + 6] ^= 0xFF
    donde index = (((data[0] * data[2]) + 10) * 6666) % (len - 8)
    """
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


# ============================================================
# FUNCIÓN RESPONSIVE PARA LA VENTANA
# ============================================================
def resize_with_aspect_ratio(frame: np.ndarray, window_name: str) -> np.ndarray:
    """
    Ajusta el frame al tamaño actual de la ventana manteniendo proporción.
    Si sobra espacio, rellena con negro.
    """
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return frame

    if win_w <= 1 or win_h <= 1:
        return frame

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame

    scale = min(win_w / w, win_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x_offset = (win_w - new_w) // 2
    y_offset = (win_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


# ============================================================
# DECODIFICADOR ROBUSTO
# ============================================================
class RobustDroneDecoder:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames: Dict[int, FrameAssembly] = {}

        self.frame_counter = 0
        self.good_frames = 0
        self.bad_frames = 0
        self.dropped_incomplete = 0
        self.dropped_seq_gap = 0
        self.dropped_decode = 0

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
            self._cleanup_expired_locked()

            frame = self.frames.get(frame_id)

            if seq == 1:
                frame = FrameAssembly(frame_id=frame_id)
                self.frames[frame_id] = frame

            if frame is None:
                return None

            if frame.invalid:
                return None

            if seq < frame.expected_next_seq and seq != 1:
                frame.invalid = True
                self.dropped_seq_gap += 1
                return None

            if seq != frame.expected_next_seq:
                frame.invalid = True
                self.dropped_seq_gap += 1
                return None

            frame.parts[seq] = payload
            frame.expected_next_seq = seq + 1
            frame.started_at = time.time()

            if flag == 0x01:
                frame.last_seq = seq
                frame.closed = True

            if not frame.closed:
                return None

            if frame.last_seq is None:
                frame.invalid = True
                self.dropped_incomplete += 1
                return None

            for i in range(1, frame.last_seq + 1):
                if i not in frame.parts:
                    frame.invalid = True
                    self.dropped_incomplete += 1
                    return None

            jpeg_bytes = self._build_jpeg(frame)
            del self.frames[frame_id]

        if jpeg_bytes is None:
            self.bad_frames += 1
            return None

        self.frame_counter += 1
        if self.frame_counter % SHOW_EVERY_N_FRAMES != 0:
            return None

        img = self._decode_jpeg(jpeg_bytes)
        if img is None:
            self.bad_frames += 1
            self.dropped_decode += 1
            return None

        img = self._postprocess(img)
        self.good_frames += 1
        return img

    def _cleanup_expired_locked(self):
        now = time.time()
        expired = [
            frame_id
            for frame_id, frame in self.frames.items()
            if now - frame.started_at > FRAME_TIMEOUT_SECONDS
        ]
        for frame_id in expired:
            del self.frames[frame_id]
            self.dropped_incomplete += 1

    def _build_jpeg(self, frame: FrameAssembly) -> Optional[bytes]:
        data = b"".join(frame.parts[i] for i in range(1, frame.last_seq + 1))

        soi = data.find(b"\xff\xd8")
        eoi = data.rfind(b"\xff\xd9")

        if soi == -1 or eoi == -1 or eoi <= soi:
            return None

        jpeg = data[soi:eoi + 2]

        if len(jpeg) < MIN_JPEG_SIZE or len(jpeg) > MAX_JPEG_SIZE:
            return None
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            return None

        return jpeg

    def _decode_jpeg(self, jpeg: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img

        try:
            pil_img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            rgb = np.array(pil_img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _postprocess(self, img: np.ndarray) -> np.ndarray:
        if ROTATE_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if ROTATE_90_CLOCKWISE:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if FLIP_HORIZONTAL:
            img = cv2.flip(img, 1)
        return img


# ============================================================
# CLIENTE UDP
# ============================================================
class DroneUDPClient:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
        self.sock.bind(("0.0.0.0", LOCAL_VIDEO_PORT))
        self.sock.settimeout(1.0)

    def send_start_camera(self):
        self.sock.sendto(START_CAMERA_CMD, (DRONE_IP, DRONE_CMD_PORT))

    def send_stop_camera(self):
        self.sock.sendto(STOP_CAMERA_CMD, (DRONE_IP, DRONE_CMD_PORT))

    def recv_packet(self):
        try:
            return self.sock.recvfrom(1500)
        except socket.timeout:
            return None, None

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

# ============================================================
# APP
# ============================================================
class DroneVideoApp:
    def __init__(self):
        self.decoder = RobustDroneDecoder()
        self.client = DroneUDPClient()

        self.running = True
        self.last_good_frame: Optional[np.ndarray] = None
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

        self.recv_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self.stats_thread = threading.Thread(target=self._stats_loop, daemon=True)

        self.window_name = "Drone Video"

    def start(self):
        print("Enviando comando de cámara...")
        self.client.send_start_camera()
        print("Esperando video...")

        self.recv_thread.start()
        self.stats_thread.start()
        self._viewer_loop()

    def stop(self):
        self.running = False
        try:
            self.client.send_stop_camera()
        except Exception:
            pass
        self.client.close()
        cv2.destroyAllWindows()

    def _receiver_loop(self):
        while self.running:
            packet, addr = self.client.recv_packet()
            if packet is None or addr is None:
                continue

            ip, port = addr
            if ip != DRONE_IP:
                continue
            if port != 7070:
                continue

            img = self.decoder.process_packet(packet, len(packet))
            if img is None:
                continue

            try:
                self.frame_queue.put_nowait(img)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(img)
                except queue.Full:
                    pass

    def _viewer_loop(self):
        saved_count = 0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 540)

        while self.running:
            try:
                img = self.frame_queue.get(timeout=1.0)
                self.last_good_frame = img
            except queue.Empty:
                img = self.last_good_frame

            if img is None:
                continue

            display_img = resize_with_aspect_ratio(img, self.window_name)
            cv2.imshow(self.window_name, display_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                self.stop()
                break
            elif key == ord("s"):
                filename = f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(filename, img)
                print(f"Captura guardada: {filename}")
                saved_count += 1

    def _stats_loop(self):
        while self.running:
            time.sleep(5)
            print(
                f"buenos={self.decoder.good_frames} | "
                f"malos={self.decoder.bad_frames} | "
                f"drop_incomplete={self.decoder.dropped_incomplete} | "
                f"drop_seq_gap={self.decoder.dropped_seq_gap} | "
                f"drop_decode={self.decoder.dropped_decode}"
            )


if __name__ == "__main__":
    app = DroneVideoApp()
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()