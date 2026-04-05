from __future__ import annotations

import queue
import socket
import threading
import cv2

from app.config import AppConfig
from app.drone.decoder import RobustDroneDecoder


class VideoSource:
    def __init__(self, config: AppConfig, frame_queue: queue.Queue):
        self.config = config
        self.frame_queue = frame_queue
        self.running = True
        self.last_frame = None
        self.last_frame_lock = threading.Lock()
        self.cap = None if config.modo_dron else cv2.VideoCapture(0)

    def stop(self) -> None:
        self.running = False
        if self.cap:
            self.cap.release()

    def get_last_frame_copy(self):
        with self.last_frame_lock:
            return None if self.last_frame is None else self.last_frame.copy()

    def _push_frame(self, frame):
        with self.last_frame_lock:
            self.last_frame = frame.copy()
        if self.frame_queue.full():
            self.frame_queue.get()
        self.frame_queue.put(frame)

    def run(self):
        if self.config.modo_dron:
            self._run_drone()
        else:
            self._run_webcam()

    def _run_drone(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.config.local_video_port))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.socket_rcvbuf)

        decoder = RobustDroneDecoder()
        sock.sendto(self.config.start_cmd, (self.config.drone_ip, self.config.drone_cmd_port))

        while self.running:
            try:
                packet, _ = sock.recvfrom(2048)
                image = decoder.process_packet(packet, len(packet))
                if image is not None:
                    self._push_frame(image)
            except Exception:
                continue

    def _run_webcam(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self._push_frame(frame)
