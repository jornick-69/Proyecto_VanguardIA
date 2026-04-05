from __future__ import annotations

import threading
import cv2
import numpy as np

from app.models import FrameAssembly


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
        self.frames: dict[int, FrameAssembly] = {}

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
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.rotate(image, cv2.ROTATE_180)
