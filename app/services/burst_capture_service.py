from __future__ import annotations

import os
import time
import cv2

from app.config import AppConfig
from app.drone.video_source import VideoSource
from app.utils.files import ensure_dir


class BurstCaptureService:
    def __init__(self, config: AppConfig, video_source: VideoSource):
        self.config = config
        self.video_source = video_source
        ensure_dir(config.output_dir)

    def capture(self, annotated_frame, event_name: str) -> list[str]:
        burst_paths: list[str] = []
        timestamp_base = int(time.time() * 1000)

        first_path = os.path.join(self.config.output_dir, f"{event_name}_{timestamp_base}_0.jpg")
        cv2.imwrite(first_path, annotated_frame)
        burst_paths.append(first_path)

        for idx in range(1, self.config.burst_frame_count):
            time.sleep(self.config.burst_frame_gap_ms / 1000.0)
            frame_copy = self.video_source.get_last_frame_copy()
            if frame_copy is None:
                frame_copy = annotated_frame.copy()
            path = os.path.join(self.config.output_dir, f"{event_name}_{timestamp_base}_{idx}.jpg")
            cv2.imwrite(path, frame_copy)
            burst_paths.append(path)

        return burst_paths
