from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from app.config import AppConfig


class PersonPoseDetector:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = YOLO(config.model_path)

    def infer(self, frame):
        if self.config.use_tracking:
            return self.model.track(
                frame,
                conf=0.5,
                imgsz=256,
                persist=True,
                tracker=self.config.tracker_config,
                verbose=False,
            )
        return self.model.predict(frame, conf=0.5, imgsz=256, verbose=False)

    @staticmethod
    def extract_arrays(result):
        coords = result.boxes.xyxy.cpu().numpy()
        kpts = result.keypoints.xy.cpu().numpy()
        if hasattr(result.boxes, "id") and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.full((len(coords),), -1, dtype=int)
        return coords, kpts, track_ids
