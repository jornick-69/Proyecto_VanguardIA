from __future__ import annotations

import base64
import os


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as file_obj:
        return base64.b64encode(file_obj.read()).decode("utf-8")
