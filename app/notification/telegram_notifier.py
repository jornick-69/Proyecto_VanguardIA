from __future__ import annotations

import requests

from app.config import AppConfig
from app.models import ValidationResult


class TelegramNotifier:
    def __init__(self, config: AppConfig):
        self.config = config

    def send_alert(self, image_path: str, evento_local: str, result: ValidationResult) -> None:
        if not self.config.telegram_enabled:
            print("[TELEGRAM] Deshabilitado")
            return

        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendPhoto"
        caption = (
            f"🚨 <b>ALERTA DETECTADA</b>\n"
            f"<b>Evento local:</b> {evento_local}\n"
            #f"<b>Evento confirmado:</b> {result.evento_detectado}\n"
            f"<b>Confianza:</b> {result.confianza}%\n"
            f"<b>Proveedor:</b> Van-GuardIA\n"
            f"<b>Detalle:</b> {result.explicacion}"
        )

        with open(image_path, "rb") as photo:
            response = requests.post(
                url,
                data={
                    "chat_id": self.config.telegram_chat_id,
                    "caption": caption,
                    "parse_mode": "HTML",
                },
                files={"photo": photo},
                timeout=30,
            )
            response.raise_for_status()

        print(f"[TELEGRAM] Alerta enviada: {image_path}")
