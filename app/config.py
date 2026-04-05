from __future__ import annotations

from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except Exception:
        return default


@dataclass(frozen=True)
class AppConfig:
    modo_dron: bool
    model_path: str
    window_name: str
    output_dir: str

    distancia_impacto: int
    velocidad_min_golpe: int
    frames_memoria: int
    umbral_aglomeracion: int

    save_interval_pelea: int
    save_interval_golpe: int
    save_interval_caido: int
    save_interval_aglomeracion: int

    kp_cabeza: int
    kp_cadera_izq: int
    kp_cadera_der: int
    kp_muneca_izq: int
    kp_muneca_der: int

    use_tracking: bool
    tracker_config: str

    validation_provider: str
    ai_threshold_confirmacion: float
    ai_timeout: int

    ollama_enabled: bool
    ollama_url: str
    ollama_model: str
    ollama_temperature: float

    gemini_enabled: bool
    gemini_model: str
    gemini_api_keys: list[str]
    gemini_timeout: int

    burst_frame_count: int
    burst_frame_gap_ms: int

    event_memory_seconds: int
    event_centroid_distance: float
    event_size_delta: float
    event_id_overlap_min: int

    cooldown_pelea: int
    cooldown_caido: int
    cooldown_aglomeracion: int

    telegram_enabled: bool
    telegram_bot_token: str
    telegram_chat_id: str

    drone_ip: str
    drone_cmd_port: int
    local_video_port: int
    socket_rcvbuf: int
    start_cmd: bytes
    stop_cmd: bytes

    @property
    def cooldowns(self) -> dict[str, int]:
        return {
            "pelea": self.cooldown_pelea,
            "caído": self.cooldown_caido,
            "aglomeración": self.cooldown_aglomeracion,
        }


def load_config() -> AppConfig:
    gemini_keys_raw = os.getenv("GEMINI_API_KEYS", "")
    gemini_keys = [k.strip() for k in gemini_keys_raw.split(",") if k.strip()]

    return AppConfig(
        modo_dron=_to_bool(os.getenv("MODO_DRON"), False),
        model_path=os.getenv("MODEL_PATH", "modelo/yolo11n-pose.pt"),
        window_name=os.getenv("WINDOW_NAME", "UCE Van-GuardIA"),
        output_dir=os.getenv("OUTPUT_DIR", "capturas"),

        distancia_impacto=_to_int(os.getenv("DISTANCIA_IMPACTO"), 100),
        velocidad_min_golpe=_to_int(os.getenv("VELOCIDAD_MIN_GOLPE"), 60),
        frames_memoria=_to_int(os.getenv("FRAMES_MEMORIA"), 5),
        umbral_aglomeracion=_to_int(os.getenv("UMBRAL_AGLOMERACION"), 4),

        save_interval_pelea=_to_int(os.getenv("SAVE_INTERVAL_PELEA"), 6),
        save_interval_golpe=_to_int(os.getenv("SAVE_INTERVAL_GOLPE"), 2),
        save_interval_caido=_to_int(os.getenv("SAVE_INTERVAL_CAIDO"), 6),
        save_interval_aglomeracion=_to_int(os.getenv("SAVE_INTERVAL_AGLOMERACION"), 10),

        kp_cabeza=_to_int(os.getenv("KP_CABEZA"), 0),
        kp_cadera_izq=_to_int(os.getenv("KP_CADERA_IZQ"), 11),
        kp_cadera_der=_to_int(os.getenv("KP_CADERA_DER"), 12),
        kp_muneca_izq=_to_int(os.getenv("KP_MUNECA_IZQ"), 9),
        kp_muneca_der=_to_int(os.getenv("KP_MUNECA_DER"), 10),

        use_tracking=_to_bool(os.getenv("USE_TRACKING"), True),
        tracker_config=os.getenv("TRACKER_CONFIG", "bytetrack.yaml"),

        validation_provider=os.getenv("VALIDATION_PROVIDER", "gemini").lower(),
        ai_threshold_confirmacion=_to_float(os.getenv("AI_THRESHOLD_CONFIRMACION"), 80.0),
        ai_timeout=_to_int(os.getenv("AI_TIMEOUT"), 60),

        ollama_enabled=_to_bool(os.getenv("OLLAMA_ENABLED"), True),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
        ollama_model=os.getenv("OLLAMA_MODEL", "moondream:1.8b-v2-q3_K_M"),
        ollama_temperature=_to_float(os.getenv("OLLAMA_TEMPERATURE"), 0.0),

        gemini_enabled=_to_bool(os.getenv("GEMINI_ENABLED"), True),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_api_keys=gemini_keys,
        gemini_timeout=_to_int(os.getenv("GEMINI_TIMEOUT"), 35),

        burst_frame_count=_to_int(os.getenv("BURST_FRAME_COUNT"), 3),
        burst_frame_gap_ms=_to_int(os.getenv("BURST_FRAME_GAP_MS"), 180),

        event_memory_seconds=_to_int(os.getenv("EVENT_MEMORY_SECONDS"), 35),
        event_centroid_distance=_to_float(os.getenv("EVENT_CENTROID_DISTANCE"), 0.18),
        event_size_delta=_to_float(os.getenv("EVENT_SIZE_DELTA"), 0.40),
        event_id_overlap_min=_to_int(os.getenv("EVENT_ID_OVERLAP_MIN"), 1),

        cooldown_pelea=_to_int(os.getenv("COOLDOWN_PELEA"), 20),
        cooldown_caido=_to_int(os.getenv("COOLDOWN_CAIDO"), 30),
        cooldown_aglomeracion=_to_int(os.getenv("COOLDOWN_AGLOMERACION"), 40),

        telegram_enabled=_to_bool(os.getenv("TELEGRAM_ENABLED"), True),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),

        drone_ip=os.getenv("DRONE_IP", "192.168.28.1"),
        drone_cmd_port=_to_int(os.getenv("DRONE_CMD_PORT"), 7080),
        local_video_port=_to_int(os.getenv("LOCAL_VIDEO_PORT"), 7070),
        socket_rcvbuf=_to_int(os.getenv("SOCKET_RCVBUF"), 8 * 1024 * 1024),
        start_cmd=bytes.fromhex(os.getenv("START_CMD_HEX", "cc 5a 01 82 02 36 b7")),
        stop_cmd=bytes.fromhex(os.getenv("STOP_CMD_HEX", "cc 5a 01 82 02 37 b6")),
    )
