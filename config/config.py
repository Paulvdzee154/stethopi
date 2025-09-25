from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Config:
    FS: int = int(os.getenv("FS", 15750))
    RECORD_SECONDS: int = int(os.getenv("RECORD_SECONDS", 5))
    COM_PORT: str = os.getenv("COM_PORT", "/dev/ttyACM0")
    BAUD: int = int(os.getenv("BAUD", 921600))
    READ_CHUNK_BYTES: int = int(os.getenv("READ_CHUNK_BYTES", 4096))
    AUDIO_BLOCK_SAMPLES: int = int(os.getenv("AUDIO_BLOCK_SAMPLES", 1024))
    SEGMENT_SECONDS: int = int(os.getenv("SEGMENT_SECONDS", 10))

    # Bundles / deployment
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    BUNDLES_DIR: str = os.getenv("BUNDLES_DIR", "models/bundles")
    DEPLOYMENT_PATH: str = os.getenv("DEPLOYMENT_PATH", "models/deployment.json")

COLORS = {
    "GREEN":  (0, 255, 0),      # Idle
    "YELLOW": (255, 255, 0),    # Record Xs
    "RED":    (255, 0, 0),      # Error
    "CYAN":   (0, 200, 255),    # Monitor
    "ORANGE": (255, 165, 0),    # Continuous record
    "WHITE":  (255, 255, 255),  # Predict border
}

# 10 klassekleuren (gebruik je eigen smaak als je wilt):
CLASS_COLORS = [
    (0, 0, 255),      # 0 - blue
    (255, 105, 180),  # 1 - pink
    (0, 200, 255),    # 2 - cyan
    (255, 140, 0),    # 3 - dark orange
    (128, 0, 128),    # 4 - purple
    (0, 128, 0),      # 5 - green
    (255, 0, 255),    # 6 - magenta
    (128, 128, 0),    # 7 - olive
    (255, 20, 147),   # 8 - deep pink
    (70, 130, 180),   # 9 - steel blue
]
