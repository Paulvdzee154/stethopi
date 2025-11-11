from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Config:
    """
    Configuration settings for the audio project.
    Values can be overridden with environment variables if needed.
    """
    # Sample rate (Hz) used for all recordings
    FS: int = int(os.getenv("FS", 15750))

    # Duration (in seconds) for one short fixed-length recording
    RECORD_SECONDS: int = int(os.getenv("RECORD_SECONDS", 5))

    # Serial port name for communication with the microcontroller
    COM_PORT: str = os.getenv("COM_PORT", "/dev/ttyACM0")

    # Serial connection speed (baud rate)
    BAUD: int = int(os.getenv("BAUD", 921600))

    # Number of bytes to read from the serial stream per chunk
    READ_CHUNK_BYTES: int = int(os.getenv("READ_CHUNK_BYTES", 4096))

    # Number of samples in one small audio block (used for playback buffering)
    AUDIO_BLOCK_SAMPLES: int = int(os.getenv("AUDIO_BLOCK_SAMPLES", 1024))

    # Length (in seconds) of one segment in continuous recording mode
    SEGMENT_SECONDS: int = int(os.getenv("SEGMENT_SECONDS", 10))

    # Default output device name for audio playback (e.g., “Headphones”)
    AUDIO_OUT_DEVICE: str = os.getenv("AUDIO_OUT_DEVICE", "Headphones")

    # Model and deployment paths for inference bundles
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    BUNDLES_DIR: str = os.getenv("BUNDLES_DIR", "models/bundles")
    DEPLOYMENT_PATH: str = os.getenv("DEPLOYMENT_PATH", "models/deployment.json")


# Colors used for LED matrix feedback (Sense HAT)
COLORS = {
    "GREEN":  (0, 255, 0),      # Idle or ready
    "YELLOW": (255, 255, 0),    # Recording fixed-length segment
    "RED":    (255, 0, 0),      # Error or stop
    "CYAN":   (0, 200, 255),    # Live monitoring
    "ORANGE": (255, 165, 0),    # Continuous segment recording
    "WHITE":  (255, 255, 255),  # Prediction border / neutral color
}

# 10 class colors used for different prediction categories
# (Feel free to customize the palette)
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
