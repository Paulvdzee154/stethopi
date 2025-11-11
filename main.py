from config.config import Config, COLORS, CLASS_COLORS
from ui.sense_ui import SenseUI, SenseHat
from iodev.serial_stream import SerialAudioReader
from audio.recorder import Recorder
from audio.segment_recorder import SegmentRecorder
from audio.monitor import Monitor
from models.manager import ModelBundleManager
from control.controller import Controller
import os, time


def get_output_dir() -> str:
    """
    Determine where to save audio recordings.
    - If a USB drive is mounted under /media/LongPi, use its 'data' folder.
    - Otherwise, use a local folder called 'data'.
    """
    usb_base = '/media/LongPi'
    try:
        # List all mounted subfolders in /media/LongPi
        mounts = [f for f in os.listdir(usb_base) if os.path.ismount(os.path.join(usb_base, f))]
    except FileNotFoundError:
        # If /media/LongPi doesn’t exist, just fall back to local storage
        mounts = []

    # Use first USB mount if available, otherwise "data" in local directory
    return os.path.join(usb_base, mounts[0], 'data') if mounts else "data"


def build_controller() -> Controller:
    """
    Build and connect all core components:
      - Config: global settings
      - SenseUI: LED matrix interface
      - SerialAudioReader: reads audio bytes from microcontroller
      - Recorder / SegmentRecorder / Monitor: handle audio capture and playback
      - ModelBundleManager: lazy-loaded model manager
      - Controller: connects joystick actions to all the above
    """
    cfg = Config()
    ui = SenseUI(COLORS)

    # Reader factory → creates a new SerialAudioReader each time it’s used
    reader_factory = lambda: SerialAudioReader(cfg.COM_PORT, cfg.BAUD, cfg.READ_CHUNK_BYTES)

    # Create the three audio workers
    rec = Recorder(
        cfg.FS, reader_factory, get_output_dir,
        out_device=cfg.AUDIO_OUT_DEVICE,
        blocksize=cfg.AUDIO_BLOCK_SAMPLES
    )

    segrec = SegmentRecorder(
        cfg.FS, reader_factory, get_output_dir,
        segment_seconds=cfg.SEGMENT_SECONDS,
        out_device=cfg.AUDIO_OUT_DEVICE,
        blocksize=cfg.AUDIO_BLOCK_SAMPLES
    )

    mon = Monitor(
        cfg.FS, reader_factory, cfg.AUDIO_BLOCK_SAMPLES,
        out_device=cfg.AUDIO_OUT_DEVICE
    )

    # Lazy model loader — only runs when prediction mode starts (DOWN press)
    def load_model_bundle_fn():
        mbm = ModelBundleManager(cfg, reader_factory)
        # This may raise FileNotFoundError etc. → Controller handles it and sets RED LED
        mbm.load_active_bundle()
        predictor = mbm.predictor
        get_input_fn = mbm.get_input_fn()
        return predictor, get_input_fn

    # Create the main controller object, wiring together all modules
    ctl = Controller(
        ui=ui,
        recorder=rec,
        segment_recorder=segrec,
        monitor=mon,
        load_model_bundle_fn=load_model_bundle_fn,
        seconds=cfg.RECORD_SECONDS,
        segment_seconds=cfg.SEGMENT_SECONDS,
        class_palette=CLASS_COLORS
    )

    # Connect error callbacks for audio components → report to controller
    mon.on_error = ctl.raise_error
    segrec.on_error = ctl.raise_error

    return ctl


if __name__ == "__main__":
    # Build and start the system
    ctl = build_controller()

    def handle_event(event):
        """
        Handle Sense HAT joystick input.
        - Only respond to press events (not hold or release)
        - Map directions to controller actions
        """
        if getattr(event, "action", None) != "pressed":
            return
        d = getattr(event, "direction", None)
        if d == "middle":
            ctl.on_middle()
        elif d == "up":
            ctl.on_up()
        elif d == "down":
            ctl.on_down()   # Model loading happens here (lazy)
        elif d == "left":
            ctl.on_left()
        elif d == "right":
            ctl.on_right()

    # Register joystick event handler
    SenseHat().stick.direction_any = handle_event

    # Set LEDs to idle (green) and show ready message
    ctl.to_idle()
    print("🟢 Idle — model loads only on DOWN (predict).")

    # Keep the program running indefinitely
    while True:
        time.sleep(0.1)
