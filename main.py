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
    usb_base = '/media/LongPi'
    try:
        mounts = [f for f in os.listdir(usb_base) if os.path.ismount(os.path.join(usb_base, f))]
    except FileNotFoundError:
        mounts = []
    return os.path.join(usb_base, mounts[0], 'data') if mounts else "data"

def build_controller() -> Controller:
    cfg = Config()
    ui = SenseUI(COLORS)
    reader_factory = lambda: SerialAudioReader(cfg.COM_PORT, cfg.BAUD, cfg.READ_CHUNK_BYTES)

    rec = Recorder(cfg.FS, reader_factory, get_output_dir, out_device=cfg.AUDIO_OUT_DEVICE, blocksize=cfg.AUDIO_BLOCK_SAMPLES)
    segrec = SegmentRecorder(cfg.FS, reader_factory, get_output_dir, segment_seconds=cfg.SEGMENT_SECONDS, out_device=cfg.AUDIO_OUT_DEVICE, blocksize=cfg.AUDIO_BLOCK_SAMPLES)
    mon = Monitor(cfg.FS, reader_factory, cfg.AUDIO_BLOCK_SAMPLES, out_device=cfg.AUDIO_OUT_DEVICE)

    # LAZY loader: wordt pas bij eerste predict uitgevoerd
    def load_model_bundle_fn():
        mbm = ModelBundleManager(cfg, reader_factory)
        mbm.load_active_bundle()                 # kan FileNotFoundError etc. gooien → controller vangt en toont RED
        predictor = mbm.predictor
        get_input_fn = mbm.get_input_fn()
        return predictor, get_input_fn

    ctl = Controller(
        ui=ui,
        recorder=rec,
        segment_recorder=segrec,
        monitor=mon,
        load_model_bundle_fn=load_model_bundle_fn,   # ← hier
        seconds=cfg.RECORD_SECONDS,
        segment_seconds=cfg.SEGMENT_SECONDS,
        class_palette=CLASS_COLORS
    )

    # Error callbacks
    mon.on_error = ctl.raise_error
    segrec.on_error = ctl.raise_error

    return ctl

if __name__ == "__main__":
    ctl = build_controller()

    def handle_event(event):
        if getattr(event, "action", None) != "pressed":
            return
        d = getattr(event, "direction", None)
        if d == "middle": ctl.on_middle()
        elif d == "up":   ctl.on_up()
        elif d == "down": ctl.on_down()       # ← hier wordt pas geprobeerd te laden
        elif d == "left": ctl.on_left()
        elif d == "right":ctl.on_right()

    SenseHat().stick.direction_any = handle_event
    ctl.to_idle()
    print("🟢 Idle — model wordt pas geladen bij DOWN (predict).")
    while True:
        time.sleep(0.1)