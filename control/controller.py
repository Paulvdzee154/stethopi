# control/controller.py
import os
import sys
import threading
import time
from typing import Callable, Tuple, Optional, List
import numpy as np

class Controller:
    """
    Connects the UI (Sense HAT LEDs) with the audio features:
      - Short recording (Up)
      - Live monitor (Left)
      - Continuous segmented recording (Right)
      - Prediction loop (Down)
    Manages state transitions, starts/stops workers, and handles errors.
    """
    def __init__(
        self,
        ui,                               # SenseUI object that controls the LED matrix
        recorder,                         # Recorder for fixed-duration recordings
        segment_recorder,                 # SegmentRecorder for continuous segments
        monitor,                          # Monitor for low-latency live listening
        load_model_bundle_fn: Callable[[], Tuple[object, Callable[[], "np.ndarray"]]],
        seconds: int,
        segment_seconds: int,
        class_palette: List[tuple]
    ):
        # Save dependencies and configuration
        self.ui = ui
        self.recorder = recorder
        self.segment_rec = segment_recorder
        self.monitor = monitor
        self.load_model_bundle_fn = load_model_bundle_fn
        self.seconds = seconds
        self.segment_seconds = segment_seconds
        self._palette = class_palette

        # Current high-level state of the controller
        self.state = "idle"

        # Prediction runtime state (lazy-initialized when first used)
        self._predictor = None            # model wrapper with .predict_one(...)
        self._get_input_fn = None         # callable that returns the model input
        self._model_loaded = False        # set True after lazy load succeeds
        self._pred_running = False        # flag for the prediction loop
        self._pred_thread: Optional[threading.Thread] = None

        # Last error text for debugging/diagnostics
        self._last_error: Optional[str] = None

    # Lifecycle helpers

    def _stop_all(self):
        """Safely stop any running activity/threads (monitor, segments, predict)."""
        try:
            if self.state == "monitoring":
                self.monitor.stop()
        except Exception:
            pass
        try:
            if self.state == "continuous":
                self.segment_rec.stop()
        except Exception:
            pass
        try:
            if self.state == "predict":
                self._pred_running = False
                if self._pred_thread:
                    self._pred_thread.join(timeout=1.0)
                    self._pred_thread = None
        except Exception:
            pass

    def to_idle(self):
        """Stop everything and set LEDs to GREEN to indicate 'ready/idle'."""
        self._stop_all()
        self.state = "idle"
        self.ui.fill("GREEN")

    # Error handling

    def raise_error(self, err):
        """Central error handler → stop all, set LEDs RED, enter 'error' state."""
        try:
            self._last_error = str(err)
            print(f"[ERR] {self._last_error}")
        finally:
            self._stop_all()
            self.state = "error"
            self.ui.fill("RED")

    def restart(self):
        """Hard restart of the current Python process (execv)."""
        print("[ctl] restarting process…")
        try:
            # Flush streams so logs are not lost on restart
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # Replace current process with a fresh interpreter running this script
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # Joystick handlers

    def on_middle(self):
        """Middle press: from error → restart; otherwise go to idle."""
        if self.state == "error":
            self.restart()
            return
        print("[ctl] -> idle")
        self.to_idle()

    def on_up(self):
        """Up press: record a fixed number of seconds once, then return to idle."""
        if self.state == "error":
            self.restart()
            return
        if self.state != "idle":
            return

        print("[ctl] record start")
        self.state = "recording"
        self.ui.fill("YELLOW")
        try:
            # Perform a blocking recording; returns path or None
            path = self.recorder.record_seconds(self.seconds)
            # Show GREEN if saved, RED if failed/empty; then return to idle
            self.ui.fill("GREEN" if path else "RED")
            self.state = "idle"
        except Exception as e:
            self.raise_error(e)

    def on_left(self):
        """Left press: toggle live monitoring on/off."""
        if self.state == "error":
            self.restart()
            return

        # If already monitoring → stop it
        if self.state == "monitoring":
            print("[ctl] stop monitor")
            try:
                self.monitor.stop()
                self.ui.fill("GREEN")
                self.state = "idle"
            except Exception as e:
                self.raise_error(e)
            return

        # Only start if currently idle
        if self.state != "idle":
            return

        print("[ctl] start monitor")
        self.state = "monitoring"
        self.ui.fill("CYAN")
        try:
            # Non-blocking start; Monitor manages its own threads/stream
            self.monitor.start()
        except Exception as e:
            self.raise_error(e)

    def on_right(self):
        """Right press: toggle continuous segment recording on/off."""
        if self.state == "error":
            self.restart()
            return

        # If already recording segments → stop it
        if self.state == "continuous":
            print("[ctl] stop continuous record")
            try:
                self.segment_rec.stop()
                self.ui.fill("GREEN")
                self.state = "idle"
            except Exception as e:
                self.raise_error(e)
            return

        # Only start if idle
        if self.state != "idle":
            return

        print("[ctl] start continuous record")
        self.state = "continuous"
        self.ui.fill("ORANGE")
        try:
            # Starts a background thread that slices and saves segments
            self.segment_rec.start()
        except Exception as e:
            self.raise_error(e)

    def on_down(self):
        """Down press: toggle prediction loop (load model on first use)."""
        if self.state == "error":
            self.restart()
            return

        # If prediction is running → stop it
        if self.state == "predict":
            print("[ctl] stop predict")
            try:
                self._pred_running = False
                if self._pred_thread:
                    self._pred_thread.join(timeout=1.0)
                    self._pred_thread = None
                self.state = "idle"
                self.ui.fill("GREEN")
            except Exception as e:
                self.raise_error(e)
            return

        # Only start prediction from idle
        if self.state != "idle":
            return

        # Lazy load the model/bundle (only the first time)
        if not self._model_loaded:
            try:
                print("[ctl] loading model bundle…")
                self._predictor, self._get_input_fn = self.load_model_bundle_fn()
                self._model_loaded = True
                print("[ctl] model bundle loaded")
            except Exception as e:
                # Hard fail as discussed: set RED and enter error state
                self.raise_error(e)
                return

        print("[ctl] start predict")
        self.state = "predict"
        self._pred_running = True

        def loop():
            # Background loop: fetch input → predict → draw LEDs
            while self._pred_running:
                try:
                    cls_idx, probs = self._predictor.predict_one(self._get_input_fn)
                    # Draw predicted class with a colored fill/border
                    self.ui.draw_pred_class(cls_idx, self._palette, border_name="WHITE")
                except Exception as e:
                    self.raise_error(e)
                    return
                # Small sleep to keep CPU usage reasonable; tune for your pipeline
                time.sleep(0.01)

        try:
            # Start prediction loop in a daemon thread
            self._pred_thread = threading.Thread(target=loop, daemon=True)
            self._pred_thread.start()
        except Exception as e:
            self.raise_error(e)
