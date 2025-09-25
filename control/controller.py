# control/controller.py
import os
import sys
import threading
import time
from typing import Callable, Tuple, Optional, List

class Controller:
    def __init__(
        self,
        ui,                               # SenseUI
        recorder,                         # Recorder
        segment_recorder,                 # SegmentRecorder
        monitor,                          # Monitor
        load_model_bundle_fn: Callable[[], Tuple[object, Callable[[], "np.ndarray"]]],
        seconds: int,
        segment_seconds: int,
        class_palette: List[tuple]
    ):
        self.ui = ui
        self.recorder = recorder
        self.segment_rec = segment_recorder
        self.monitor = monitor
        self.load_model_bundle_fn = load_model_bundle_fn
        self.seconds = seconds
        self.segment_seconds = segment_seconds
        self._palette = class_palette

        self.state = "idle"

        # Predict runtime (lazy init)
        self._predictor = None
        self._get_input_fn = None
        self._model_loaded = False
        self._pred_running = False
        self._pred_thread: Optional[threading.Thread] = None

        # Error info
        self._last_error: Optional[str] = None

    # ---------------- Lifecycle helpers ----------------

    def _stop_all(self):
        """Stop veilig alle mogelijke lopende activiteiten/threads."""
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
        self._stop_all()
        self.state = "idle"
        self.ui.fill("GREEN")

    # ---------------- Error handling ----------------

    def raise_error(self, err):
        """Centrale error handler → stop alles, LED RED, error-state."""
        try:
            self._last_error = str(err)
            print(f"[ERR] {self._last_error}")
        finally:
            self._stop_all()
            self.state = "error"
            self.ui.fill("RED")

    def restart(self):
        """Harde restart van het huidige Python proces (execv)."""
        print("[ctl] restarting process…")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # ---------------- Joystick handlers ----------------

    def on_middle(self):
        if self.state == "error":
            self.restart()
            return
        print("[ctl] -> idle")
        self.to_idle()

    def on_up(self):
        if self.state == "error":
            self.restart()
            return
        if self.state != "idle":
            return

        print("[ctl] record start")
        self.state = "recording"
        self.ui.fill("YELLOW")
        try:
            path = self.recorder.record_seconds(self.seconds)
            self.ui.fill("GREEN" if path else "RED")
            # Als je 'geen data' als soft-fout wilt behandelen, zet dan gewoon terug naar idle:
            self.state = "idle"
        except Exception as e:
            self.raise_error(e)

    def on_left(self):
        if self.state == "error":
            self.restart()
            return

        if self.state == "monitoring":
            print("[ctl] stop monitor")
            try:
                self.monitor.stop()
                self.ui.fill("GREEN")
                self.state = "idle"
            except Exception as e:
                self.raise_error(e)
            return

        if self.state != "idle":
            return

        print("[ctl] start monitor")
        self.state = "monitoring"
        self.ui.fill("CYAN")
        try:
            self.monitor.start()
        except Exception as e:
            self.raise_error(e)

    def on_right(self):
        if self.state == "error":
            self.restart()
            return

        if self.state == "continuous":
            print("[ctl] stop continuous record")
            try:
                self.segment_rec.stop()
                self.ui.fill("GREEN")
                self.state = "idle"
            except Exception as e:
                self.raise_error(e)
            return

        if self.state != "idle":
            return

        print("[ctl] start continuous record")
        self.state = "continuous"
        self.ui.fill("ORANGE")
        try:
            self.segment_rec.start()
        except Exception as e:
            self.raise_error(e)

    def on_down(self):
        if self.state == "error":
            self.restart()
            return

        # Toggle stop predict
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

        # Alleen vanuit idle starten
        if self.state != "idle":
            return

        # Lazy load van het model/bundle
        if not self._model_loaded:
            try:
                print("[ctl] loading model bundle…")
                self._predictor, self._get_input_fn = self.load_model_bundle_fn()
                self._model_loaded = True
                print("[ctl] model bundle loaded")
            except Exception as e:
                # Hard fail zoals besproken: RED + error-state
                self.raise_error(e)
                return

        print("[ctl] start predict")
        self.state = "predict"
        self._pred_running = True

        def loop():
            while self._pred_running:
                try:
                    cls_idx, probs = self._predictor.predict_one(self._get_input_fn)
                    self.ui.draw_pred_class(cls_idx, self._palette, border_name="WHITE")
                except Exception as e:
                    self.raise_error(e)
                    return
                time.sleep(0.01)  # CPU sparen; stem af op je pipeline

        try:
            self._pred_thread = threading.Thread(target=loop, daemon=True)
            self._pred_thread.start()
        except Exception as e:
            self.raise_error(e)
