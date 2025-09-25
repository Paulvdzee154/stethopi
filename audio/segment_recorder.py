import os, threading, numpy as np
from datetime import datetime
from scipy.io.wavfile import write

class SegmentRecorder:
    def __init__(self, fs: int, reader_factory, output_dir_fn, segment_seconds: int = 10, on_error=None):
        self.fs = fs
        self.reader_factory = reader_factory
        self.output_dir_fn = output_dir_fn
        self.segment_seconds = segment_seconds
        self.segment_bytes = fs * segment_seconds * 2
        self.on_error = on_error
        self._stop = threading.Event()
        self._t = None
        self._buf = bytearray()

    def _write_segment(self, seg_bytes: bytes):
        n = len(seg_bytes) // 2
        if n == 0: return
        data = np.frombuffer(seg_bytes, dtype=np.int16)
        ts = datetime.now().strftime("T%H_%M_%S_D%d_%m_%Y")
        outdir = self.output_dir_fn(); os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{ts}_seg.wav")
        write(path, self.fs, data)
        print(f"[segrec] Saved: {path} ({n} samples, {(n/self.fs):.2f}s)")

    def _loop(self):
        try:
            with self.reader_factory() as reader:
                while not self._stop.is_set():
                    raw = reader.read_bytes()
                    if raw:
                        self._buf.extend(raw)
                        while len(self._buf) >= self.segment_bytes:
                            seg = bytes(self._buf[:self.segment_bytes])
                            del self._buf[:self.segment_bytes]
                            self._write_segment(seg)
        except Exception as e:
            if self.on_error: self.on_error(e)
            self._stop.set()

    def start(self):
        try:
            if self._t and self._t.is_alive(): return
            self._stop.clear()
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
            print("[segrec] started")
        except Exception as e:
            if self.on_error: self.on_error(e)
            self.stop()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0); self._t = None
        self._buf.clear()
        print("[segrec] stopped")
