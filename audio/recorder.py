import numpy as np
import os
import time
from datetime import datetime
from scipy.io.wavfile import write


class Recorder:
    def __init__(self, fs: int, reader_factory, output_dir_fn):
        self.fs = fs
        self.reader_factory = reader_factory
        self.output_dir_fn = output_dir_fn


    def record_seconds(self, seconds: int) -> str | None:
        buf = bytearray()
        start = time.monotonic()
        try:
            with self.reader_factory() as reader:
                while time.monotonic() - start < seconds:
                    raw = reader.read_bytes()
                    if raw:
                        buf.extend(raw)
        except Exception as e:
            print(f"[record] Serial error: {e}")
            return None


        n = len(buf) // 2 # int16 mono
        if n == 0:
            print("[record] No data; got 0 samples")
            return None
        data = np.frombuffer(memoryview(buf)[: n * 2], dtype=np.int16)
        ts = datetime.now().strftime("T%H_%M_%S_D%d_%m_%Y")
        outdir = self.output_dir_fn()
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{ts}.wav")
        try:
            write(path, self.fs, data)
        except Exception as e:
            print(f"[record] Save failed: {e}")
            return None
        print(f"[record] Saved: {path} ({n} samples, {(n/self.fs):.2f}s)")
        return path