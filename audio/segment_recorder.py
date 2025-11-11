import os, threading, numpy as np
from datetime import datetime
from scipy.io.wavfile import write
from .recorder import AudioTap  # NOTE: relative import within the same package

class SegmentRecorder:
    """
    Continuously reads raw int16 mono bytes from a reader and:
      - Plays them live to a chosen output device (monitoring)
      - Slices the stream into fixed-length segments
      - Saves each segment as a WAV file with a timestamped name
    """
    def __init__(self, fs: int, reader_factory, output_dir_fn,
                 segment_seconds: int = 10, on_error=None, out_device=None, blocksize=1024):
        # Save configuration parameters
        self.fs = fs                                # sample rate (Hz)
        self.reader_factory = reader_factory        # callable -> context manager with read_bytes()
        self.output_dir_fn = output_dir_fn          # callable that returns directory path for output
        self.segment_seconds = segment_seconds      # length of each WAV segment in seconds
        self.segment_bytes = fs * segment_seconds * 2  # bytes per segment (2 bytes per int16 sample)
        self.on_error = on_error                    # optional error callback
        self.out_device = out_device                # output device index/name for live monitoring
        self.blocksize = blocksize                  # audio block size used by AudioTap

        # Runtime state
        self._stop = threading.Event()              # set to request the worker thread to stop
        self._t = None                              # background worker thread handle
        self._buf = bytearray()                     # rolling buffer of raw bytes awaiting segmentation
        self._tap = None                            # AudioTap for live monitoring

    # Convert raw bytes into int16 samples and save as a timestamped WAV file
    def _write_segment(self, raw_bytes: bytes):
        n = len(raw_bytes) // 2                     # number of int16 samples in the chunk
        if n == 0:
            return
        data = np.frombuffer(memoryview(raw_bytes)[: n * 2], dtype=np.int16)

        # Build output path: directory exists or is created; filename includes time and date
        ts = datetime.now().strftime("T%H_%M_%S_D%d_%m_%Y")
        outdir = self.output_dir_fn()
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"seg_{ts}.wav")

        # Write the WAV file with the configured sample rate
        write(path, self.fs, data)
        print(f"[segrec] Saved: {path} ({n} samples, {(n/self.fs):.2f}s)")

    # Main background loop: read -> optional monitor -> accumulate -> slice -> save
    def _loop(self):
        import time
        try:
            # Open the reader that provides raw audio bytes
            with self.reader_factory() as reader:
                # Start live monitoring on the chosen output device
                self._tap = AudioTap(self.fs, device=self.out_device, blocksize=self.blocksize)

                # Keep pulling data until stop is requested
                while not self._stop.is_set():
                    raw = reader.read_bytes()
                    if raw:
                        # Hear the audio while segmenting (live monitoring)
                        if self._tap:
                            self._tap.write(raw)
                        # Add incoming bytes to the rolling buffer
                        self._buf.extend(raw)

                        # If buffer has at least one full segment, cut and save it
                        while len(self._buf) >= self.segment_bytes:
                            seg = bytes(self._buf[:self.segment_bytes])  # copy exact segment
                            del self._buf[:self.segment_bytes]           # remove it from buffer
                            self._write_segment(seg)
        except Exception as e:
            # Report error to caller (if provided) and request stop
            if self.on_error:
                self.on_error(e)
            self._stop.set()
        finally:
            # Always close the monitoring stream on exit
            if self._tap:
                try:
                    self._tap.close()
                finally:
                    self._tap = None

    # Start the background segmentation thread (no-op if already running)
    def start(self):
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        print("[segrec] started")

    # Stop the worker, close monitoring, and clear buffers
    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
            self._t = None
        if self._tap:
            try:
                self._tap.close()
            finally:
                self._tap = None
        self._buf.clear()
        print("[segrec] stopped")
