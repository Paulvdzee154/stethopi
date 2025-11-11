import numpy as np
import os
import time
from datetime import datetime
from scipy.io.wavfile import write
import sounddevice as sd

class AudioTap:
    """
    Plays raw int16 mono bytes directly to the selected audio device.
    Use this to monitor (hear) the microphone while recording.
    """
    def __init__(self, fs: int, device=None, blocksize=1024):
        # Create a raw output stream (mono, 16-bit) for low-latency monitoring
        self._s = sd.RawOutputStream(
            samplerate=fs, channels=1, dtype='int16',
            blocksize=blocksize, latency='low', device=device
        )
        # Start the audio stream immediately
        self._s.start()

    def write(self, raw: bytes):
        # Push raw bytes to the output device if data is present
        if raw:
            self._s.write(raw)

    def close(self):
        # Safely stop and close the output stream
        try:
            self._s.stop()
            self._s.close()
        except Exception:
            # Ignore teardown errors; we’re shutting down anyway
            pass


class Recorder:
    """
    Records audio chunks coming from a reader (as raw int16 mono bytes),
    optionally monitors them live via the selected output device,
    and saves the result to a WAV file in a timestamped path.
    """
    def __init__(self, fs: int, reader_factory, output_dir_fn, out_device=None, blocksize=1024):
        # Store configuration
        self.fs = fs
        self.reader_factory = reader_factory     # must return a context manager with read_bytes()
        self.output_dir_fn = output_dir_fn       # callable that returns the output directory path
        self.out_device = out_device             # sounddevice output device index/name for monitoring
        self.blocksize = blocksize               # audio block size to keep latency stable

    def record_seconds(self, seconds: int) -> str | None:
        """
        Record for a fixed duration.
        - Reads raw bytes from the reader.
        - Feeds the same bytes into AudioTap to monitor while recording.
        - Converts to int16 array and writes a WAV file.
        Returns the saved file path or None on failure.
        """
        buf = bytearray()        # accumulates all raw bytes for this take
        tap = None               # will hold the AudioTap for live monitoring
        start = time.monotonic() # monotonic clock to measure duration accurately

        try:
            # Open the reader (e.g., serial reader) that yields raw audio bytes
            with self.reader_factory() as reader:
                # Start live monitoring to the chosen output (e.g., AV jack headphones)
                tap = AudioTap(self.fs, device=self.out_device, blocksize=self.blocksize)

                # Keep reading until the requested number of seconds has passed
                while time.monotonic() - start < seconds:
                    raw = reader.read_bytes()  # get the next raw chunk (may be empty)
                    if raw:
                        buf.extend(raw)        # store for saving
                        tap.write(raw)         # monitor while recording
        except Exception as e:
            # Reader failures (e.g., serial disconnect) are reported and we abort
            print(f"[record] Serial error: {e}")
            return None
        finally:
            # Always close the monitoring stream
            if tap:
                tap.close()

        # Convert accumulated bytes to int16 mono samples
        n = len(buf) // 2  # 2 bytes per int16 sample
        if n == 0:
            print("[record] No data; got 0 samples")
            return None

        # Build a NumPy array without copying more than needed
        data = np.frombuffer(memoryview(buf)[: n * 2], dtype=np.int16)

        # Construct an output file path with a readable timestamp
        ts = datetime.now().strftime("T%H_%M_%S_D%d_%m_%Y")
        outdir = self.output_dir_fn()
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{ts}.wav")

        # Write the WAV file; on error, report and abort
        try:
            write(path, self.fs, data)
        except Exception as e:
            print(f"[record] Save failed: {e}")
            return None

        # Log a short summary and return the path
        print(f"[record] Saved: {path} ({n} samples, {(n/self.fs):.2f}s)")
        return path
