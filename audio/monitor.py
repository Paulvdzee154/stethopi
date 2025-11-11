import queue, threading, numpy as np, sounddevice as sd

class Monitor:
    """
    Live monitoring: reads raw int16 mono bytes from a reader and plays them out.
    - out_device: explicitly choose your AV jack device (e.g., index 2 on your Pi)
    - blocksize: keep this equal to your READ_CHUNK / AUDIO_BLOCK_SAMPLES
    """
    def __init__(self, fs: int, reader_factory, blocksize: int, on_error=None, out_device=None):
        # Save constructor parameters for later use
        self.fs, self.reader_factory = fs, reader_factory
        self.blocksize = blocksize
        self.on_error = on_error
        self.out_device = out_device

        # Thread control and shared state
        self._stop = threading.Event()   # used to stop all threads
        self._t_reader = None            # background thread that pulls bytes from the reader
        self._t_pump = None              # background thread that moves bytes into the play buffer

        # Producer/consumer data structures
        self._q = queue.Queue(maxsize=64)   # reader_thread puts raw byte chunks here
        self._buf = bytearray()             # pump_thread appends here; audio callback consumes from here

        # Will hold the sounddevice output stream
        self._stream = None

    # Audio callback: sounddevice calls this regularly to fill the output buffer
    def _audio_cb(self, outdata, frames, time_info, status):
        try:
            # We need 2 bytes per mono int16 frame
            need = frames * 2
            # If we don't have enough data, output silence
            if len(self._buf) < need:
                outdata[:] = 0
                return
            # Take exactly the bytes we need and remove them from the buffer
            chunk = self._buf[:need]
            del self._buf[:need]
            # OutputStream with dtype='int16' expects int16 samples in outdata[:, 0]
            outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16)
        except Exception:
            # Fail-safe: output silence if anything goes wrong in the callback
            outdata[:] = 0

    # Background thread: continuously read raw bytes from the reader and queue them
    def _reader_thread(self):
        try:
            # reader_factory must return a context manager that yields an object with read_bytes()
            with self.reader_factory() as reader:
                while not self._stop.is_set():
                    raw = reader.read_bytes()
                    if raw:
                        try:
                            self._q.put_nowait(raw)  # do not block; drop if queue is full
                        except queue.Full:
                            pass
        except Exception as e:
            # Report errors to the optional handler and stop everything
            if self.on_error:
                self.on_error(e)
            self._stop.set()

    # Background thread: move queued chunks into the playback buffer that the audio callback reads
    def _pump_thread(self):
        try:
            while not self._stop.is_set():
                try:
                    raw = self._q.get(timeout=0.1)  # wait briefly for data
                except queue.Empty:
                    continue
                if raw:
                    self._buf.extend(raw)  # append to the playback buffer
        except Exception as e:
            # Report errors and stop
            if self.on_error:
                self.on_error(e)
            self._stop.set()

    # Start live monitoring: spin up threads and open the audio output stream
    def start(self):
        try:
            self._stop.clear()

            # Start the producer (reader) and the pump threads
            self._t_reader = threading.Thread(target=self._reader_thread, daemon=True)
            self._t_reader.start()
            self._t_pump = threading.Thread(target=self._pump_thread, daemon=True)
            self._t_pump.start()

            # Open audio output on the explicit device (e.g., AV jack)
            self._stream = sd.OutputStream(
                samplerate=self.fs,
                channels=1,
                dtype='int16',
                callback=self._audio_cb,
                blocksize=self.blocksize,
                latency='low',
                device=self.out_device  # IMPORTANT: ensures playback goes to the chosen device
            )
            self._stream.start()
            print(f"[monitor] started (device={self.out_device})")
        except Exception as e:
            print(f"[monitor] start failed: {e}")
            if self.on_error:
                self.on_error(e)

    # Stop live monitoring: stop audio, join threads, and clear buffers
    def stop(self):
        self._stop.set()

        # Stop and close the audio stream if it exists
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Join background threads (give them a short timeout)
        if self._t_reader:
            self._t_reader.join(timeout=1.0)
            self._t_reader = None
        if self._t_pump:
            self._t_pump.join(timeout=1.0)
            self._t_pump = None

        # Clear playback buffer for a clean next start
        self._buf.clear()
        print("[monitor] stopped")
