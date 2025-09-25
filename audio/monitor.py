import queue, threading, numpy as np, sounddevice as sd

class Monitor:
    def __init__(self, fs: int, reader_factory, blocksize: int, on_error=None):
        self.fs, self.reader_factory = fs, reader_factory
        self.blocksize = blocksize
        self.on_error = on_error
        self._stop = threading.Event()
        self._t_reader = None
        self._q = queue.Queue(maxsize=64)
        self._buf = bytearray()
        self._stream = None

    def _audio_cb(self, outdata, frames, time_info, status):
        try:
            bytes_per_frame = 2
            need = frames * bytes_per_frame
            while len(self._buf) < need and not self._q.empty():
                self._buf.extend(self._q.get_nowait())
            chunk = bytes(need) if len(self._buf) < need else bytes(self._buf[:need])
            if len(self._buf) >= need: del self._buf[:need]
            outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1,1)
        except Exception as e:
            if self.on_error: self.on_error(e)
            raise

    def start(self):
        try:
            self._stop.clear()
            self._t_reader = threading.Thread(target=self._reader_loop, daemon=True)
            self._t_reader.start()
            self._stream = sd.OutputStream(samplerate=self.fs, channels=1, dtype='int16',
                                           callback=self._audio_cb, blocksize=self.blocksize, latency='low')
            self._stream.start()
            print("[monitor] started")
        except Exception as e:
            if self.on_error: self.on_error(e)
            self.stop()

    def _reader_loop(self):
        try:
            with self.reader_factory() as reader:
                while not self._stop.is_set():
                    raw = reader.read_bytes()
                    if raw:
                        try:
                            self._q.put(raw, timeout=0.1)
                        except queue.Full:
                            try: self._q.get_nowait()
                            except queue.Empty: pass
                            try: self._q.put_nowait(raw)
                            except queue.Full: pass
        except Exception as e:
            if self.on_error: self.on_error(e)
            self._stop.set()

    def stop(self):
        self._stop.set()
        if self._stream:
            try: self._stream.stop(); self._stream.close()
            except Exception: pass
        if self._t_reader:
            self._t_reader.join(timeout=1.0)
        print("[monitor] stopped")
