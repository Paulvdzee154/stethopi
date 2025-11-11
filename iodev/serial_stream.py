import time
import serial

class SerialAudioReader:
    """
    Robust reader for audio bytes over USB serial.

    - Opens the port exclusively (on Linux) to prevent double access.
    - Automatically reopens on transient errors like:
      "device reports readiness to read but returned no data".
    """
    def __init__(self, port: str, baud: int, chunk_bytes: int):
        # Store connection parameters and the per-read byte size
        self.port = port
        self.baud = baud
        self.chunk = chunk_bytes
        self.ser: serial.Serial | None = None

    def __enter__(self):
        # Allow usage as a context manager: open on enter
        self._open()
        return self

    def __exit__(self, *exc):
        # Close the port when leaving the context manager
        self._close()

    # Internal helpers

    def _open(self):
        """
        Open the serial port with sane timeouts and exclusive access.
        - exclusive=True prevents race conditions with multiple processes (POSIX only).
        - inter_byte_timeout helps stabilize fixed-size block reads.
        """
        self.ser = serial.Serial(
            self.port,
            baudrate=self.baud,
            timeout=0.2,
            inter_byte_timeout=0.2,
            exclusive=True  # POSIX-only; ignored on Windows
        )

    def _close(self):
        # Safely close the serial port if it is open
        if self.ser:
            try:
                if self.ser.is_open:
                    self.ser.close()
            except Exception:
                # Ignore teardown errors during close
                pass
            self.ser = None

    def _reopen(self):
        # Close → brief pause → open again to recover from transient errors
        self._close()
        time.sleep(0.1)
        self._open()

    # Public API

    def read_bytes(self) -> bytes:
        """
        Read a chunk of bytes from the serial port.

        On a transient SerialException (e.g., "readiness to read but returned no data"
        or "multiple access on port"), try to reopen once and return b'' so the caller
        can continue instead of failing immediately.
        """
        # Ensure the port is open before reading
        if not self.ser:
            self._open()

        try:
            # Read exactly up to self.chunk bytes (may return fewer)
            return self.ser.read(self.chunk)

        except serial.SerialException as e:
            # Known pyserial case:
            # "device reports readiness to read but returned no data
            #  (device disconnected or multiple access on port?)"
            msg = str(e).lower()
            if (
                "readiness to read" in msg
                or "multiple access" in msg
                or "returned no data" in msg
            ):
                # Attempt a clean recovery; if it fails, we still return empty bytes
                try:
                    self._reopen()
                except Exception:
                    # If reopen fails, let the caller keep looping until time runs out
                    pass
                return b""

            # For other serial errors, bubble the exception up
            raise
