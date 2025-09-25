import serial

class SerialAudioReader:
    def __init__(self, port: str, baud: int, chunk_bytes: int):
        self.port, self.baud, self.chunk = port, baud, chunk_bytes
        self.ser = None

    def __enter__(self):
        self.ser = serial.Serial(self.port, baudrate=self.baud, timeout=0.2)
        return self

    def __exit__(self, *exc):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def read_bytes(self) -> bytes:
        return self.ser.read(self.chunk)