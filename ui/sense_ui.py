try:
    from sense_hat import SenseHat
except ImportError:
    class SenseHat:  # type: ignore
        def __init__(self):
            self.stick = type("S", (), {"direction_any": None})()
        def clear(self, color=(0,0,0)):
            print(f"[sense] clear {color}")
        def set_pixel(self, x, y, color):
            pass

class SenseUI:
    def __init__(self, colors: dict):
        self.sense = SenseHat()
        self.c = colors

    def fill(self, name: str):
        self.sense.clear(self.c[name])

    def draw_pred_class(self, cls_idx: int, palette: list[tuple[int,int,int]], border_name: str = "WHITE"):
        """Witte rand + invulkleur uit palette op basis van cls_idx (wrap met modulo)."""
        s = self.sense
        s.clear()  # reset voor zekerheid
        border = self.c[border_name]
        for i in range(8):
            s.set_pixel(i, 0, border)
            s.set_pixel(i, 7, border)
            s.set_pixel(0, i, border)
            s.set_pixel(7, i, border)

        color = palette[cls_idx % len(palette)]
        for x in range(1, 7):
            for y in range(1, 7):
                s.set_pixel(x, y, color)
