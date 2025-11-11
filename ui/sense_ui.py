# Simple Sense HAT UI helper for showing states and predictions on the 8x8 LED matrix.
# Falls back to a tiny stub when the real Sense HAT library is not available
# (useful for development on laptops).

try:
    from sense_hat import SenseHat
except ImportError:
    # Minimal stub so the rest of the code can run without hardware.
    class SenseHat:  # type: ignore
        def __init__(self):
            # Provide a placeholder for the joystick attribute used elsewhere
            self.stick = type("S", (), {"direction_any": None})()

        def clear(self, color=(0, 0, 0)):
            # Print the action so you can see something in logs during development
            print(f"[sense] clear {color}")

        def set_pixel(self, x, y, color):
            # No-op in the stub
            pass


class SenseUI:
    """
    Thin wrapper around SenseHat for consistent, easy-to-read UI actions.
    - fill(name): fill the whole matrix with a named color from the palette
    - draw_pred_class(idx, palette, border_name): draw a white border and fill
      the inside with a class color based on index
    """
    def __init__(self, colors: dict):
        # Store the device handle and a mapping of color names → RGB tuples
        self.sense = SenseHat()
        self.c = colors

    def fill(self, name: str):
        """
        Fill the entire LED matrix with a named color.
        The name must exist as a key in the provided color dictionary.
        """
        self.sense.clear(self.c[name])

    def draw_pred_class(self, cls_idx: int, palette: list[tuple[int, int, int]], border_name: str = "WHITE"):
        """
        Draw a white border and fill with a class color selected by cls_idx.

        Translation of original Dutch note:
        "White border + fill color from palette based on cls_idx (wrap with modulo)."

        Behavior:
        - Clears the matrix first (safety reset)
        - Draws a 1-pixel border in the named border color (default: WHITE)
        - Picks the interior color from the given palette using cls_idx % len(palette)
        - Fills the 6x6 interior area with that color
        """
        s = self.sense

        # Clear any previous drawing to avoid leftover pixels
        s.clear()

        # Get the RGB color for the border (defaults to WHITE if not changed)
        border = self.c[border_name]

        # Draw the 8x8 border: top, bottom, left, right edges
        for i in range(8):
            s.set_pixel(i, 0, border)  # top row
            s.set_pixel(i, 7, border)  # bottom row
            s.set_pixel(0, i, border)  # left column
            s.set_pixel(7, i, border)  # right column

        # Choose an interior color based on class index (safe modulo range)
        color = palette[cls_idx % len(palette)]

        # Fill the 6x6 interior area (coordinates 1..6 inclusive)
        for x in range(1, 7):
            for y in range(1, 7):
                s.set_pixel(x, y, color)
