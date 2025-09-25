import numpy as np

# Gebruik tflite_runtime op de Pi (lichter dan vol TensorFlow)
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # fallback als je lokaal test met TF geïnstalleerd
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

class TFLitePredictor:
    def __init__(self, model_path: str):
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

        self.input_shape = tuple(self.inp["shape"])       # bv (1, 513, 173, 1)
        self.input_dtype = self.inp["dtype"]
        self.output_dtype = self.out["dtype"]

    def predict_one(self, get_input_fn) -> tuple[int, np.ndarray]:
        """
        get_input_fn() moet een numpy array teruggeven met shape == input_shape.
        Retourneert (pred_class_index, probs) waarbij probs 1D softmax is.
        """
        x = get_input_fn().astype(self.input_dtype, copy=False)
        if tuple(x.shape) != tuple(self.input_shape):
            raise ValueError(f"Bad input shape: got {x.shape}, expected {self.input_shape}")

        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])  # shape (1, C) of (C,)
        y = np.squeeze(y)

        # Softmax indien nodig (sommige modellen geven logits)
        if y.ndim == 1:
            # detecteer of al softmax-achtig is
            if not np.allclose(np.sum(y), 1.0, atol=1e-3) or np.any(y < 0):
                e = np.exp(y - np.max(y))
                y = e / (np.sum(e) + 1e-12)
        else:
            raise ValueError(f"Unexpected output shape: {y.shape}")

        return int(np.argmax(y)), y
