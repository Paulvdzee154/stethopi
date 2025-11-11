import numpy as np

# Prefer the lightweight TFLite runtime on the Raspberry Pi.
# If it's not available (e.g., running on a dev machine), fall back to TensorFlow's TFLite.
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore


class TFLitePredictor:
    """
    Minimal wrapper around a TensorFlow Lite model for single-step predictions.
    - Loads the model, allocates tensors, and caches input/output details.
    - Provides predict_one(get_input_fn) that returns (class_index, probabilities).
    """
    def __init__(self, model_path: str):
        # Create the interpreter for the given TFLite model and allocate buffers
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()

        # Cache the first (and usually only) input/output tensor details
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

        # Expose input/output metadata for validation elsewhere
        self.input_shape = tuple(self.inp["shape"])   # e.g., (1, 513, 173, 1)
        self.input_dtype = self.inp["dtype"]
        self.output_dtype = self.out["dtype"]

    def predict_one(self, get_input_fn) -> tuple[int, np.ndarray]:
        """
        Run a single forward pass.
        - get_input_fn() must return a NumPy array with shape == self.input_shape.
        - Returns (pred_class_index, probs), where probs is a 1D softmax vector.
        """
        # Retrieve and ensure the input is the exact dtype the model expects
        x = get_input_fn().astype(self.input_dtype, copy=False)

        # Strict shape check for safety
        if tuple(x.shape) != tuple(self.input_shape):
            raise ValueError(f"Bad input shape: got {x.shape}, expected {self.input_shape}")

        # Feed the input tensor and invoke the model
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()

        # Read the output tensor; shape is typically (1, C) or (C,)
        y = self.interp.get_tensor(self.out["index"])
        y = np.squeeze(y)  # ensure 1D if it was (1, C)

        # If the model outputs logits (not normalized), apply softmax
        if y.ndim == 1:
            # Heuristic: if sum is not ~1.0 or any value is negative, treat as logits
            if not np.allclose(np.sum(y), 1.0, atol=1e-3) or np.any(y < 0):
                e = np.exp(y - np.max(y))
                y = e / (np.sum(e) + 1e-12)
        else:
            raise ValueError(f"Unexpected output shape: {y.shape}")

        # Return the argmax class index and the probability vector
        return int(np.argmax(y)), y
