import os, json, hashlib, importlib.util
from typing import Callable, Optional
import numpy as np
from config.config import Config
from predict.tflite_predictor import TFLitePredictor

class ModelBundleManager:
    """
    Loads and validates the *active* model bundle as defined in deployment.json.
    Responsibilities:
      - Locate the active model directory
      - Validate model integrity (SHA-256)
      - Check input shape, dtype, and number of classes
      - Load the preprocessor plugin (preprocess.py)
      - Prepare callable input function for predictions
    """

    def __init__(self, cfg: Config, reader_factory: Callable):
        # Store references for configuration and audio data source
        self.cfg = cfg
        self.reader_factory = reader_factory

        # Internal state placeholders
        self.bundle_dir: Optional[str] = None
        self.manifest: Optional[dict] = None
        self.labels: Optional[list] = None
        self.predictor: Optional[TFLitePredictor] = None
        self._preprocess_callable: Optional[Callable[[], np.ndarray]] = None

        # Shape/type info from manifest
        self.input_shape = None
        self.input_dtype = None
        self.num_classes = None

    # ---------- Filesystem utilities ----------

    def _read_json(self, path: str):
        """Read and parse a JSON file from disk."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _sha256(self, path: str) -> str:
        """Compute the SHA-256 checksum of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ---------- Public methods ----------

    def load_active_bundle(self) -> bool:
        """
        Load the currently active model bundle and validate it.
        Steps:
          1. Read deployment.json → find the 'active' path
          2. Load manifest.json, labels.json, and model.tflite
          3. Verify model checksum if present
          4. Load TFLite model using TFLitePredictor
          5. Validate input shape, dtype, and num_classes
          6. Load preprocessor plugin and test output shape/dtype
        """
        # Read the deployment configuration file
        dep = self._read_json(self.cfg.DEPLOYMENT_PATH)
        active = dep.get("active")

        # Fallback if 'active' field missing → use symlink models/current
        if not active:
            active = os.path.realpath(os.path.join(self.cfg.MODELS_DIR, "current"))

        # Build absolute path to the bundle directory
        self.bundle_dir = active if os.path.isabs(active) else os.path.join(self.cfg.MODELS_DIR, active)

        # Build all file paths we expect to find
        man_path = os.path.join(self.bundle_dir, "manifest.json")
        lab_path = os.path.join(self.bundle_dir, "labels.json")
        mdl_path = os.path.join(self.bundle_dir, "model.tflite")

        # Load manifest and label files
        self.manifest = self._read_json(man_path)
        self.labels = self._read_json(lab_path)

        # (Optional) Verify model file integrity using SHA-256
        expected_sha = self.manifest.get("model_sha256")
        if expected_sha:
            actual_sha = self._sha256(mdl_path)
            if actual_sha.lower() != expected_sha.lower():
                raise ValueError(
                    f"Model SHA mismatch: got {actual_sha}, expected {expected_sha}"
                )

        # Load the TensorFlow Lite model wrapper
        self.predictor = TFLitePredictor(mdl_path)

        # Read shape/dtype info from manifest
        self.input_shape = tuple(self.manifest["input_shape"])
        self.input_dtype = np.dtype(self.manifest.get("input_dtype", str(self.predictor.input_dtype)))

        # Check that manifest and model agree on input shape
        if tuple(self.predictor.input_shape) != tuple(self.input_shape):
            raise ValueError(
                f"Input shape mismatch: predictor {self.predictor.input_shape} vs manifest {self.input_shape}"
            )

        # Verify that labels and manifest agree on number of classes
        self.num_classes = len(self.labels)
        if "num_classes" in self.manifest:
            if int(self.manifest["num_classes"]) != int(self.num_classes):
                raise ValueError(
                    f"num_classes mismatch: labels({self.num_classes}) vs manifest({self.manifest['num_classes']})"
                )

        # Check preprocessing mode — must be 'plugin'
        mode = self.manifest.get("preprocess_mode", "plugin")
        if mode != "plugin":
            raise ValueError(f"Only 'plugin' preprocess_mode is supported; got {mode}")

        # Load the custom preprocessing function from preprocess.py
        self._load_plugin_preprocessor()

        # Run a quick test: preprocessor output must match expected shape/dtype
        test = self.get_input_fn()()
        if tuple(test.shape) != tuple(self.input_shape) or np.dtype(test.dtype) != self.input_dtype:
            raise ValueError(
                f"Preprocessor produced {test.shape}, {test.dtype}; "
                f"expected {self.input_shape}, {self.input_dtype}"
            )

        print(f"[bundle] Loaded OK: {self.bundle_dir}")
        return True

    def _load_plugin_preprocessor(self):
        """
        Dynamically import preprocess.py inside the bundle directory
        and extract its build_preprocessor(reader_factory, cfg, manifest) function.
        """
        assert self.bundle_dir is not None
        pp_path = os.path.join(self.bundle_dir, "preprocess.py")
        if not os.path.exists(pp_path):
            raise FileNotFoundError("preprocess.py not found (plugin mode)")

        # Load Python module dynamically
        spec = importlib.util.spec_from_file_location("bundle_preprocess", pp_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        # Must define build_preprocessor() returning a callable that yields input tensors
        if not hasattr(mod, "build_preprocessor"):
            raise AttributeError(
                "preprocess.py must define build_preprocessor(reader_factory, cfg, manifest) -> callable"
            )

        # Build callable that returns preprocessed numpy array per prediction
        self._preprocess_callable = mod.build_preprocessor(self.reader_factory, self.cfg, self.manifest)

    # ---------- API for predictor ----------

    def get_input_fn(self) -> Callable[[], np.ndarray]:
        """
        Return a callable that generates one preprocessed input tensor.
        This wraps the plugin preprocessor and ensures type/shape safety.
        """
        if self._preprocess_callable is None:
            raise RuntimeError("Bundle not loaded or preprocessor missing")

        def _fn():
            x = self._preprocess_callable()
            # Ensure dtype and shape exactly match what the model expects
            if x.dtype != self.input_dtype:
                x = x.astype(self.input_dtype, copy=False)
            if tuple(x.shape) != tuple(self.input_shape):
                raise ValueError(f"Preprocessor produced {x.shape}; expected {self.input_shape}")
            return x

        return _fn

    def predict_one(self):
        """
        Convenience helper: preprocess and run a single prediction.
        Returns the class index and probability distribution.
        """
        x = self.get_input_fn()()
        return self.predictor.predict_one(lambda: x)
