import os, json, hashlib, importlib.util
from typing import Callable, Optional
import numpy as np
from config.config import Config
from predict.tflite_predictor import TFLitePredictor

class ModelBundleManager:
    """
    Laadt active bundle op basis van deployment.json (active/previous).
    Validaties:
      - model sha256 (optioneel)
      - input shape/dtype vs. manifest & TFLite model
      - labels.json lengte vs. (optionele) num_classes
    Preprocessing:
      - Alleen 'plugin' modus: bundle/preprocess.py met build_preprocessor(reader_factory,cfg,manifest)->callable
      - Callable moet een numpy array (exact input_shape, dtype) teruggeven per predict
    """
    def __init__(self, cfg: Config, reader_factory: Callable):
        self.cfg = cfg
        self.reader_factory = reader_factory
        self.bundle_dir: Optional[str] = None
        self.manifest: Optional[dict] = None
        self.labels: Optional[list] = None
        self.predictor: Optional[TFLitePredictor] = None
        self._preprocess_callable: Optional[Callable[[], np.ndarray]] = None
        self.input_shape = None
        self.input_dtype = None
        self.num_classes = None

    # ---------- filesystem ----------
    def _read_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ---------- public ----------
    def load_active_bundle(self) -> bool:
        dep = self._read_json(self.cfg.DEPLOYMENT_PATH)
        active = dep.get("active")
        if not active:
            # fallback naar symlink models/current
            active = os.path.realpath(os.path.join(self.cfg.MODELS_DIR, "current"))
        self.bundle_dir = active if os.path.isabs(active) else os.path.join(self.cfg.MODELS_DIR, active)

        man_path = os.path.join(self.bundle_dir, "manifest.json")
        lab_path = os.path.join(self.bundle_dir, "labels.json")
        mdl_path = os.path.join(self.bundle_dir, "model.tflite")

        self.manifest = self._read_json(man_path)
        self.labels = self._read_json(lab_path)

        # SHA check (optioneel)
        expected_sha = self.manifest.get("model_sha256")
        if expected_sha:
            actual_sha = self._sha256(mdl_path)
            if actual_sha.lower() != expected_sha.lower():
                raise ValueError(f"Model SHA mismatch. got {actual_sha}, expected {expected_sha}")

        # TFLite laden
        self.predictor = TFLitePredictor(mdl_path)

        # Shapes / dtype
        self.input_shape = tuple(self.manifest["input_shape"])
        # dtype in manifest is string; maak er np.dtype van. fallback naar predictor.input_dtype.
        self.input_dtype = np.dtype(self.manifest.get("input_dtype", str(self.predictor.input_dtype)))

        if tuple(self.predictor.input_shape) != tuple(self.input_shape):
            raise ValueError(f"Input shape mismatch predictor {self.predictor.input_shape} vs manifest {self.input_shape}")

        # Labels / classes
        self.num_classes = len(self.labels)
        if "num_classes" in self.manifest:
            if int(self.manifest["num_classes"]) != int(self.num_classes):
                raise ValueError(f"num_classes mismatch labels({self.num_classes}) vs manifest({self.manifest['num_classes']})")

        # Preprocess-mode = plugin (verplicht)
        mode = self.manifest.get("preprocess_mode", "plugin")
        if mode != "plugin":
            raise ValueError(f"Only 'plugin' preprocess_mode is supported here; got {mode}")

        self._load_plugin_preprocessor()

        # Dry-run: moet exact shape/dtype opleveren
        test = self.get_input_fn()()
        if tuple(test.shape) != tuple(self.input_shape) or np.dtype(test.dtype) != self.input_dtype:
            raise ValueError(f"Preprocessor produced {test.shape}, {test.dtype}; expected {self.input_shape}, {self.input_dtype}")

        return True

    def _load_plugin_preprocessor(self):
        assert self.bundle_dir is not None
        pp_path = os.path.join(self.bundle_dir, "preprocess.py")
        if not os.path.exists(pp_path):
            raise FileNotFoundError("preprocess.py not found (plugin mode)")

        spec = importlib.util.spec_from_file_location("bundle_preprocess", pp_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        if not hasattr(mod, "build_preprocessor"):
            raise AttributeError("preprocess.py must define build_preprocessor(reader_factory, cfg, manifest) -> callable")

        # Geeft callable terug die per predict de input tensor produceert
        self._preprocess_callable = mod.build_preprocessor(self.reader_factory, self.cfg, self.manifest)

    # ---------- API ----------
    def get_input_fn(self) -> Callable[[], np.ndarray]:
        if self._preprocess_callable is None:
            raise RuntimeError("Bundle not loaded or preprocessor missing")
        def _fn():
            x = self._preprocess_callable()
            # veiligheid: cast/reshape
            if x.dtype != self.input_dtype:
                x = x.astype(self.input_dtype, copy=False)
            if tuple(x.shape) != tuple(self.input_shape):
                raise ValueError(f"Preprocessor produced {x.shape}; expected {self.input_shape}")
            return x
        return _fn

    def predict_one(self):
        x = self.get_input_fn()()
        return self.predictor.predict_one(lambda: x)