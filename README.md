# Audio Classification with TFLite on Raspberry Pi

This project uses a **Raspberry Pi**, a **Sense HAT**, and a **TFLite model** to **record, monitor, and classify audio** via a serial interface.  
Predictions are displayed on the Sense HAT LED matrix, while audio recordings are stored locally.

---

## Project Structure

```text
.
├── main.py                   # Entry point: builds controller and links joystick events
├── config/
│   └── config.py              # Global configuration (sample rate, ports, paths, colors)
├── control/
│   └── controller.py          # State machine for recording, monitoring, predicting
├── audio/
│   ├── recorder.py            # Records audio and saves to .wav
│   ├── segment_recorder.py    # Continuously records audio in segments
│   └── monitor.py             # Streams incoming audio live
├── iodev/
│   └── serial_stream.py       # Reads audio bytes from serial
├── models/
│   ├── manager.py             # Loads active model bundle (manifest, labels, model, preprocessor)
│   └── bundles/               # Place model bundles here (manifest.json, labels.json, preprocess.py)
├── predict/
│   ├── tflite_predictor.py    # Wrapper around TFLite Interpreter
├── ui/
│   └── sense_ui.py            # Sense HAT LED matrix output
└── README.md                  # Project documentation
```

---

## Hardware Setup

This project is designed to run on embedded hardware. The minimal tested setup:

- **Raspberry Pi 3**  
  The main compute unit that runs the Python code and TensorFlow Lite model.

- **Digital stethoscope (USB)**  
  Connected via USB. The Pi reads the raw audio stream over a serial interface (`/dev/ttyACM0` by default, see `Config.COM_PORT`).

- **USB stick (storage)**  
  Used as external storage for recorded audio and segment files.  
  Mounted under `/media/LongPi/...` and automatically detected in `main.py:get_output_dir()`.  
  If no USB is present, data falls back to the local `data/` directory.

- **Sense HAT**  
  Provides:
  - 8×8 RGB LED matrix for status and prediction output (colors indicate states/classes).
  - Joystick used as the main control interface (Up/Down/Left/Right/Middle).

### Wiring & connections

- Plug the stethoscope into any USB port on the Pi. Confirm it shows up as `/dev/ttyACM0`.  
- Plug in a USB stick. A subdirectory `data/` will be created automatically for audio storage.  
- Attach the Sense HAT to the Pi’s GPIO header. Ensure it is firmly connected and not tilted.  
- Connect speakers or headphones to the Pi (3.5mm jack or HDMI) if you want to monitor audio output.

### Tested environment

- **Raspberry Pi OS (Bookworm)**  
- Pi 3 Model B  
- Sense HAT v1.0  
- Generic USB digital stethoscope (CDC ACM device)  
- 16 GB USB flash drive (FAT32 formatted)

---

## Installation

1. Clone the repo onto your Raspberry Pi:

```bash
git clone <repo-url>
cd <repo-dir>
```

2. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3. Install TFLite runtime (lighter than full TensorFlow, recommended for Pi):

```bash
pip install tflite-runtime
```


### System Dependencies (Raspberry Pi)

Install these system packages **before** running `pip install -r requirements.txt`:

```bash
sudo apt-get update
sudo apt-get install -y   libportaudio2 libportaudiocpp0 portaudio19-dev   libsndfile1   python3-dev
```

> `libportaudio2` and `libsndfile1` are runtime deps.  
> `portaudio19-dev` + `python3-dev` help if wheels are not available and a local build is needed.

Quick device check:

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

---

## Usage

Run the application:

```bash
python main.py
```

### Joystick Functions (Sense HAT)

- **⬆️ Up** → record audio (`Config.RECORD_SECONDS`)  
- **⬇️ Down** → start/stop prediction (model is lazy-loaded)  
- **⬅️ Left** → start/stop live monitoring  
- **➡️ Right** → start/stop continuous segment recording  
- **⏺️ Middle** → return to idle or reset after error  

### LED Status Colors
- 🟢 **Green** → idle  
- 🟡 **Yellow** → recording in progress  
- 🔴 **Red** → error  
- 🔵 **Cyan** → monitor mode  
- 🟠 **Orange** → continuous recording  
- ⚪ **White border** → prediction mode, inner color = predicted class  

---

## Model Bundles

A bundle contains:

- `manifest.json`  
  ```json
  {
    "input_shape": [1, 513, 173, 1],
    "input_dtype": "float32",
    "num_classes": 10,
    "model_sha256": "...",
    "preprocess_mode": "plugin"
  }
  ```
- `labels.json` → list of class labels  
- `model.tflite` → TensorFlow Lite model  
- `preprocess.py` → defines `build_preprocessor(reader_factory, cfg, manifest)`  

The active model is set in `models/deployment.json`:
```json
{
  "active": "bundles/my_model_v1"
}
```

---

## Configuration (`config.py`)

Key variables:
- `FS` → sample rate (default 15750 Hz)  
- `COM_PORT` → serial port (default `/dev/ttyACM0`)  
- `BAUD` → baudrate (default 921600)  
- `RECORD_SECONDS` → duration of single recordings  
- `SEGMENT_SECONDS` → segment length in continuous recording  
- `MODELS_DIR`, `DEPLOYMENT_PATH` → model/bundle paths  

Colors: `COLORS` and `CLASS_COLORS` control LED status and prediction display.

---

## Development & Maintenance

- **New models** → place as a bundle under `models/bundles/` and update `deployment.json`.  
- **New preprocessors** → implement in the bundle’s `preprocess.py`.  
- **Error handling** → errors stop running processes and set LED to 🔴.  
- **Testing** → use `recorder.py` to record samples and validate predictions.  

---

## Troubleshooting

| Problem | Possible Cause | Fix |
|---------|----------------|-----|
| `ModuleNotFoundError: tflite_runtime` | TFLite runtime not installed | `pip install tflite-runtime` |
| `Bad input shape` error | Preprocessor output shape/dtype mismatch | Adjust `preprocess.py` to match `input_shape` |
| Model not loading | Invalid `deployment.json` | Verify bundle path and manifest.json |
| No LED output | Sense HAT missing | Fallback in `sense_ui.py` prints to console |

---

## TODO / Roadmap

Below are the immediate tasks I plan to do. Each item has a short implementation note and acceptance criteria so it's easy to pick up and finish.

- [ ] **Allow live monitoring while `segment_recorder` is running (ORANGE mode)**
  - **What:** when continuous segment recording is active, allow optionally listening to the incoming audio in realtime.
  - **Where:** `segment_recorder.py` (+ small hook in `controller.py` to toggle the behaviour).
  - **How:** add an option `monitor_during_segments: bool` (config or ctor arg). If true, spawn or reuse the `Monitor` output stream in `SegmentRecorder._loop` and push raw chunks to the same queue used by `Monitor`. Alternatively call into `Monitor` instead of duplicating code.
  - **Acceptance:** when `Right` starts continuous recording and `monitor_during_segments=true`, audio should be audible on the Pi speakers with negligible extra CPU. Stopping segments also stops the monitor stream.

- [ ] **Test & validate bundle-loading workflow (ModelBundleManager)**
  - **What:** verify `load_active_bundle()` works for expected bundle layout (manifest, labels, model, preprocess plugin) and fail-safes behave cleanly.
  - **Where:** `manager.py` (class `ModelBundleManager`) and `models/deployment.json`.
  - **How:** write unit tests that:
    1. Create a temporary bundle dir containing a minimal `manifest.json`, `labels.json`, a tiny `model.tflite` (or dummy file), and a `preprocess.py` implementing `build_preprocessor`.
    2. Call `load_active_bundle()` and assert: returns True, `predictor` initialized, `get_input_fn()` returns array shaped to `manifest["input_shape"]`.
    3. Test failure cases: missing `preprocess.py`, input-shape mismatch, bad `model_sha256`.
  - **Acceptance:** automated tests pass; if a bundle is malformed the controller moves to error state and LED shows RED with a clear console error.

- [ ] **Add 'bypass filter' toggle to the listen/preprocess pipeline (stethoscope-dependent)**
  - **What:** some stethoscopes require the internal filter to be bypassed; add a toggle to enable/disable bypass at runtime.
  - **Where:** plugin preprocessor inside each bundle (`preprocess.py`) + a global control exposed via `Config` or `Controller` UI handlers.
  - **How:** design the preprocessor factory signature to accept a `bypass_filter` flag (or read it from `cfg`). Example pattern in `preprocess.py`:
    ```python
    def build_preprocessor(reader_factory, cfg, manifest):
        bypass = cfg.BYPASS_FILTER   # or read from runtime state
        def preproc():
            raw = read_bytes_from_reader()
            if not bypass:
                raw = apply_filter(raw)
            return final_array
        return preproc
    ```
    Expose a joystick combination or a small REST endpoint / config file toggle to flip `BYPASS_FILTER` at runtime (Controller can reload or pass updated flag to the bundle if you implement hot-reload).
  - **Acceptance:** with bypass on/off the preprocessor output changes accordingly, and the model either tolerates both modes or you can log/flag which mode was used for each prediction.

- [ ] **Provide a method to control playback/output volume externally**
  - **What:** allow the system to control audio output level from software (for monitor playback or recorded playback).
  - **Where:** `monitor.py` (playback path) and optionally `main.py`/`controller.py` to expose the control.
  - **How:** two pragmatic options:
    1. Use system mixer via `amixer` (ALSA). Example CLI to set volume:
       ```bash
       amixer set 'PCM' 50%    # set PCM to 50%
       ```
       Call `subprocess.run(["amixer","set","PCM","50%"])` from Python where needed.
    2. Use an ALSA Python binding (`pyalsaaudio`) to control volume programmatically.
    Add a small API in `Monitor` like `set_volume(percent: int)` that uses one of the above.
  - **Acceptance:** calling the API/CLI changes the audible volume. Provide a small CLI or joystick mapping to increase/decrease volume during monitor mode.

### Notes / priorities
- Start with **bundle tests** first — they reduce risk when you lazy-load models during runtime.
- Implement **monitor during segments** next; it's low-risk and improves UX.
- The **bypass filter** requires coordination with bundle preprocessors and is tied to the stethoscope hardware; keep it modular and config-driven.
- **Volume control** can be implemented last; `amixer` is the fastest route.

---

## License

MIT.
