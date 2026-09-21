"""TensorFlow Lite object-detection wrapper.

Supports the models trained in the thesis Colab notebooks
(TF2 Object Detection API -> export_tflite_graph_tf2.py -> TFLiteConverter),
i.e. SSD-MobileNet-V2-FPNLite-320 and EfficientDet, distributed as
``custom_model_lite.zip`` containing ``detect.tflite`` and ``labelmap.txt``.
Also works with TF1-style and quantised (uint8 / int8) SSD exports.
"""
import hashlib
import importlib
import os
import re
import zipfile

import numpy as np


def install_help():
    pin = f' "numpy=={np.__version__}"'  # keep QGIS' numpy, upgrading it can break QGIS
    return (
        "No TensorFlow Lite runtime found in QGIS' Python.\n"
        "Install one (only once), then restart QGIS:\n"
        "  Windows: close QGIS, open 'OSGeo4W Shell' from the Start menu and run:\n"
        f"      python -m pip install ai-edge-litert{pin}\n"
        "  macOS / Linux: run with the Python that QGIS uses:\n"
        f"      python3 -m pip install --user ai-edge-litert{pin}\n"
        "Alternatives: 'tflite-runtime' or the full 'tensorflow' package."
    )


def load_interpreter_class():
    """Return (Interpreter class, backend name)."""
    for mod_name in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            mod = importlib.import_module(mod_name)
            return mod.Interpreter, mod_name.split(".")[0]
        except Exception:  # noqa: BLE001 - try the next backend
            continue
    try:
        import tensorflow as tf  # noqa: WPS433
        return tf.lite.Interpreter, "tensorflow"
    except Exception as exc:  # noqa: BLE001
        raise ImportError(install_help()) from exc


def resolve_model(path, cache_dir):
    """Accept a .tflite file, a folder or a .zip. Return (tflite_path, labelmap_path|None)."""
    if not path:
        raise FileNotFoundError("No model file given")
    path = os.path.abspath(path)
    if os.path.isdir(path):
        folder = path
    elif path.lower().endswith(".zip"):
        with open(path, "rb") as fh:
            digest = hashlib.md5(fh.read()).hexdigest()[:12]  # noqa: S324 - cache key only
        folder = os.path.join(cache_dir, digest)
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    if member.lower().endswith((".tflite", ".txt", ".pbtxt")) and ".." not in member:
                        zf.extract(member, folder)
    elif path.lower().endswith(".tflite"):
        labels = _find_file(os.path.dirname(path), ("labelmap.txt", "labels.txt"))
        return path, labels
    else:
        raise ValueError(f"Unsupported model file: {path} (use .zip, .tflite or a folder)")

    tflites = []
    for root, _dirs, files in os.walk(folder):
        tflites += [os.path.join(root, f) for f in files if f.lower().endswith(".tflite")]
    if not tflites:
        raise FileNotFoundError(f"No .tflite model found in {path}")
    # prefer detect.tflite (Colab export name), else the largest file
    tflites.sort(key=lambda p: (os.path.basename(p).lower() != "detect.tflite", -os.path.getsize(p)))
    model = tflites[0]
    labels = _find_file(os.path.dirname(model), ("labelmap.txt", "labels.txt")) or _find_file(
        folder, ("labelmap.txt", "labels.txt"), recursive=True)
    return model, labels


def _find_file(folder, names, recursive=False):
    if not folder or not os.path.isdir(folder):
        return None
    walker = os.walk(folder) if recursive else [(folder, None, os.listdir(folder))]
    for root, _d, files in walker:
        lower = {f.lower(): f for f in files}
        for n in names:
            if n in lower:
                return os.path.join(root, lower[n])
    return None


def read_labels(path):
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    if "item" in text and "name" in text:  # .pbtxt label map
        names = re.findall(r"(?:display_name|name)\s*:\s*['\"]([^'\"]+)['\"]", text)
        return list(dict.fromkeys(names))
    labels = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if labels and labels[0] == "???":  # TF1 COCO style background entry
        labels = labels[1:]
    return labels


def _suffix(name):
    m = re.search(r":(\d+)$", name)
    return int(m.group(1)) if m else 0


class TFLiteDetector:
    def __init__(self, model_path, labels=None, num_threads=4):
        interpreter_cls, self.backend = load_interpreter_class()
        try:
            self.interpreter = interpreter_cls(model_path=model_path, num_threads=int(num_threads))
        except TypeError:
            self.interpreter = interpreter_cls(model_path=model_path)
        self.interpreter.allocate_tensors()
        inp = self.interpreter.get_input_details()[0]
        self.input_index = inp["index"]
        self.height, self.width = int(inp["shape"][1]), int(inp["shape"][2])
        self.input_dtype = inp["dtype"]
        self.input_quant = inp.get("quantization", (0.0, 0))
        self.outputs = self.interpreter.get_output_details()
        self.labels = labels or []
        if len(self.outputs) < 3:
            raise ValueError(
                "This .tflite model does not have detection outputs (boxes / classes / scores). "
                "Export it with export_tflite_graph_tf2.py as in the training notebook.")

    @property
    def input_size(self):
        return self.width, self.height

    def _prepare(self, rgb):
        x = rgb[np.newaxis, ...]
        if self.input_dtype == np.float32:
            return ((x.astype(np.float32) - 127.5) / 127.5).astype(np.float32)
        if self.input_dtype == np.int8:
            scale, zero = self.input_quant
            if scale:
                return np.clip(np.round(x / 255.0 / scale + zero), -128, 127).astype(np.int8)
            return (x.astype(np.int16) - 128).astype(np.int8)
        return x.astype(np.uint8)

    def _read(self, detail):
        val = self.interpreter.get_tensor(detail["index"])
        scale, zero = detail.get("quantization", (0.0, 0))
        if val.dtype != np.float32 and scale:
            val = (val.astype(np.float32) - zero) * scale
        return np.asarray(val, dtype=np.float32)

    def _split_outputs(self):
        tensors = [(d["name"], self._read(d)) for d in self.outputs]
        boxes = num = None
        two_d = []
        for name, t in tensors:
            if t.ndim == 3 and t.shape[-1] == 4 and boxes is None:
                boxes = t[0]
            elif t.size == 1 and num is None:
                num = int(round(float(t.reshape(-1)[0])))
            elif t.ndim == 2:
                two_d.append((name, t[0]))
        if boxes is None or len(two_d) < 2:
            raise ValueError("Could not identify boxes / classes / scores in the model outputs")
        (n_a, a), b = two_d[0], two_d[1][1]
        a_int = np.allclose(a, np.round(a))
        b_int = np.allclose(b, np.round(b))
        if a_int and not b_int:
            classes, scores = a, b
        elif b_int and not a_int:
            classes, scores = b, a
        else:
            # ambiguous values: fall back to known export layouts
            by_suffix = {_suffix(n): t for n, t in two_d}
            if "StatefulPartitionedCall" in n_a:  # TF2 export: :0 scores, :3 classes
                scores, classes = by_suffix.get(0, a), by_suffix.get(3, b)
            else:  # TF1 TFLite_Detection_PostProcess: :1 classes, :2 scores
                classes, scores = by_suffix.get(1, a), by_suffix.get(2, b)
        n = len(scores) if num is None else min(num, len(scores))
        return boxes[:n], classes[:n], scores[:n]

    def detect(self, rgb, min_score=0.5):
        """rgb: uint8 (H, W, 3) already resized to the model input size.

        Returns list of dicts with normalised ymin, xmin, ymax, xmax, score, label.
        """
        self.interpreter.set_tensor(self.input_index, self._prepare(rgb))
        self.interpreter.invoke()
        boxes, classes, scores = self._split_outputs()
        out = []
        for box, cls, score in zip(boxes, classes, scores):
            if score < min_score:
                continue
            ymin, xmin, ymax, xmax = [float(np.clip(v, 0.0, 1.0)) for v in box]
            if xmax <= xmin or ymax <= ymin:
                continue
            ci = int(round(float(cls)))
            label = self.labels[ci] if 0 <= ci < len(self.labels) else f"class_{ci}"
            out.append({"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax,
                        "score": float(score), "label": label})
        return out
