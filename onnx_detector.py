"""ONNX Runtime object detection / instance segmentation wrapper.

Supported exports
-----------------
* **RF-DETR** (Roboflow, Apache 2.0) - ``model.export()``: input ``[1,3,R,R]``
  ImageNet-normalised RGB; outputs ``dets`` (1,Q,4 normalised cx,cy,w,h),
  ``labels`` (1,Q,C+1 logits, background slot last) and, for segmentation
  models, ``masks`` (1,Q,h,w logits).
* **YOLO** (Ultralytics ``format="onnx"``, no NMS): input ``[1,3,H,W]`` RGB 0-1;
  output ``(1, 4+C, N)`` with pixel cx,cy,w,h and class scores. Note that
  Ultralytics models are AGPL-3.0 licensed.

Class names come from, in order: a ``model.json`` / ``<model>.json`` sidecar
(``{"class_names": {...}}`` written by the Grid Mapper training notebook), the
ONNX metadata (``rfdetr_notes`` / ``names``), or default to "substation".
"""
import ast
import json
import os

import numpy as np

from .postprocess import nms


def install_help():
    pin = f' "numpy=={np.__version__}"'
    return (
        "ONNX Runtime is not installed in QGIS' Python.\n"
        "Install it once, then restart QGIS:\n"
        "  Windows: close QGIS, open 'OSGeo4W Shell' from the Start menu and run:\n"
        f"      python -m pip install onnxruntime{pin}\n"
        "  macOS / Linux: with the Python that QGIS uses:\n"
        f"      python3 -m pip install --user onnxruntime{pin}"
    )


def _load_ort():
    try:
        import onnxruntime as ort
        return ort
    except Exception as exc:  # noqa: BLE001
        raise ImportError(install_help()) from exc


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _read_sidecar(model_path):
    stem = os.path.splitext(model_path)[0]
    for cand in (stem + ".json", os.path.join(os.path.dirname(model_path), "model.json")):
        if os.path.isfile(cand):
            try:
                with open(cand, encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                pass
    return {}


def _names_from(obj):
    """Normalise class names given as list or {id: name} -> {int id: name}."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(obj, (list, tuple)):
        return {i: str(v) for i, v in enumerate(obj)}
    return {}


class OnnxDetector:
    def __init__(self, model_path, labels=None, num_threads=4):
        ort = _load_ort()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(num_threads))
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.backend = f"onnxruntime {ort.__version__}"
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = [d if isinstance(d, int) else None for d in inp.shape]
        if len(shape) != 4 or shape[1] not in (1, 3):
            raise ValueError(f"Unsupported ONNX input shape {inp.shape} (expected [1,3,H,W])")
        self.height = shape[2] or 640
        self.width = shape[3] or 640
        self.input_dtype = np.float32
        self.out_names = [o.name for o in self.session.get_outputs()]
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        side = _read_sidecar(model_path)

        if "dets" in self.out_names and "labels" in self.out_names:
            self.arch = "rfdetr"
        elif len(self.out_names) == 1 and len(self.session.get_outputs()[0].shape) == 3:
            self.arch = "yolo"
        else:
            self.arch = side.get("arch", "")
            if self.arch not in ("rfdetr", "yolo"):
                raise ValueError(f"Unrecognised ONNX outputs {self.out_names}. Supported: RF-DETR "
                                 "(dets/labels[/masks]) or Ultralytics YOLO exports.")
        self.segmentation = "masks" in self.out_names

        names = _names_from(side.get("class_names"))
        if not names and meta.get("rfdetr_notes"):
            try:
                names = _names_from(json.loads(meta["rfdetr_notes"]).get("class_names"))
            except (ValueError, AttributeError):
                pass
        if not names and meta.get("names"):  # Ultralytics stores a python dict literal
            try:
                names = _names_from(ast.literal_eval(meta["names"]))
            except (ValueError, SyntaxError):
                pass
        if not names and labels:
            names = _names_from(list(labels))
        self.class_names = names
        self.labels = [names[k] for k in sorted(names)] or ["substation"]
        self.background_slot = side.get("background_class_id", -1)

    @property
    def input_size(self):
        return self.width, self.height

    def _label(self, cid):
        if cid in self.class_names:
            return self.class_names[cid]
        if len(self.class_names) == 1:
            return next(iter(self.class_names.values()))
        return "substation" if not self.class_names else f"class_{cid}"

    def _prepare(self, rgb):
        x = rgb.astype(np.float32) / 255.0
        if self.arch == "rfdetr":
            x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)

    def detect(self, rgb, min_score=0.3):
        """rgb: uint8 (H, W, 3) resized to input_size.

        Returns dicts with normalised ymin/xmin/ymax/xmax, score, label and, for
        segmentation models, ``polygon_norm`` (list of normalised x, y).
        """
        outs = self.session.run(None, {self.input_name: self._prepare(rgb)})
        res = dict(zip(self.out_names, outs))
        return self._rfdetr(res, min_score) if self.arch == "rfdetr" else self._yolo(outs[0], min_score)

    def _rfdetr(self, res, min_score):
        boxes = res["dets"][0]
        logits = res["labels"][0]
        scores = _sigmoid(logits)
        n_slots = scores.shape[1]
        class_ids = np.arange(n_slots)
        if self.background_slot is not None and n_slots > 1:
            keep = class_ids != (self.background_slot % n_slots)
            scores, class_ids = scores[:, keep], class_ids[keep]
        q_idx, c_idx = np.nonzero(scores >= min_score)
        masks = res.get("masks")
        out = []
        for q, c in zip(q_idx, c_idx):
            cx, cy, bw, bh = [float(v) for v in boxes[q]]
            d = {"xmin": max(0.0, cx - bw / 2), "ymin": max(0.0, cy - bh / 2),
                 "xmax": min(1.0, cx + bw / 2), "ymax": min(1.0, cy + bh / 2),
                 "score": float(scores[q, c]), "label": self._label(int(class_ids[c]))}
            if d["xmax"] <= d["xmin"] or d["ymax"] <= d["ymin"]:
                continue
            if masks is not None:
                d["mask"] = masks[0][q]
            out.append(d)
        # queries can duplicate each other: light class-agnostic NMS
        if len(out) > 1:
            keep = nms([[d["xmin"], d["ymin"], d["xmax"], d["ymax"]] for d in out],
                       [d["score"] for d in out], 0.7)
            out = [out[i] for i in keep]
        for d in out:
            m = d.pop("mask", None)
            if m is not None:
                d["polygon_norm"] = self._mask_polygon(m)
        return out

    @staticmethod
    def _mask_polygon(mask_logits):
        from .masks import largest_ring, mask_to_polygons
        binary = (mask_logits > 0).astype(np.uint8)
        h, w = binary.shape
        rings = mask_to_polygons(binary, [1], 1.0 / w, 1.0 / h, min_pixels=4)
        return largest_ring(rings)

    def _yolo(self, pred, min_score):
        p = pred[0]
        if p.shape[0] > p.shape[1]:  # (N, 4+C) -> (4+C, N)
            p = p.T
        box, cls = p[:4], p[4:]
        if cls.shape[0] == 0:
            return []
        cid = cls.argmax(axis=0)
        score = cls.max(axis=0)
        sel = np.nonzero(score >= min_score)[0]
        if sel.size == 0:
            return []
        cx, cy, bw, bh = box[:, sel]
        x1, y1 = (cx - bw / 2) / self.width, (cy - bh / 2) / self.height
        x2, y2 = (cx + bw / 2) / self.width, (cy + bh / 2) / self.height
        rects = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(rects, score[sel], 0.5)
        out = []
        for k in keep:
            a, b, c, d = [float(np.clip(v, 0, 1)) for v in rects[k]]
            if c > a and d > b:
                out.append({"xmin": a, "ymin": b, "xmax": c, "ymax": d,
                            "score": float(score[sel][k]), "label": self._label(int(cid[sel][k]))})
        return out
