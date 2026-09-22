"""Minimal Roboflow hosted-inference client (uses QGIS network settings / proxy)."""
import base64
import json
import os
import urllib.parse

import numpy as np

DEFAULT_URLS = {
    "detect": "https://detect.roboflow.com",
    "segment": "https://segment.roboflow.com",
}


def _post(url, body, content_type="application/x-www-form-urlencoded", timeout_ms=60000):
    try:
        from qgis.core import QgsBlockingNetworkRequest
        from qgis.PyQt.QtCore import QByteArray, QUrl
        from qgis.PyQt.QtNetwork import QNetworkRequest
    except ImportError as exc:  # pragma: no cover - hosted inference runs inside QGIS
        raise RuntimeError("QGIS network APIs are unavailable") from exc

    req = QNetworkRequest(QUrl(url))
    try:  # Qt6 / QGIS 4
        content_type_header = QNetworkRequest.KnownHeaders.ContentTypeHeader
    except AttributeError:  # Qt5 / QGIS 3
        content_type_header = QNetworkRequest.ContentTypeHeader
    req.setHeader(content_type_header, content_type)
    try:
        req.setTransferTimeout(timeout_ms)
    except AttributeError:
        pass
    blocking = QgsBlockingNetworkRequest()
    err = blocking.post(req, QByteArray(body))
    reply = blocking.reply()
    try:  # Qt6 / QGIS 4
        status_attr = QNetworkRequest.Attribute.HttpStatusCodeAttribute
    except AttributeError:  # Qt5 / QGIS 3
        status_attr = QNetworkRequest.HttpStatusCodeAttribute
    status = reply.attribute(status_attr)
    content = bytes(reply.content())
    if not content:
        raise RuntimeError(blocking.errorMessage() or f"No response (error code {err})")
    try:
        data = json.loads(content.decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"Roboflow returned HTTP {status}: {content[:300]!r}") from exc
    http_error = bool(status) and int(status) >= 400
    api_error = isinstance(data, dict) and "error" in data and "predictions" not in data
    if http_error or api_error:
        msg = (data.get("message") or data.get("error")) if isinstance(data, dict) else data
        raise RuntimeError(f"Roboflow error (HTTP {status}): {msg}")
    return data


class RoboflowClient:
    def __init__(self, model_id, version, api_key, task="detect", base_url="",
                 confidence=0.4, overlap=0.3):
        self.model_id = model_id.strip().strip("/")
        self.version = str(version).strip()
        self.api_key = (api_key or os.environ.get("ROBOFLOW_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("A Roboflow API key is required (parameter or ROBOFLOW_API_KEY env var)")
        self.task = task
        self.base = (base_url or DEFAULT_URLS["segment" if task == "semantic" else "detect"]).rstrip("/")
        self.confidence = confidence
        self.overlap = overlap

    def url(self):
        q = {"api_key": self.api_key, "format": "json"}
        if self.task != "semantic":
            q.update({"confidence": int(round(self.confidence * 100)),
                      "overlap": int(round(self.overlap * 100))})
        return f"{self.base}/{self.model_id}/{self.version}?{urllib.parse.urlencode(q)}"

    def infer(self, jpeg_bytes):
        return _post(self.url(), base64.b64encode(jpeg_bytes))

    # ---- response parsing ------------------------------------------------
    @staticmethod
    def parse_detections(resp, img_w, img_h):
        """Object detection / instance segmentation -> pixel-space dicts."""
        info = resp.get("image") or {}
        sx = img_w / float(info.get("width") or img_w)
        sy = img_h / float(info.get("height") or img_h)
        out = []
        for p in resp.get("predictions", []):
            x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
            poly = None
            if p.get("points"):
                poly = [(float(pt["x"]) * sx, float(pt["y"]) * sy) for pt in p["points"]]
                if len(poly) < 3:
                    poly = None
            out.append({
                "xmin": (x - w / 2) * sx, "ymin": (y - h / 2) * sy,
                "xmax": (x + w / 2) * sx, "ymax": (y + h / 2) * sy,
                "score": float(p.get("confidence", 0.0)),
                "label": str(p.get("class", "substation")),
                "polygon": poly,
            })
        return out

    @staticmethod
    def decode_mask(resp):
        """Semantic segmentation -> (class-index numpy mask, {index: name})."""
        b64 = resp.get("segmentation_mask")
        if not b64:
            raise RuntimeError("No 'segmentation_mask' in Roboflow response - is this a semantic "
                               "segmentation model?")
        from qgis.PyQt.QtGui import QImage
        img = QImage.fromData(base64.b64decode(b64))
        img = img.convertToFormat(QImage.Format.Format_Grayscale8)
        w, h, bpl = img.width(), img.height(), img.bytesPerLine()
        ptr = img.constBits()
        ptr.setsize(h * bpl)
        mask = np.frombuffer(ptr, dtype=np.uint8).reshape(h, bpl)[:, :w].copy()
        class_map = {int(k): v for k, v in (resp.get("class_map") or {}).items()}
        return mask, class_map
