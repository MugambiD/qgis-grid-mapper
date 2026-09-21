import os

from qgis.core import (
    QgsApplication,
    QgsProcessingException,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
)

from ..core import tiling
from ..core.compat import FILE_BEHAVIOR, NUM_INT
from ..core.tflite_detector import TFLiteDetector, read_labels, resolve_model
from .base_detect import BaseDetectionAlgorithm, gsd_center_note, tr


class DetectSubstationsTFLite(BaseDetectionAlgorithm):
    MODEL = "MODEL"
    LABELS = "LABELS"
    THREADS = "THREADS"

    def name(self):
        return "detect_tflite"

    def displayName(self):  # noqa: N802
        return tr("Detect substations (TFLite model)")

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Scans the imagery tile by tile with a TensorFlow Lite object-detection model and "
            "returns one polygon per detected substation.</p>"
            "<p><b>Model</b>: the <code>custom_model_lite.zip</code> exported by the thesis Colab "
            "notebooks (SSD-MobileNet-V2-FPNLite-320 - best open-source model, mAP 51 %; or "
            "EfficientDet), or any <code>detect.tflite</code> + <code>labelmap.txt</code> trained "
            "with the TensorFlow Object Detection API.</p>"
            "<p><b>Requirement</b>: a TFLite runtime in QGIS' Python - in the OSGeo4W Shell run "
            "<code>python -m pip install ai-edge-litert</code> and restart QGIS.</p>"
        ) + gsd_center_note()

    def createInstance(self):  # noqa: N802
        return DetectSubstationsTFLite()

    def add_model_parameters(self):
        self.addParameter(QgsProcessingParameterFile(
            self.MODEL, tr("Model (custom_model_lite.zip, detect.tflite or folder)"),
            behavior=FILE_BEHAVIOR, fileFilter="TFLite model (*.zip *.tflite)"))
        self.addParameter(QgsProcessingParameterFile(
            self.LABELS, tr("Label map (optional, labelmap.txt - read from the zip if omitted)"),
            behavior=FILE_BEHAVIOR, fileFilter="Label map (*.txt *.pbtxt)",
            optional=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.THREADS, tr("CPU threads"), NUM_INT,
            max(1, min(8, (os.cpu_count() or 2))), minValue=1, maxValue=64))

    def init_detector(self, parameters, context, feedback):
        model_path = self.parameterAsFile(parameters, self.MODEL, context)
        labels_path = self.parameterAsFile(parameters, self.LABELS, context)
        threads = self.parameterAsInt(parameters, self.THREADS, context)
        cache = os.path.join(QgsApplication.qgisSettingsDirPath(), "grid_mapper", "models")
        try:
            tflite_path, found_labels = resolve_model(model_path, cache)
        except Exception as exc:  # noqa: BLE001
            raise QgsProcessingException(str(exc)) from exc
        labels = read_labels(labels_path or found_labels) or ["substation"]
        try:
            self.detector = TFLiteDetector(tflite_path, labels, threads)
        except ImportError as exc:
            raise QgsProcessingException(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise QgsProcessingException(tr(f"Could not load model {tflite_path}: {exc}")) from exc
        self._model_label = f"{os.path.basename(model_path)} ({os.path.basename(tflite_path)})"
        feedback.pushInfo(tr(
            f"TFLite runtime: {self.detector.backend}; input {self.detector.width}x{self.detector.height} "
            f"{self.detector.input_dtype.__name__}; labels: {', '.join(labels)}"))

    def model_name(self):
        return getattr(self, "_model_label", "tflite")

    def detect_tile(self, image, feedback):
        rgb = tiling.qimage_to_rgb(image, self.detector.input_size)
        w, h = image.width(), image.height()
        out = []
        for d in self.detector.detect(rgb, min_score=0.01):
            out.append({
                "xmin": d["xmin"] * w, "ymin": d["ymin"] * h,
                "xmax": d["xmax"] * w, "ymax": d["ymax"] * h,
                "score": d["score"], "label": d["label"], "polygon": None,
            })
        return out
