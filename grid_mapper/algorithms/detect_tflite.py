import os

from qgis.core import (
    QgsApplication,
    QgsProcessingException,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
)

from ..core import tiling
from ..core.compat import FILE_BEHAVIOR, NUM_INT
from ..core.tflite_detector import load_local_detector
from ..core.model_registry import active_model_path, default_registry_dir
from .base_detect import BaseDetectionAlgorithm, gsd_center_note, tr


class DetectSubstationsTFLite(BaseDetectionAlgorithm):
    MODEL = "MODEL"
    LABELS = "LABELS"
    THREADS = "THREADS"

    def name(self):
        return "detect_tflite"

    def displayName(self):  # noqa: N802
        return tr("Detect substations (local model: TFLite / ONNX)")

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Scans the imagery tile by tile with a model that runs on your computer and returns "
            "one polygon per detected substation.</p>"
            "<p><b>Models</b>:</p><ul>"
            "<li><b>ONNX - RF-DETR</b> (recommended): <code>.onnx</code> or the <code>gridmapper_model.zip</code> "
            "from the Grid Mapper training notebook. Segmentation models return true footprints.</li>"
            "<li>ONNX - Ultralytics YOLO exports (AGPL-3.0 licence).</li>"
            "<li><b>TFLite</b>: the thesis <code>custom_model_lite.zip</code> (SSD-MobileNet-V2-FPNLite-320 or "
            "EfficientDet) or any TF Object Detection API <code>detect.tflite</code>.</li></ul>"
            "<p>If no model file is selected, Grid Mapper uses the current <b>active model</b> from the "
            "continual-learning model registry.</p>"
            "<p><b>Requirement</b> (once, in the OSGeo4W Shell, then restart QGIS): "
            "<code>python -m pip install onnxruntime</code> for ONNX, "
            "<code>python -m pip install ai-edge-litert</code> for TFLite. The error message shows the exact "
            "command with your numpy version pinned.</p>"
        ) + gsd_center_note()

    def createInstance(self):  # noqa: N802
        return DetectSubstationsTFLite()

    def add_model_parameters(self):
        self.addParameter(QgsProcessingParameterFile(
            self.MODEL, tr("Model (.onnx, .tflite or model .zip)"),
            behavior=FILE_BEHAVIOR, fileFilter="Detection model (*.onnx *.zip *.tflite)", optional=True))
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
        settings_dir = QgsApplication.qgisSettingsDirPath()
        if not model_path:
            registry_dir = default_registry_dir(settings_dir)
            model_path = active_model_path(registry_dir)
            if not model_path:
                raise QgsProcessingException(tr(
                    "No model selected and no active model is registered. Select a model file, or run "
                    "'Register / safely promote a candidate model' first."))
            feedback.pushInfo(tr(f"Using active registered model: {model_path}"))
        cache = os.path.join(settings_dir, "grid_mapper", "models")
        try:
            self.detector, model_file = load_local_detector(model_path, labels_path, threads, cache)
        except ImportError as exc:
            raise QgsProcessingException(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise QgsProcessingException(tr(f"Could not load model {model_path}: {exc}")) from exc
        self._model_label = f"{os.path.basename(model_path)} ({os.path.basename(model_file)})"
        d = self.detector
        feedback.pushInfo(tr(
            f"Runtime: {d.backend}; model {getattr(d, 'arch', '?')}"
            f"{' (segmentation)' if getattr(d, 'segmentation', False) else ''}; input {d.width}x{d.height}; "
            f"labels: {', '.join(d.labels) or 'substation'}"))

    def model_name(self):
        return getattr(self, "_model_label", "tflite")

    def detect_tile(self, image, feedback):
        rgb = tiling.qimage_to_rgb(image, self.detector.input_size)
        w, h = image.width(), image.height()
        out = []
        for d in self.detector.detect(rgb, min_score=0.05):
            poly = d.get("polygon_norm")
            out.append({
                "xmin": d["xmin"] * w, "ymin": d["ymin"] * h,
                "xmax": d["xmax"] * w, "ymax": d["ymax"] * h,
                "score": d["score"], "label": d["label"],
                "polygon": [(x * w, y * h) for x, y in poly] if poly and len(poly) >= 4 else None,
            })
        return out
