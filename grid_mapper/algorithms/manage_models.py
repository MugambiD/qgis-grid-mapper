"""Register candidate local models and promote only when validation gates pass."""
from qgis.core import (
    QgsApplication,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputNumber,
    QgsProcessingOutputString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from ..core.compat import FILE_BEHAVIOR, NUM_DOUBLE
from ..core.model_registry import default_registry_dir, register_candidate
from .base_detect import ICON, tr


class RegisterCandidateModel(QgsProcessingAlgorithm):
    CANDIDATE = "CANDIDATE"
    MODEL_ID = "MODEL_ID"
    MIN_GAIN = "MIN_GAIN"
    MAX_GEO_DROP = "MAX_GEO_DROP"
    PROMOTE = "PROMOTE"

    def name(self):
        return "register_candidate_model"

    def displayName(self):  # noqa: N802
        return tr("Register / safely promote a candidate model")

    def group(self):
        return tr("Continual learning")

    def groupId(self):  # noqa: N802
        return "continual_learning"

    def icon(self):
        return QIcon(ICON)

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Copies a trained local model into Grid Mapper's versioned model registry. The model "
            "metadata exported by the training notebook must contain validation metrics. A candidate "
            "is activated only if the overall metric improves by the configured amount and no shared "
            "geography validation group regresses beyond the allowed drop.</p>"
            "<p>If the gates fail, the candidate remains registered but the current active model is "
            "left unchanged. The local detector can use the active model automatically when no model "
            "file is selected.</p>"
        )

    def createInstance(self):  # noqa: N802
        return RegisterCandidateModel()

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFile(
            self.CANDIDATE, tr("Candidate model (.zip, .onnx or .tflite)"), behavior=FILE_BEHAVIOR,
            fileFilter="Grid Mapper model (*.zip *.onnx *.tflite)"))
        self.addParameter(QgsProcessingParameterString(
            self.MODEL_ID, tr("Model/version ID (optional; otherwise read from metadata)"), "", optional=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_GAIN, tr("Minimum overall validation improvement"), NUM_DOUBLE, 0.005,
            minValue=0.0, maxValue=1.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_GEO_DROP, tr("Maximum allowed drop in any shared geography"), NUM_DOUBLE, 0.03,
            minValue=0.0, maxValue=1.0))
        self.addParameter(QgsProcessingParameterBoolean(
            self.PROMOTE, tr("Promote automatically when all gates pass"), True))
        self.addOutput(QgsProcessingOutputString("DECISION", tr("Promotion decision")))
        self.addOutput(QgsProcessingOutputString("ACTIVE_MODEL", tr("Active model")))
        self.addOutput(QgsProcessingOutputString("REGISTRY", tr("Registry folder")))
        self.addOutput(QgsProcessingOutputNumber("CANDIDATE_METRIC", tr("Candidate validation metric")))
        self.addOutput(QgsProcessingOutputNumber("CURRENT_METRIC", tr("Current validation metric")))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        candidate = self.parameterAsFile(parameters, self.CANDIDATE, context)
        model_id = self.parameterAsString(parameters, self.MODEL_ID, context) or None
        min_gain = self.parameterAsDouble(parameters, self.MIN_GAIN, context)
        max_drop = self.parameterAsDouble(parameters, self.MAX_GEO_DROP, context)
        promote = self.parameterAsBool(parameters, self.PROMOTE, context)
        root = default_registry_dir(QgsApplication.qgisSettingsDirPath())
        try:
            entry, promoted, registry = register_candidate(
                root, candidate, model_id=model_id, min_gain=min_gain,
                max_geo_regression=max_drop, promote_if_passed=promote)
        except (OSError, ValueError) as exc:
            raise QgsProcessingException(str(exc)) from exc
        decision = entry.get("promotion_decision") or {}
        verdict = "PROMOTED" if promoted else ("PASSED (not promoted)" if decision.get("passed") else "REJECTED")
        message = f"{verdict}: {decision.get('reason', 'no reason supplied')}"
        feedback.pushInfo(tr(message))
        if decision.get("geo_regressions"):
            for geo, info in decision["geo_regressions"].items():
                feedback.pushInfo(tr(
                    f"  {geo}: {info['current']:.4f} -> {info['candidate']:.4f} "
                    f"(drop {info['drop']:.4f})"))
        if decision.get("missing_geographies"):
            feedback.pushInfo(tr("Missing validation geographies: " + ", ".join(decision["missing_geographies"])))
        active = registry.get("active_model_id") or ""
        return {
            "DECISION": message,
            "ACTIVE_MODEL": active,
            "REGISTRY": root,
            "CANDIDATE_METRIC": decision.get("candidate_metric") if decision.get("candidate_metric") is not None else -1,
            "CURRENT_METRIC": decision.get("current_metric") if decision.get("current_metric") is not None else -1,
        }
