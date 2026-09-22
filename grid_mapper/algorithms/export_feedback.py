"""Export the append-only feedback store to a versioned COCO snapshot."""
from datetime import datetime, timezone

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputNumber,
    QgsProcessingOutputString,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from ..core.feedback_store import export_snapshot
from .base_detect import ICON, tr


class ExportFeedbackDataset(QgsProcessingAlgorithm):
    DATASET = "DATASET"
    SNAPSHOT = "SNAPSHOT"
    CLASS_NAME = "CLASS_NAME"

    def name(self):
        return "export_feedback_dataset"

    def displayName(self):  # noqa: N802
        return tr("Export continual-learning dataset snapshot")

    def group(self):
        return tr("Continual learning")

    def groupId(self):  # noqa: N802
        return "continual_learning"

    def icon(self):
        return QIcon(ICON)

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Exports the reviewed feedback store into a COCO dataset with fixed "
            "<code>train/valid/test</code> splits. Negative examples are preserved as images with "
            "no annotations.</p><p>Point the RF-DETR Colab notebook at the resulting snapshot. "
            "A snapshot is immutable; use a new name for the next training round.</p>"
        )

    def createInstance(self):  # noqa: N802
        return ExportFeedbackDataset()

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.DATASET, tr("Persistent feedback dataset folder")))
        self.addParameter(QgsProcessingParameterString(
            self.SNAPSHOT, tr("Snapshot name"), datetime.now(timezone.utc).strftime("feedback_%Y%m%d")))
        self.addParameter(QgsProcessingParameterString(self.CLASS_NAME, tr("Class name"), "substation"))
        self.addOutput(QgsProcessingOutputString("SNAPSHOT_FOLDER", tr("Snapshot folder")))
        self.addOutput(QgsProcessingOutputNumber("SAMPLES", tr("Samples exported")))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        root = self.parameterAsString(parameters, self.DATASET, context)
        name = self.parameterAsString(parameters, self.SNAPSHOT, context)
        class_name = self.parameterAsString(parameters, self.CLASS_NAME, context) or "substation"
        try:
            out, summary = export_snapshot(root, name, class_name)
        except (OSError, ValueError) as exc:
            raise QgsProcessingException(str(exc)) from exc
        n = int(summary.get("total_records") or 0)
        feedback.pushInfo(tr(f"Exported {n} reviewed samples to {out}"))
        return {"SNAPSHOT_FOLDER": out, "SAMPLES": n}
