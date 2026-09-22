"""Prioritise detections that are most useful for human review."""
from qgis.core import (
    QgsFeature,
    QgsFields,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtGui import QIcon

from ..core.compat import FAST_INSERT, NUM_DOUBLE, SRC_ANY, make_field
from ..core.feedback_store import active_learning_priority
from .base_detect import ICON, tr


class PrioritiseReview(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    SCORE_FIELD = "SCORE_FIELD"
    REVIEW_FIELD = "REVIEW_FIELD"
    THRESHOLD = "THRESHOLD"
    OUTPUT = "OUTPUT"

    def name(self):
        return "prioritise_review"

    def displayName(self):  # noqa: N802
        return tr("Prioritise detections for active-learning review")

    def group(self):
        return tr("Continual learning")

    def groupId(self):  # noqa: N802
        return "continual_learning"

    def icon(self):
        return QIcon(ICON)

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Adds an <code>al_priority</code> score (0–1) to detections. Unreviewed predictions "
            "nearest the model decision threshold rank highest because confirming or rejecting them "
            "usually provides more information than reviewing very confident predictions.</p>"
            "<p>Already reviewed TP/FP/corrected features receive zero priority. Sort the result by "
            "<code>al_priority</code> descending and review the top records first.</p>"
        )

    def createInstance(self):  # noqa: N802
        return PrioritiseReview()

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT, tr("Detections"), [SRC_ANY]))
        self.addParameter(QgsProcessingParameterField(
            self.SCORE_FIELD, tr("Confidence score field"), "score", self.INPUT, optional=True))
        self.addParameter(QgsProcessingParameterField(
            self.REVIEW_FIELD, tr("Review field"), "review", self.INPUT, optional=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.THRESHOLD, tr("Decision threshold"), NUM_DOUBLE, 0.5, minValue=0.01, maxValue=0.99))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, tr("Prioritised review queue")))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        src = self.parameterAsSource(parameters, self.INPUT, context)
        if src is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        score_field = self.parameterAsString(parameters, self.SCORE_FIELD, context) or "score"
        review_field = self.parameterAsString(parameters, self.REVIEW_FIELD, context) or "review"
        threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)

        fields = QgsFields(src.fields())
        fields.append(make_field("al_priority", "double", length=8, prec=4))
        fields.append(make_field("al_reason", "string", length=64))
        sink, dest = self.parameterAsSink(parameters, self.OUTPUT, context, fields,
                                          src.wkbType(), src.sourceCrs())
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))
        total = src.featureCount() or 1
        for i, f in enumerate(src.getFeatures()):
            if feedback.isCanceled():
                break
            feedback.setProgress(100.0 * i / total)
            score = f[score_field] if f.fields().indexOf(score_field) >= 0 else threshold
            review = f[review_field] if f.fields().indexOf(review_field) >= 0 else "unreviewed"
            priority, reason = active_learning_priority(score, review, threshold)
            nf = QgsFeature(fields)
            nf.setGeometry(f.geometry())
            nf.setAttributes(f.attributes() + [round(priority, 4), reason])
            sink.addFeature(nf, FAST_INSERT)
        return {self.OUTPUT: dest}
