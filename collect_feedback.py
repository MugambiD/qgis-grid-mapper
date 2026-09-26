"""Capture reviewed detections and misses as persistent continual-learning samples."""
import os

from ..core.xmltext import escape

from qgis.core import (
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputNumber,
    QgsProcessingOutputString,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsRectangle,
)
from qgis.PyQt.QtGui import QIcon, QImage

from ..core import tiling
from ..core.compat import NUM_DOUBLE, NUM_INT, SRC_ANY
from ..core.feedback_store import (
    FeedbackStore,
    active_learning_priority,
    normalize_review,
    sample_id_for,
    stable_split,
)
from .base_detect import ICON, tr


class CollectTrainingFeedback(QgsProcessingAlgorithm):
    DETECTIONS = "DETECTIONS"
    REVIEW_FIELD = "REVIEW_FIELD"
    SCORE_FIELD = "SCORE_FIELD"
    LABEL_FIELD = "LABEL_FIELD"
    MODEL_FIELD = "MODEL_FIELD"
    INPUT = "INPUT"
    MISSED = "MISSED"
    GEOGRAPHY = "GEOGRAPHY"
    DATASET = "DATASET"
    CHIP_SIZE = "CHIP_SIZE"
    GSD = "GSD"
    VALID_PCT = "VALID_PCT"
    TEST_PCT = "TEST_PCT"
    POINT_BOX_M = "POINT_BOX_M"

    def name(self):
        return "collect_feedback"

    def displayName(self):  # noqa: N802
        return tr("Capture reviewed feedback for continual learning")

    def group(self):
        return tr("Continual learning")

    def groupId(self):  # noqa: N802
        return "continual_learning"

    def icon(self):
        return QIcon(ICON)

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Turns human QA into reusable training data. Edit the detection geometry if needed, "
            "then set the review field to <b>TP/confirmed</b>, <b>FP/rejected</b>, or "
            "<b>corrected</b>. Optionally add the <i>Missed reference substations</i> output from "
            "the validation tool.</p>"
            "<p>Positive/corrected/missed examples are saved with an annotation. False positives "
            "are saved as hard-negative images with no annotation. Every example receives a fixed "
            "train/valid/test split, so later additions never reshuffle the validation set.</p>"
            "<p>The dataset is append-only. Re-running the same reviewed feature does not duplicate it.</p>"
        )

    def createInstance(self):  # noqa: N802
        return CollectTrainingFeedback()

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.DETECTIONS, tr("Reviewed detections"), [SRC_ANY]))
        self.addParameter(QgsProcessingParameterField(
            self.REVIEW_FIELD, tr("Review field"), "review", self.DETECTIONS, optional=True))
        self.addParameter(QgsProcessingParameterField(
            self.SCORE_FIELD, tr("Confidence score field (optional)"), "score", self.DETECTIONS,
            optional=True))
        self.addParameter(QgsProcessingParameterField(
            self.LABEL_FIELD, tr("Class/label field (optional)"), "label", self.DETECTIONS,
            optional=True))
        self.addParameter(QgsProcessingParameterField(
            self.MODEL_FIELD, tr("Model field (optional)"), "model", self.DETECTIONS,
            optional=True))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, tr("Imagery layer used for the detections")))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.MISSED, tr("Missed substations / false negatives (optional)"), [SRC_ANY], optional=True))
        self.addParameter(QgsProcessingParameterString(
            self.GEOGRAPHY, tr("Geography / validation group (e.g. Kenya, Zambia)"), "unspecified"))
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.DATASET, tr("Persistent feedback dataset folder")))
        self.addParameter(QgsProcessingParameterNumber(
            self.CHIP_SIZE, tr("Training chip size (pixels)"), NUM_INT, 750, minValue=128, maxValue=4096))
        self.addParameter(QgsProcessingParameterNumber(
            self.GSD, tr("Ground resolution (metres per pixel)"), NUM_DOUBLE, 0.5, minValue=0.02, maxValue=30))
        self.addParameter(QgsProcessingParameterNumber(
            self.VALID_PCT, tr("Fixed validation split (%)"), NUM_DOUBLE, 10.0, minValue=0, maxValue=40))
        self.addParameter(QgsProcessingParameterNumber(
            self.TEST_PCT, tr("Fixed test split (%)"), NUM_DOUBLE, 10.0, minValue=0, maxValue=40))
        self.addParameter(QgsProcessingParameterNumber(
            self.POINT_BOX_M, tr("Default annotation width for point-only misses (metres)"),
            NUM_DOUBLE, 80.0, minValue=5, maxValue=1000))
        for key in ("CAPTURED", "POSITIVES", "NEGATIVES", "MISSED_COUNT", "DUPLICATES"):
            self.addOutput(QgsProcessingOutputNumber(key, key.replace("_", " ").title()))
        self.addOutput(QgsProcessingOutputString("MANIFEST", tr("Feedback manifest")))

    def prepareAlgorithm(self, parameters, context, feedback):  # noqa: N802
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException(tr("Invalid imagery layer"))
        self._layer = layer.clone()
        self._layer_crs = layer.crs()
        self._layer_name = layer.name()
        self._transform_context = context.transformContext()
        return True

    @staticmethod
    def _value(feat, field_name, default=None):
        if not field_name:
            return default
        idx = feat.fields().indexOf(field_name)
        if idx < 0:
            return default
        value = feat[field_name]
        return default if value in (None, "") else value

    @staticmethod
    def _square_if_degenerate(geom, width_m, m_per_unit):
        bb = geom.boundingBox()
        if bb.width() > 1e-9 and bb.height() > 1e-9:
            return geom
        c = geom.centroid().asPoint()
        half = width_m / max(m_per_unit, 1e-12) / 2.0
        return QgsGeometry.fromRect(QgsRectangle(c.x() - half, c.y() - half, c.x() + half, c.y() + half))

    @staticmethod
    def _annotation(geom, vis, size):
        clipped = geom.intersection(QgsGeometry.fromRect(vis))
        if clipped.isEmpty():
            return None
        bb = clipped.boundingBox()
        x1 = (bb.xMinimum() - vis.xMinimum()) / vis.width() * size
        x2 = (bb.xMaximum() - vis.xMinimum()) / vis.width() * size
        y1 = (vis.yMaximum() - bb.yMaximum()) / vis.height() * size
        y2 = (vis.yMaximum() - bb.yMinimum()) / vis.height() * size
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(size), x2), min(float(size), y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        rings = []
        try:
            if clipped.isMultipart():
                polys = clipped.asMultiPolygon()
                rings = [p[0] for p in polys if p and p[0]]
            else:
                poly = clipped.asPolygon()
                rings = [poly[0]] if poly and poly[0] else []
        except (TypeError, ValueError):
            rings = []

        seg = []
        best_area = -1.0
        for ring in rings:
            pts = []
            for p in ring:
                px = (p.x() - vis.xMinimum()) / vis.width() * size
                py = (vis.yMaximum() - p.y()) / vis.height() * size
                pts.extend([max(0.0, min(size, px)), max(0.0, min(size, py))])
            if len(pts) < 6:
                continue
            area = 0.0
            xy = list(zip(pts[0::2], pts[1::2]))
            for (ax, ay), (bx, by) in zip(xy, xy[1:] + xy[:1]):
                area += ax * by - bx * ay
            if abs(area) > best_area:
                best_area, seg = abs(area), pts
        return {"class": "substation", "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": [seg] if seg else []}

    def _capture(self, store, feat, geom, status, render_crs, size, gsd, geography,
                 valid_pct, test_pct, point_box_m, score, label, model, source_kind):
        center = geom.centroid().asPoint()
        upp = tiling.units_per_pixel(render_crs, gsd, center, self._transform_context)
        m_per_unit = gsd / upp
        ann_geom = self._square_if_degenerate(geom, point_box_m, m_per_unit)
        identity = sample_id_for(
            source_kind, feat.id(), status, geography, model, self._layer_name,
            round(center.x(), 6), round(center.y(), 6), ann_geom.asWkt(6), size, gsd,
        )
        if store.has(identity):
            return False

        rect = tiling.square_rect(center, size, upp)
        img, vis = tiling.render_tile([self._layer], render_crs, rect, size, self._transform_context)
        rel_img = os.path.join("samples", identity + ".jpg")
        rel_world = os.path.join("samples", identity + ".jgw")
        rel_aux = rel_img + ".aux.xml"
        abs_img = os.path.join(store.root, rel_img)
        abs_world = os.path.join(store.root, rel_world)
        abs_aux = os.path.join(store.root, rel_aux)
        img.convertToFormat(QImage.Format.Format_RGB888).save(abs_img, "JPG", 95)
        with open(abs_world, "w", encoding="utf-8") as fh:
            fh.write("\n".join(tiling.world_file_lines(vis, size, size)) + "\n")
        with open(abs_aux, "w", encoding="utf-8") as fh:
            fh.write(f"<PAMDataset><SRS>{escape(render_crs.toWkt())}</SRS></PAMDataset>\n")

        annotations = []
        if status in {"positive", "corrected", "missed"}:
            ann = self._annotation(ann_geom, vis, size)
            if ann:
                ann["class"] = str(label or "substation")
                annotations = [ann]
        priority, reason = active_learning_priority(score, status)
        try:
            score_value = float(score) if score not in (None, "") else None
        except (TypeError, ValueError):
            score_value = None
        record = {
            "sample_id": identity,
            "status": status,
            "split": stable_split(identity, valid_pct, test_pct),
            "geography": geography or "unspecified",
            "model": str(model or ""),
            "score": score_value,
            "label": str(label or "substation"),
            "imagery": self._layer_name,
            "source_fid": int(feat.id()),
            "source_kind": source_kind,
            "image": rel_img.replace(os.sep, "/"),
            "world_file": rel_world.replace(os.sep, "/"),
            "aux_xml": rel_aux.replace(os.sep, "/"),
            "chip_size": int(size),
            "width": int(size),
            "height": int(size),
            "gsd_m": float(gsd),
            "crs": render_crs.authid(),
            "geometry_wkt": ann_geom.asWkt(6),
            "annotations": annotations,
            "active_learning_priority": priority,
            "active_learning_reason": reason,
        }
        if store.append(record):
            return True
        for p in (abs_img, abs_world, abs_aux):
            try:
                os.remove(p)
            except OSError:
                pass
        return False

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        detections = self.parameterAsSource(parameters, self.DETECTIONS, context)
        if detections is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.DETECTIONS))
        review_field = self.parameterAsString(parameters, self.REVIEW_FIELD, context) or "review"
        score_field = self.parameterAsString(parameters, self.SCORE_FIELD, context) or "score"
        label_field = self.parameterAsString(parameters, self.LABEL_FIELD, context) or "label"
        model_field = self.parameterAsString(parameters, self.MODEL_FIELD, context) or "model"
        missed = self.parameterAsSource(parameters, self.MISSED, context)
        geography = self.parameterAsString(parameters, self.GEOGRAPHY, context) or "unspecified"
        root = self.parameterAsString(parameters, self.DATASET, context)
        size = self.parameterAsInt(parameters, self.CHIP_SIZE, context)
        gsd = self.parameterAsDouble(parameters, self.GSD, context)
        valid_pct = self.parameterAsDouble(parameters, self.VALID_PCT, context)
        test_pct = self.parameterAsDouble(parameters, self.TEST_PCT, context)
        point_box_m = self.parameterAsDouble(parameters, self.POINT_BOX_M, context)
        if valid_pct + test_pct >= 100:
            raise QgsProcessingException(tr("Validation + test percentages must be below 100 %."))

        store = FeedbackStore(root)
        render_crs = tiling.choose_render_crs(self._layer_crs)
        tr_det = QgsCoordinateTransform(detections.sourceCrs(), render_crs, self._transform_context)
        counts = {"captured": 0, "positive": 0, "negative": 0, "missed": 0, "duplicates": 0}
        features = list(detections.getFeatures())
        total = len(features) + (missed.featureCount() if missed is not None else 0)
        total = max(1, total)
        done = 0

        for feat in features:
            if feedback.isCanceled():
                break
            done += 1
            feedback.setProgress(100.0 * done / total)
            raw_review = self._value(feat, review_field, "unreviewed")
            status = normalize_review(raw_review)
            if status not in {"positive", "negative", "corrected"}:
                continue
            geom = QgsGeometry(feat.geometry())
            if geom.isEmpty():
                continue
            geom.transform(tr_det)
            score = self._value(feat, score_field, None)
            label = self._value(feat, label_field, "substation")
            model = self._value(feat, model_field, "")
            added = self._capture(store, feat, geom, status, render_crs, size, gsd, geography,
                                  valid_pct, test_pct, point_box_m, score, label, model, "detection")
            if added:
                counts["captured"] += 1
                counts["negative" if status == "negative" else "positive"] += 1
            else:
                counts["duplicates"] += 1

        if missed is not None:
            tr_missed = QgsCoordinateTransform(missed.sourceCrs(), render_crs, self._transform_context)
            for feat in missed.getFeatures():
                if feedback.isCanceled():
                    break
                done += 1
                feedback.setProgress(100.0 * done / total)
                geom = QgsGeometry(feat.geometry())
                if geom.isEmpty():
                    continue
                geom.transform(tr_missed)
                added = self._capture(store, feat, geom, "missed", render_crs, size, gsd, geography,
                                      valid_pct, test_pct, point_box_m, None, "substation", "", "missed")
                if added:
                    counts["captured"] += 1
                    counts["positive"] += 1
                    counts["missed"] += 1
                else:
                    counts["duplicates"] += 1

        feedback.pushInfo(tr(
            f"Captured {counts['captured']} new samples: {counts['positive']} positive/corrected/missed, "
            f"{counts['negative']} hard negatives; {counts['duplicates']} duplicates skipped."))
        summary = store.summary()
        feedback.pushInfo(tr(f"Dataset now contains {summary['records']} reviewed samples."))
        return {
            "CAPTURED": counts["captured"], "POSITIVES": counts["positive"],
            "NEGATIVES": counts["negative"], "MISSED_COUNT": counts["missed"],
            "DUPLICATES": counts["duplicates"], "MANIFEST": store.manifest_path,
        }
