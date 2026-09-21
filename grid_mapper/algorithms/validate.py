from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFields,
    QgsGeometry,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputNumber,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsSpatialIndex,
)
from qgis.PyQt.QtGui import QIcon

from .base_detect import ICON, tr
from ..core.compat import FAST_INSERT, FIELD_NUMERIC, NUM_DOUBLE, SRC_ANY, make_field


class ValidateDetections(QgsProcessingAlgorithm):
    DETECTIONS = "DETECTIONS"
    SCORE_FIELD = "SCORE_FIELD"
    REFERENCE = "REFERENCE"
    DISTANCE = "DISTANCE"
    OUTPUT = "OUTPUT"
    MISSED = "MISSED"
    REPORT = "REPORT"

    def name(self):
        return "validate_detections"

    def displayName(self):  # noqa: N802
        return tr("Validate detections against reference substations")

    def group(self):
        return tr("Substation detection")

    def groupId(self):  # noqa: N802
        return "detection"

    def icon(self):
        return QIcon(ICON)

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Compares detected substations with a reference layer (e.g. the thesis "
            "<code>Substations.shp</code>, utility / KETRACO data or OpenStreetMap "
            "<code>power=substation</code>). A detection is a true positive when it overlaps, or "
            "lies within the match distance of, a reference substation not already matched "
            "(greedy, highest score first).</p>"
            "<p>Outputs: detections with a <code>match</code> field (TP / FP), missed reference "
            "substations (FN), and precision, recall and F1. Restrict the reference layer to the "
            "scanned area first, or reference sites outside it count as missed.</p>")

    def createInstance(self):  # noqa: N802
        return ValidateDetections()

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.DETECTIONS, tr("Detected substations"), [SRC_ANY]))
        self.addParameter(QgsProcessingParameterField(
            self.SCORE_FIELD, tr("Score field (optional)"), "score", self.DETECTIONS,
            FIELD_NUMERIC, optional=True))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.REFERENCE, tr("Reference substations"), [SRC_ANY]))
        self.addParameter(QgsProcessingParameterNumber(
            self.DISTANCE, tr("Match distance (metres)"), NUM_DOUBLE,
            150, minValue=0))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, tr("Validated detections")))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.MISSED, tr("Missed reference substations"), optional=True))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.REPORT, tr("Report"), "HTML files (*.html)", optional=True, createByDefault=False))
        for key in ("TP", "FP", "FN"):
            self.addOutput(QgsProcessingOutputNumber(key, key))
        for key in ("PRECISION", "RECALL", "F1"):
            self.addOutput(QgsProcessingOutputNumber(key, key.title()))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        det = self.parameterAsSource(parameters, self.DETECTIONS, context)
        ref = self.parameterAsSource(parameters, self.REFERENCE, context)
        if det is None or ref is None:
            raise QgsProcessingException(tr("Invalid input layers"))
        score_field = self.parameterAsString(parameters, self.SCORE_FIELD, context)
        max_dist = self.parameterAsDouble(parameters, self.DISTANCE, context)

        # work in a metric CRS: a local UTM zone derived from the detections' centre
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        c = QgsCoordinateTransform(det.sourceCrs(), wgs84, context.transformContext()) \
            .transformBoundingBox(det.sourceExtent()).center()
        zone = int((c.x() + 180) // 6) + 1
        epsg = (32600 if c.y() >= 0 else 32700) + max(1, min(60, zone))
        metric = QgsCoordinateReferenceSystem(f"EPSG:{epsg}")
        tr_det = QgsCoordinateTransform(det.sourceCrs(), metric, context.transformContext())
        tr_ref = QgsCoordinateTransform(ref.sourceCrs(), metric, context.transformContext())
        feedback.pushInfo(tr(f"Measuring distances in {metric.authid()}"))

        ref_geoms, index = {}, QgsSpatialIndex()
        for f in ref.getFeatures():
            g = QgsGeometry(f.geometry())
            if g.isEmpty():
                continue
            g.transform(tr_ref)
            nf = QgsFeature(f.id())
            nf.setGeometry(g)
            index.addFeature(nf)
            ref_geoms[f.id()] = (g, f)

        dets = []
        for f in det.getFeatures():
            g = QgsGeometry(f.geometry())
            if g.isEmpty():
                continue
            g.transform(tr_det)
            s = f[score_field] if score_field and f.fields().indexOf(score_field) >= 0 else 1.0
            dets.append((float(s) if s not in (None, "") else 0.0, f, g))
        dets.sort(key=lambda t: -t[0])

        matched_ref = set()
        results = []
        for score, f, g in dets:
            best, best_d = None, None
            search = g.boundingBox().buffered(max_dist)
            for rid in index.intersects(search):
                if rid in matched_ref:
                    continue
                d = g.distance(ref_geoms[rid][0])
                if d <= max_dist and (best_d is None or d < best_d):
                    best, best_d = rid, d
            if best is not None:
                matched_ref.add(best)
                results.append((f, "TP", best, best_d))
            else:
                results.append((f, "FP", None, None))

        fields = QgsFields(det.fields())
        fields.append(make_field("match", "string", length=4))
        fields.append(make_field("ref_fid", "long"))
        fields.append(make_field("dist_m", "double", length=10, prec=1))
        sink, dest = self.parameterAsSink(parameters, self.OUTPUT, context, fields,
                                          det.wkbType(), det.sourceCrs())
        for f, m, rid, d in results:
            nf = QgsFeature(fields)
            nf.setGeometry(f.geometry())
            nf.setAttributes(f.attributes() + [m, rid, round(d, 1) if d is not None else None])
            sink.addFeature(nf, FAST_INSERT)

        missed_sink, missed_dest = self.parameterAsSink(parameters, self.MISSED, context, ref.fields(),
                                                        ref.wkbType(), ref.sourceCrs())
        missed = [rid for rid in ref_geoms if rid not in matched_ref]
        if missed_sink is not None:
            for rid in missed:
                missed_sink.addFeature(ref_geoms[rid][1], FAST_INSERT)

        tp = sum(1 for r in results if r[1] == "TP")
        fp = len(results) - tp
        fn = len(missed)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        feedback.pushInfo(tr(
            f"TP={tp}  FP={fp}  FN={fn}  precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}"))

        out = {self.OUTPUT: dest, "TP": tp, "FP": fp, "FN": fn,
               "PRECISION": precision, "RECALL": recall, "F1": f1}
        if missed_sink is not None:
            out[self.MISSED] = missed_dest
        report = self.parameterAsFileOutput(parameters, self.REPORT, context)
        if report:
            with open(report, "w", encoding="utf-8") as fh:
                fh.write(
                    "<html><head><meta charset='utf-8'><title>Substation detection validation</title></head>"
                    "<body style='font-family:sans-serif'><h2>Substation detection validation</h2>"
                    f"<p>Match distance: {max_dist:g} m ({metric.authid()})</p>"
                    "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
                    f"<tr><td>True positives</td><td>{tp}</td></tr>"
                    f"<tr><td>False positives</td><td>{fp}</td></tr>"
                    f"<tr><td>Missed (false negatives)</td><td>{fn}</td></tr>"
                    f"<tr><td>Precision</td><td>{precision:.1%}</td></tr>"
                    f"<tr><td>Recall</td><td>{recall:.1%}</td></tr>"
                    f"<tr><td>F1</td><td>{f1:.1%}</td></tr></table></body></html>\n")
            out[self.REPORT] = report
        return out

