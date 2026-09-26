import csv
import os
import re

from ..core.xmltext import escape

from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsFields,
    QgsGeometry,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsSpatialIndex,
)
from qgis.PyQt.QtGui import QIcon, QImage

from ..core import tiling
from ..core.compat import FAST_INSERT, NUM_DOUBLE, NUM_INT, SRC_ANY, SRC_POLYGON, WKB_POLYGON, make_field
from .base_detect import ICON, tr

FORMATS = [("JPG", ".jpg", ".jgw"), ("PNG", ".png", ".pgw")]
LABEL_FORMATS = ["none", "voc", "yolo"]


def _safe(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "chip"


class ExtractTrainingChips(QgsProcessingAlgorithm):
    POINTS = "POINTS"
    NAME_FIELD = "NAME_FIELD"
    INPUT = "INPUT"
    CHIP_SIZE = "CHIP_SIZE"
    GSD = "GSD"
    FORMAT = "FORMAT"
    FOOTPRINTS = "FOOTPRINTS"
    CLASS_NAME = "CLASS_NAME"
    LABELS = "LABELS"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    OUTPUT = "OUTPUT"

    def name(self):
        return "extract_chips"

    def displayName(self):  # noqa: N802
        return tr("Extract training chips around substations")

    def group(self):
        return tr("Training data")

    def groupId(self):  # noqa: N802
        return "training"

    def icon(self):
        return QIcon(ICON)

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Reproduces the thesis dataset step: for every known substation (point or polygon) "
            "an image chip (default 750 x 750 px) is rendered from the imagery layer and saved with "
            "a world file (.jgw / .pgw) and CRS (.aux.xml), so it opens georeferenced in QGIS.</p>"
            "<p>If a layer of substation <b>footprint polygons</b> is supplied, bounding-box labels "
            "are written for every footprint visible in the chip - Pascal VOC XML (LabelImg / "
            "TensorFlow Object Detection API, as used in the Colab notebooks) or YOLO txt "
            "(Roboflow / Ultralytics). Without footprints, label the chips in Roboflow or LabelImg.</p>"
            "<p>An <code>index.csv</code> and a footprint layer of all chips are also produced.</p>")

    def createInstance(self):  # noqa: N802
        return ExtractTrainingChips()

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.POINTS, tr("Substation locations (points or polygons)"),
            [SRC_ANY]))
        self.addParameter(QgsProcessingParameterField(
            self.NAME_FIELD, tr("Field used for file names (optional)"), parentLayerParameterName=self.POINTS,
            optional=True))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, tr("Imagery layer")))
        self.addParameter(QgsProcessingParameterNumber(
            self.CHIP_SIZE, tr("Chip size (pixels)"), NUM_INT, 750,
            minValue=64, maxValue=4096))
        self.addParameter(QgsProcessingParameterNumber(
            self.GSD, tr("Ground resolution (metres per pixel)"), NUM_DOUBLE,
            0.3, minValue=0.02, maxValue=30))
        self.addParameter(QgsProcessingParameterEnum(
            self.FORMAT, tr("Image format"), [f[0] for f in FORMATS], defaultValue=0))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.FOOTPRINTS, tr("Substation footprint polygons for labels (optional)"),
            [SRC_POLYGON], optional=True))
        self.addParameter(QgsProcessingParameterString(
            self.CLASS_NAME, tr("Class name"), "substation"))
        self.addParameter(QgsProcessingParameterEnum(
            self.LABELS, tr("Label format"),
            [tr("None"), tr("Pascal VOC XML"), tr("YOLO txt")], defaultValue=1))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, tr("Output folder")))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, tr("Chip footprints"), SRC_POLYGON))

    def prepareAlgorithm(self, parameters, context, feedback):  # noqa: N802
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException(tr("Invalid imagery layer"))
        self._layer = layer.clone()
        self._layer_crs = layer.crs()
        self._transform_context = context.transformContext()
        return True

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        source = self.parameterAsSource(parameters, self.POINTS, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.POINTS))
        name_field = self.parameterAsString(parameters, self.NAME_FIELD, context)
        size = self.parameterAsInt(parameters, self.CHIP_SIZE, context)
        gsd = self.parameterAsDouble(parameters, self.GSD, context)
        fmt, ext, wext = FORMATS[self.parameterAsEnum(parameters, self.FORMAT, context)]
        class_name = self.parameterAsString(parameters, self.CLASS_NAME, context) or "substation"
        label_fmt = LABEL_FORMATS[self.parameterAsEnum(parameters, self.LABELS, context)]
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)

        render_crs = tiling.choose_render_crs(self._layer_crs)
        to_render = QgsCoordinateTransform(source.sourceCrs(), render_crs, self._transform_context)

        fp_source = self.parameterAsSource(parameters, self.FOOTPRINTS, context)
        fp_geoms, fp_index = {}, None
        if fp_source is not None:
            fp_tr = QgsCoordinateTransform(fp_source.sourceCrs(), render_crs, self._transform_context)
            fp_index = QgsSpatialIndex()
            for f in fp_source.getFeatures():
                g = QgsGeometry(f.geometry())
                if g.isEmpty():
                    continue
                g.transform(fp_tr)
                nf = QgsFeature(f.id())
                nf.setGeometry(g)
                fp_index.addFeature(nf)
                fp_geoms[f.id()] = g
        elif label_fmt != "none":
            feedback.pushInfo(tr("No footprint layer given: chips will be written without labels."))

        fields = QgsFields()
        fields.append(make_field("chip", "string", length=120))
        fields.append(make_field("src_fid", "long"))
        fields.append(make_field("n_labels", "int"))
        sink, dest = self.parameterAsSink(parameters, self.OUTPUT, context, fields,
                                          WKB_POLYGON, render_crs)

        wkt = render_crs.toWkt()
        total = source.featureCount() or 1
        used_names = set()
        rows = []
        for i, feat in enumerate(source.getFeatures(QgsFeatureRequest())):
            if feedback.isCanceled():
                break
            feedback.setProgress(100.0 * i / total)
            geom = QgsGeometry(feat.geometry())
            if geom.isEmpty():
                continue
            geom.transform(to_render)
            center = geom.centroid().asPoint()
            upp = tiling.units_per_pixel(render_crs, gsd, center, self._transform_context)
            rect = tiling.square_rect(center, size, upp)
            img, vis = tiling.render_tile([self._layer], render_crs, rect, size, self._transform_context)

            base = _safe(feat[name_field]) if name_field else f"{feat.id()}"
            name, k = base, 1
            while name in used_names:
                k += 1
                name = f"{base}_{k}"
            used_names.add(name)
            img_path = os.path.join(out_dir, name + ext)
            img.convertToFormat(QImage.Format.Format_RGB888).save(img_path, fmt, 95)
            with open(os.path.join(out_dir, name + wext), "w") as fh:
                fh.write("\n".join(tiling.world_file_lines(vis, size, size)) + "\n")
            with open(img_path + ".aux.xml", "w") as fh:
                fh.write(f"<PAMDataset><SRS>{escape(wkt)}</SRS></PAMDataset>\n")

            boxes = []
            if fp_index is not None:
                chip_geom = QgsGeometry.fromRect(vis)
                for fid in fp_index.intersects(vis):
                    inter = fp_geoms[fid].intersection(chip_geom)
                    if inter.isEmpty():
                        continue
                    bb = inter.boundingBox()
                    x1 = (bb.xMinimum() - vis.xMinimum()) / vis.width() * size
                    x2 = (bb.xMaximum() - vis.xMinimum()) / vis.width() * size
                    y1 = (vis.yMaximum() - bb.yMaximum()) / vis.height() * size
                    y2 = (vis.yMaximum() - bb.yMinimum()) / vis.height() * size
                    if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                        boxes.append((max(0, x1), max(0, y1), min(size, x2), min(size, y2)))
                if label_fmt == "voc":
                    self._write_voc(out_dir, name, ext, size, class_name, boxes)
                elif label_fmt == "yolo":
                    with open(os.path.join(out_dir, name + ".txt"), "w") as fh:
                        for x1, y1, x2, y2 in boxes:
                            fh.write(f"0 {(x1 + x2) / 2 / size:.6f} {(y1 + y2) / 2 / size:.6f} "
                                     f"{(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}\n")

            f = QgsFeature(fields)
            f.setGeometry(QgsGeometry.fromRect(vis))
            f.setAttributes([name + ext, feat.id(), len(boxes)])
            sink.addFeature(f, FAST_INSERT)
            rows.append([name + ext, feat.id(), center.x(), center.y(), render_crs.authid(), gsd, len(boxes)])

        if label_fmt == "yolo" and fp_index is not None:
            with open(os.path.join(out_dir, "classes.txt"), "w") as fh:
                fh.write(class_name + "\n")
        with open(os.path.join(out_dir, "labelmap.txt"), "w") as fh:
            fh.write(class_name + "\n")
        with open(os.path.join(out_dir, "index.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["chip", "src_fid", "center_x", "center_y", "crs", "gsd_m", "n_labels"])
            w.writerows(rows)
        feedback.pushInfo(tr(f"{len(rows)} chips written to {out_dir}"))
        return {self.OUTPUT_FOLDER: out_dir, self.OUTPUT: dest, "CHIPS": len(rows)}

    @staticmethod
    def _write_voc(out_dir, name, ext, size, class_name, boxes):
        objs = "".join(
            f"<object><name>{escape(class_name)}</name><pose>Unspecified</pose><truncated>"
            f"{int(x1 <= 0 or y1 <= 0 or x2 >= size or y2 >= size)}</truncated><difficult>0</difficult>"
            f"<bndbox><xmin>{int(round(x1))}</xmin><ymin>{int(round(y1))}</ymin>"
            f"<xmax>{int(round(x2))}</xmax><ymax>{int(round(y2))}</ymax></bndbox></object>"
            for x1, y1, x2, y2 in boxes)
        xml = (f"<annotation><folder>{escape(os.path.basename(out_dir))}</folder>"
               f"<filename>{escape(name + ext)}</filename><path>{escape(os.path.join(out_dir, name + ext))}</path>"
               f"<source><database>Grid Mapper</database></source>"
               f"<size><width>{size}</width><height>{size}</height><depth>3</depth></size>"
               f"<segmented>0</segmented>{objs}</annotation>\n")
        with open(os.path.join(out_dir, name + ".xml"), "w", encoding="utf-8") as fh:
            fh.write(xml)
