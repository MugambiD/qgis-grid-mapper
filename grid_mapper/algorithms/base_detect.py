"""Shared tiling / rendering / output logic for the detection algorithms."""
import os

from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsFields,
    QgsGeometry,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsRectangle,
)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon, QImage

from ..core import tiling
from ..core.compat import (FAST_INSERT, NUM_DOUBLE, NUM_INT, SRC_POINT, SRC_POLYGON,
                           WKB_POINT, WKB_POLYGON, make_field)
from ..core.postprocess import edge_touching, nms

ICON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons", "icon.png")


def tr(text):
    return QCoreApplication.translate("GridMapper", text)


class BaseDetectionAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    EXTENT = "EXTENT"
    AOI = "AOI"
    GSD = "GSD"
    TILE_SIZE = "TILE_SIZE"
    OVERLAP = "OVERLAP"
    SCORE = "SCORE"
    NMS_IOU = "NMS_IOU"
    MAX_TILES = "MAX_TILES"
    SKIP_BLANK = "SKIP_BLANK"
    SAVE_TILES = "SAVE_TILES"
    OUTPUT = "OUTPUT"
    OUTPUT_POINTS = "OUTPUT_POINTS"

    DEFAULT_TILE = 750
    DEFAULT_SCORE = 0.5

    # ---- to implement in subclasses -------------------------------------
    def add_model_parameters(self):
        raise NotImplementedError

    def init_detector(self, parameters, context, feedback):
        raise NotImplementedError

    def detect_tile(self, image, feedback):
        """Return list of dicts in *tile pixel* coords:
        {xmin, ymin, xmax, ymax, score, label, polygon (list[(px,py)] | None)}"""
        raise NotImplementedError

    def model_name(self):
        return self.displayName()

    # ---- QgsProcessingAlgorithm -----------------------------------------
    def group(self):
        return tr("Substation detection")

    def groupId(self):  # noqa: N802
        return "detection"

    def icon(self):
        return QIcon(ICON)

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, tr("Imagery layer (satellite / aerial / drone raster or XYZ basemap)")))
        self.addParameter(QgsProcessingParameterExtent(
            self.EXTENT, tr("Area to scan (extent)"), optional=True))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.AOI, tr("Area of interest polygons (optional, restricts tiles)"),
            [SRC_POLYGON], optional=True))
        self.add_model_parameters()
        p = QgsProcessingParameterNumber(
            self.GSD, tr("Ground resolution (metres per pixel)"),
            NUM_DOUBLE, 0.6, minValue=0.02, maxValue=30)
        self.addParameter(p)
        self.addParameter(QgsProcessingParameterNumber(
            self.TILE_SIZE, tr("Tile size (pixels)"), NUM_INT,
            self.DEFAULT_TILE, minValue=64, maxValue=4096))
        self.addParameter(QgsProcessingParameterNumber(
            self.OVERLAP, tr("Tile overlap (%)"), NUM_DOUBLE,
            25, minValue=0, maxValue=75))
        self.addParameter(QgsProcessingParameterNumber(
            self.SCORE, tr("Minimum confidence (0-1)"), NUM_DOUBLE,
            self.DEFAULT_SCORE, minValue=0.0, maxValue=1.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.NMS_IOU, tr("Merge overlapping detections above IoU"),
            NUM_DOUBLE, 0.3, minValue=0.0, maxValue=1.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_TILES, tr("Safety limit: maximum number of tiles"),
            NUM_INT, 2500, minValue=1))
        self.addParameter(QgsProcessingParameterBoolean(
            self.SKIP_BLANK, tr("Skip tiles without imagery"), True))
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.SAVE_TILES, tr("Save rendered tiles to folder (optional, for review/training)"),
            optional=True, createByDefault=False))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, tr("Detected substations (footprints)"), SRC_POLYGON))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_POINTS, tr("Detected substations (points)"), SRC_POINT,
            optional=True, createByDefault=False))

    def prepareAlgorithm(self, parameters, context, feedback):  # noqa: N802
        # clone the layer on the main thread so it can be rendered in the worker thread
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException(tr("Invalid imagery layer"))
        self._layer = layer.clone()
        self._layer_crs = layer.crs()
        self._layer_extent = layer.extent()
        self._layer_name = layer.name()
        self._transform_context = context.transformContext()
        return True

    # ---------------------------------------------------------------------
    def _output_fields(self):
        fields = QgsFields()
        fields.append(make_field("det_id", "int"))
        fields.append(make_field("label", "string", length=64))
        fields.append(make_field("score", "double", length=10, prec=4))
        fields.append(make_field("area_m2", "double", length=14, prec=1))
        fields.append(make_field("length_m", "double", length=10, prec=1))
        fields.append(make_field("width_m", "double", length=10, prec=1))
        fields.append(make_field("tile", "string", length=24))
        fields.append(make_field("model", "string", length=120))
        fields.append(make_field("imagery", "string", length=120))
        fields.append(make_field("review", "string", length=20))
        return fields

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        render_crs = tiling.choose_render_crs(self._layer_crs)
        extent = self.parameterAsExtent(parameters, self.EXTENT, context, render_crs)
        aoi_source = self.parameterAsSource(parameters, self.AOI, context)
        aoi_geom = None
        if aoi_source is not None:
            tr_aoi = QgsCoordinateTransform(aoi_source.sourceCrs(), render_crs, self._transform_context)
            geoms = []
            for f in aoi_source.getFeatures():
                g = QgsGeometry(f.geometry())
                if g.isEmpty():
                    continue
                g.transform(tr_aoi)
                geoms.append(g)
            if geoms:
                aoi_geom = QgsGeometry.unaryUnion(geoms)
                extent = aoi_geom.boundingBox() if extent.isEmpty() else extent.intersect(aoi_geom.boundingBox())
        if extent is None or extent.isEmpty():
            # fall back to layer extent only for real rasters (not global XYZ basemaps)
            layer_ext = QgsCoordinateTransform(self._layer_crs, render_crs, self._transform_context) \
                .transformBoundingBox(self._layer_extent)
            if layer_ext.width() > 2_000_000:  # > 2000 km: clearly a world basemap
                raise QgsProcessingException(tr(
                    "Please set 'Area to scan' (extent) or an area-of-interest polygon layer - "
                    "the imagery layer covers the whole world."))
            extent = layer_ext

        gsd = self.parameterAsDouble(parameters, self.GSD, context)
        tile_px = self.parameterAsInt(parameters, self.TILE_SIZE, context)
        overlap = self.parameterAsDouble(parameters, self.OVERLAP, context) / 100.0
        min_score = self.parameterAsDouble(parameters, self.SCORE, context)
        nms_iou = self.parameterAsDouble(parameters, self.NMS_IOU, context)
        max_tiles = self.parameterAsInt(parameters, self.MAX_TILES, context)
        skip_blank = self.parameterAsBool(parameters, self.SKIP_BLANK, context)
        save_dir = self.parameterAsString(parameters, self.SAVE_TILES, context)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        upp = tiling.units_per_pixel(render_crs, gsd, extent.center(), self._transform_context)
        n_total = tiling.count_tiles(extent, tile_px, upp, overlap)
        tiles = list(tiling.tile_grid(extent, tile_px, upp, overlap, aoi_geom))
        feedback.pushInfo(tr(
            f"Scanning {extent.width() * gsd / upp / 1000:.2f} x {extent.height() * gsd / upp / 1000:.2f} km "
            f"at {gsd} m/px in {render_crs.authid()}: {len(tiles)} tiles of {tile_px}px "
            f"({n_total} before AOI filter)."))
        if len(tiles) > max_tiles:
            raise QgsProcessingException(tr(
                f"{len(tiles)} tiles needed, above the safety limit of {max_tiles}. "
                "Use a smaller area, a coarser resolution, or raise the limit."))
        if not tiles:
            raise QgsProcessingException(tr("The area of interest produced no tiles."))

        self.init_detector(parameters, context, feedback)
        feedback.pushInfo(tr(f"Model: {self.model_name()}"))

        cand_rects, cand_scores, cand_meta, cand_edge = [], [], [], []
        layers = [self._layer]
        for i, (row, col, rect) in enumerate(tiles):
            if feedback.isCanceled():
                break
            feedback.setProgress(100.0 * i / len(tiles))
            img, vis = tiling.render_tile(layers, render_crs, rect, tile_px, self._transform_context)
            if skip_blank:
                small = tiling.qimage_to_rgb(img, (64, 64))
                if tiling.blank_fraction(small) > 0.9:
                    continue
            tile_id = f"r{row}_c{col}"
            if save_dir:
                base = os.path.join(save_dir, tile_id)
                img.convertToFormat(QImage.Format.Format_RGB888).save(base + ".jpg", "JPG", 92)
                with open(base + ".jgw", "w") as fh:
                    fh.write("\n".join(tiling.world_file_lines(vis, tile_px, tile_px)) + "\n")
                with open(base + ".jpg.aux.xml", "w") as fh:
                    fh.write(f"<PAMDataset><SRS>{render_crs.toWkt()}</SRS></PAMDataset>\n")
            try:
                dets = self.detect_tile(img, feedback)
            except QgsProcessingException:
                raise
            except Exception as exc:  # noqa: BLE001
                raise QgsProcessingException(tr(f"Detection failed on tile {tile_id}: {exc}")) from exc
            w, h = img.width(), img.height()
            for d in dets:
                if d["score"] < min_score:
                    continue
                if d.get("polygon"):
                    ring = [tiling.pixel_to_map(vis, w, h, px, py) for px, py in d["polygon"]]
                    geom = QgsGeometry.fromPolygonXY([ring + [ring[0]]]).makeValid()
                else:
                    p1 = tiling.pixel_to_map(vis, w, h, d["xmin"], d["ymin"])
                    p2 = tiling.pixel_to_map(vis, w, h, d["xmax"], d["ymax"])
                    geom = QgsGeometry.fromRect(QgsRectangle(p1, p2))
                if geom is None or geom.isEmpty():
                    continue
                bb = geom.boundingBox()
                score = d["score"]
                on_edge = edge_touching(d["xmin"], d["ymin"], d["xmax"], d["ymax"], w, h)
                rank = score * (0.85 if on_edge else 1.0)  # prefer un-clipped copies
                cand_rects.append([bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum()])
                cand_scores.append(rank)
                cand_edge.append(on_edge)
                cand_meta.append((geom, d, tile_id))
            if dets:
                feedback.pushDebugInfo(f"{tile_id}: {len(dets)} raw detections")

        keep = nms(cand_rects, cand_scores, nms_iou, cand_edge)
        feedback.pushInfo(tr(f"{len(cand_meta)} raw detections -> {len(keep)} substations after merging."))

        fields = self._output_fields()
        sink, dest = self.parameterAsSink(parameters, self.OUTPUT, context, fields,
                                          WKB_POLYGON, render_crs)
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))
        pt_sink, pt_dest = self.parameterAsSink(parameters, self.OUTPUT_POINTS, context, fields,
                                                WKB_POINT, render_crs)
        m_per_unit = gsd / upp
        for n, idx in enumerate(keep, start=1):
            geom, d, tile_id = cand_meta[idx]
            obb = geom.orientedMinimumBoundingBox()
            length = max(obb[3], obb[4]) * m_per_unit if obb and len(obb) > 4 else 0.0
            width = min(obb[3], obb[4]) * m_per_unit if obb and len(obb) > 4 else 0.0
            attrs = [n, d.get("label", "substation"), round(d["score"], 4),
                     round(geom.area() * m_per_unit ** 2, 1), round(length, 1), round(width, 1),
                     tile_id, self.model_name()[:120], self._layer_name[:120], "unreviewed"]
            f = QgsFeature(fields)
            f.setGeometry(geom)
            f.setAttributes(attrs)
            sink.addFeature(f, FAST_INSERT)
            if pt_sink is not None:
                fp = QgsFeature(fields)
                fp.setGeometry(geom.pointOnSurface())
                fp.setAttributes(attrs)
                pt_sink.addFeature(fp, FAST_INSERT)
        results = {self.OUTPUT: dest, "DETECTIONS": len(keep), "TILES": len(tiles)}
        if pt_sink is not None:
            results[self.OUTPUT_POINTS] = pt_dest
        return results


def gsd_center_note():
    return tr(
        "<p><b>Ground resolution</b>: the thesis chips were 750 x 750 px screenshots of "
        "Google / Bing satellite imagery, roughly 0.3-0.6 m/px (XYZ zoom 18-19). Use a similar "
        "value so substations appear at the scale the model learned.</p>"
        "<p><b>Imagery</b>: add a satellite basemap through Browser &gt; XYZ Tiles (e.g. Google "
        "Satellite <code>https://mt1.google.com/vt/lyrs=s&amp;x={x}&amp;y={y}&amp;z={z}</code> or Esri "
        "World Imagery), or use your own drone orthomosaic / GeoTIFF. Respect the imagery provider's "
        "terms of use when scanning large areas.</p>"
        "<p>Overlapping tiles are merged with non-maximum suppression; each result gets a "
        "<code>review</code> field so you can mark true / false positives before validation.</p>")
