import numpy as np
from qgis.core import (
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)

from ..core import tiling
from ..core.compat import NUM_INT
from ..core.roboflow_client import RoboflowClient
from .base_detect import BaseDetectionAlgorithm, gsd_center_note, tr

TASKS = ["detect", "instance", "semantic"]


def mask_to_polygons(mask, keep_values, scale_x, scale_y, min_pixels=20):
    """Vectorise class pixels of a mask into pixel-space rings using GDAL."""
    from osgeo import gdal, ogr

    binary = np.isin(mask, list(keep_values)).astype(np.uint8)
    if binary.sum() < min_pixels:
        return []
    h, w = binary.shape
    ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((0, 1, 0, 0, 0, 1))  # map coords == pixel coords
    band = ds.GetRasterBand(1)
    band.WriteArray(binary)
    drv = ogr.GetDriverByName("MEM") or ogr.GetDriverByName("Memory")  # GDAL >= 3.11 / older
    vds = drv.CreateDataSource("mask")
    lyr = vds.CreateLayer("poly", geom_type=ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(band, band, lyr, 0, [], callback=None)
    rings = []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        if geom is None or feat.GetField("v") != 1 or geom.GetArea() < min_pixels:
            continue
        geom = geom.SimplifyPreserveTopology(1.0)
        outer = geom.GetGeometryRef(0)
        rings.append([(outer.GetX(i) * scale_x, outer.GetY(i) * scale_y)
                      for i in range(outer.GetPointCount())])
    return rings


class DetectSubstationsRoboflow(BaseDetectionAlgorithm):
    MODEL_ID = "MODEL_ID"
    VERSION = "VERSION"
    API_KEY = "API_KEY"
    TASK = "TASK"
    BASE_URL = "BASE_URL"

    DEFAULT_SCORE = 0.4

    def name(self):
        return "detect_roboflow"

    def displayName(self):  # noqa: N802
        return tr("Detect substations (Roboflow API)")

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Scans the imagery tile by tile and sends each tile to a model hosted on Roboflow.</p>"
            "<p>Thesis models: <b>ss-2 / version 1</b> - Roboflow 2.0 object detection "
            "(mAP 85.6 %, precision 90.3 %, recall 85.5 %) and <b>ss-ovfnj / 1</b> - SegFormer "
            "semantic segmentation (mIoU 79 %). Use your private API key from Roboflow "
            "(Settings &gt; API Keys) or set the ROBOFLOW_API_KEY environment variable.</p>"
            "<p>Instance-segmentation models return true footprints; object detection returns "
            "boxes; semantic segmentation masks are vectorised with GDAL.</p>"
            "<p>Every tile is uploaded to Roboflow, so API usage counts against your plan.</p>"
        ) + gsd_center_note()

    def createInstance(self):  # noqa: N802
        return DetectSubstationsRoboflow()

    def add_model_parameters(self):
        self.addParameter(QgsProcessingParameterString(self.MODEL_ID, tr("Roboflow model ID"), "ss-2"))
        self.addParameter(QgsProcessingParameterNumber(
            self.VERSION, tr("Model version"), NUM_INT, 1, minValue=1))
        self.addParameter(QgsProcessingParameterString(self.API_KEY, tr("Roboflow API key"), "",
                                                       optional=True))
        self.addParameter(QgsProcessingParameterEnum(
            self.TASK, tr("Model type"),
            [tr("Object detection (boxes)"), tr("Instance segmentation (footprints)"),
             tr("Semantic segmentation (mask)")], defaultValue=0))
        self.addParameter(QgsProcessingParameterString(
            self.BASE_URL, tr("Inference URL (leave empty for Roboflow cloud)"), "", optional=True))

    def init_detector(self, parameters, context, feedback):
        task = TASKS[self.parameterAsEnum(parameters, self.TASK, context)]
        try:
            self.client = RoboflowClient(
                self.parameterAsString(parameters, self.MODEL_ID, context),
                self.parameterAsInt(parameters, self.VERSION, context),
                self.parameterAsString(parameters, self.API_KEY, context),
                task=task,
                base_url=self.parameterAsString(parameters, self.BASE_URL, context),
                confidence=self.parameterAsDouble(parameters, self.SCORE, context),
                overlap=self.parameterAsDouble(parameters, self.NMS_IOU, context),
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        self.task = task
        feedback.pushInfo(tr(f"Roboflow endpoint: {self.client.base}/{self.client.model_id}/"
                             f"{self.client.version} ({task})"))

    def model_name(self):
        c = getattr(self, "client", None)
        return f"roboflow:{c.model_id}/{c.version}" if c else "roboflow"

    def detect_tile(self, image, feedback):
        w, h = image.width(), image.height()
        resp = self.client.infer(tiling.qimage_to_bytes(image, "JPG", 90))
        if self.task != "semantic":
            dets = self.client.parse_detections(resp, w, h)
            if self.task == "detect":
                for d in dets:
                    d["polygon"] = None
            return dets
        mask, class_map = self.client.decode_mask(resp)
        keep = [k for k, v in class_map.items() if str(v).lower() not in ("background", "_background_")]
        if not keep:
            keep = [v for v in np.unique(mask) if v != 0]
        out = []
        for ring in mask_to_polygons(mask, keep, w / mask.shape[1], h / mask.shape[0]):
            xs, ys = [p[0] for p in ring], [p[1] for p in ring]
            out.append({"xmin": min(xs), "ymin": min(ys), "xmax": max(xs), "ymax": max(ys),
                        "score": 1.0, "label": "substation", "polygon": ring})
        return out
