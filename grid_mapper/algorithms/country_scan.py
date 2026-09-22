"""Resumable country-scale grid mapping, segment by segment.

This algorithm deliberately separates the country crawl into small jobs. OSM is
collected into a persistent GeoPackage and SQLite state database; optional AI
substation scanning uses a second queue over OSM-complete segments. Re-running
with the same workspace resumes rather than starts over.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputFile,
    QgsProcessingOutputNumber,
    QgsProcessingOutputString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessingUtils,
    QgsRectangle,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QIcon

from ..core import osm
from ..core.compat import FOLDER_BEHAVIOR, NUM_DOUBLE, NUM_INT, SRC_POLYGON, advanced
from ..core.country_outputs import CountryGeoPackage, gpkg_layer_uri
from ..core.country_scan_store import CountryScanStore
from ..core.model_registry import active_model_path, default_registry_dir
from .base_detect import ICON, tr
from .detect_roboflow import DetectSubstationsRoboflow
from .detect_tflite import DetectSubstationsTFLite
from .map_grid import IMAGERY, geometry_from_geojson, utm_for, xyz_layer

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")
ASSET_OPTIONS = ["lines", "substations", "plants", "towers", "poles"]
AI_OPTIONS = ["none", "local", "roboflow"]


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "country"


def _signature(area, segment_km, overlap_km):
    # Stable enough across runs while not writing huge WKT into the state DB.
    raw = f"{area.asWkt(5)}|{float(segment_km):.3f}|{float(overlap_km):.3f}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _bbox_with_overlap(row, overlap_km):
    west, south, east, north = row["west"], row["south"], row["east"], row["north"]
    midlat = (south + north) / 2.0
    dlat = overlap_km / 111.32
    dlon = overlap_km / max(111.32 * math.cos(math.radians(midlat)), 1e-3)
    return west - dlon, south - dlat, east + dlon, north + dlat


def build_segments(area, segment_km):
    """Create deterministic approximate-km WGS84 cells intersecting *area*."""
    if segment_km <= 0:
        raise ValueError("segment_km must be positive")
    bb = area.boundingBox()
    lat_step = segment_km / 111.32
    segments = []
    row_idx = 0
    south = bb.yMinimum()
    while south < bb.yMaximum() - 1e-12:
        north = min(bb.yMaximum(), south + lat_step)
        midlat = (south + north) / 2.0
        lon_step = segment_km / max(111.32 * math.cos(math.radians(midlat)), 1e-3)
        col_idx = 0
        west = bb.xMinimum()
        while west < bb.xMaximum() - 1e-12:
            east = min(bb.xMaximum(), west + lon_step)
            rect = QgsRectangle(west, south, east, north)
            cell = QgsGeometry.fromRect(rect)
            if area.intersects(cell):
                segments.append({
                    "segment_id": f"r{row_idx:04d}c{col_idx:04d}",
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "west": west,
                    "south": south,
                    "east": east,
                    "north": north,
                })
            west = east
            col_idx += 1
        south = north
        row_idx += 1
    return segments


def _write_segments_geojson(path, rows, place_name):
    features = []
    for r in rows:
        w, s, e, n = r["west"], r["south"], r["east"], r["north"]
        geom = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
        props = {
            "segment_id": r["segment_id"],
            "osm_status": r["status"],
            "osm_attempts": r["attempts"],
            "osm_features": r["feature_count"],
            "osm_error": r.get("last_error"),
            "ai_status": r.get("ai_status"),
            "ai_attempts": r.get("ai_attempts", 0),
            "ai_tiles": r.get("ai_tiles", 0),
            "ai_detections": r.get("ai_detections", 0),
            "ai_error": r.get("ai_last_error"),
        }
        features.append({"type": "Feature", "properties": props, "geometry": geom})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "name": place_name, "features": features}, fh)


def _element_point(el, coords):
    if el.get("type") == "way" and len(coords) >= 4 and coords[0] == coords[-1]:
        return QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in coords]]).pointOnSurface()
    if len(coords) == 1:
        return QgsGeometry.fromPointXY(QgsPointXY(*coords[0]))
    return QgsGeometry.fromMultiPointXY([QgsPointXY(x, y) for x, y in coords]).centroid()


class CountryScan(QgsProcessingAlgorithm):
    PLACE = "PLACE"
    AOI = "AOI"
    WORKSPACE = "WORKSPACE"
    ASSETS = "ASSETS"
    SEGMENT_KM = "SEGMENT_KM"
    OVERLAP_KM = "OVERLAP_KM"
    MAX_SEGMENTS = "MAX_SEGMENTS"
    DELAY = "DELAY"
    MAX_ATTEMPTS = "MAX_ATTEMPTS"
    RETRY_FAILED = "RETRY_FAILED"
    AI_MODE = "AI_MODE"
    MAX_AI_SEGMENTS = "MAX_AI_SEGMENTS"
    IMAGERY = "IMAGERY"
    IMAGERY_LAYER = "IMAGERY_LAYER"
    MODEL = "MODEL"
    RF_MODEL = "RF_MODEL"
    RF_VERSION = "RF_VERSION"
    API_KEY = "API_KEY"
    GSD = "GSD"
    SCORE = "SCORE"
    TOWN_RADIUS = "TOWN_RADIUS"
    NODE_RADIUS = "NODE_RADIUS"
    KNOWN_RADIUS = "KNOWN_RADIUS"
    AI_MAX_TILES = "AI_MAX_TILES"

    def name(self):
        return "country_scan"

    def displayName(self):  # noqa: N802
        return tr("Scan an entire country (resumable)")

    def group(self):
        return tr("Grid mapping")

    def groupId(self):  # noqa: N802
        return "grid"

    def icon(self):
        return QIcon(ICON)

    def createInstance(self):  # noqa: N802
        return CountryScan()

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Builds a country-scale grid database <b>slowly and resumably</b>. The country is divided into "
            "small cells; each cell is requested from OpenStreetMap independently and committed immediately. "
            "A timeout only affects that cell.</p>"
            "<p>Re-run with the same workspace to resume. The scan state is stored in SQLite, grid assets in a "
            "GeoPackage, successful raw OSM responses per segment in JSON, and segment status in GeoJSON.</p>"
            "<p>Optional AI is a second independent queue over OSM-complete cells. You can therefore finish the "
            "OSM country crawl first and register an RF-DETR/ONNX model later without losing progress.</p>"
            "<p><b>Recommended start:</b> 25 km OSM cells, 10-25 cells per run, 2 seconds between requests, "
            "no distribution poles. For AI, start with 1-5 cells per run and Esri or a licensed/local raster.</p>"
        )

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterString(self.PLACE, tr("Country / large place to scan"), "Zambia"))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.AOI, tr("...or use my own country/AOI polygon (optional)"), [SRC_POLYGON], optional=True))
        self.addParameter(QgsProcessingParameterFile(
            self.WORKSPACE, tr("Persistent country-scan workspace folder"), behavior=FOLDER_BEHAVIOR))
        self.addParameter(QgsProcessingParameterEnum(
            self.ASSETS, tr("OSM asset layers"),
            [tr("Power lines"), tr("Substations"), tr("Power plants"),
             tr("Transmission towers (slower)"), tr("Distribution poles (very large)")],
            allowMultiple=True, defaultValue=[0, 1, 2]))
        self.addParameter(QgsProcessingParameterNumber(
            self.SEGMENT_KM, tr("OSM segment size (km)"), NUM_DOUBLE, 25.0, minValue=5, maxValue=100))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.OVERLAP_KM, tr("OSM query overlap around each segment (km)"), NUM_DOUBLE, 1.0, minValue=0, maxValue=10)))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_SEGMENTS, tr("Maximum OSM segments this run (0 = until finished)"), NUM_INT, 20, minValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            self.DELAY, tr("Delay between OSM segments (seconds)"), NUM_DOUBLE, 2.0, minValue=0, maxValue=60))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.MAX_ATTEMPTS, tr("Maximum attempts per failed segment across runs"), NUM_INT, 5, minValue=1, maxValue=50)))
        self.addParameter(advanced(QgsProcessingParameterBoolean(
            self.RETRY_FAILED, tr("Retry previously failed segments"), True)))

        self.addParameter(QgsProcessingParameterEnum(
            self.AI_MODE, tr("Optional substation AI queue"),
            [tr("No AI - map OSM country grid only"),
             tr("Local model - ONNX RF-DETR / YOLO or TFLite"),
             tr("Roboflow hosted model")], defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_AI_SEGMENTS, tr("Maximum AI segments this run (0 = until finished)"),
            NUM_INT, 2, minValue=0))
        self.addParameter(QgsProcessingParameterEnum(
            self.IMAGERY, tr("AI imagery"), [i[0] for i in IMAGERY], defaultValue=0))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.IMAGERY_LAYER, tr("Imagery layer (when using selected layer)"), optional=True))
        self.addParameter(QgsProcessingParameterFile(
            self.MODEL, tr("Local AI model (.onnx, .tflite or model .zip)"),
            optional=True, fileFilter="Detection model (*.onnx *.zip *.tflite)"))
        self.addParameter(advanced(QgsProcessingParameterString(self.RF_MODEL, tr("Roboflow model ID"), "ss-2")))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.RF_VERSION, tr("Roboflow model version"), NUM_INT, 1, minValue=1)))
        self.addParameter(QgsProcessingParameterString(self.API_KEY, tr("Roboflow API key"), "", optional=True))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.GSD, tr("AI ground resolution (m/px)"), NUM_DOUBLE, 0.6, minValue=0.1, maxValue=5)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.SCORE, tr("Minimum AI confidence"), NUM_DOUBLE, 0.5, minValue=0, maxValue=1)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.TOWN_RADIUS, tr("AI scan radius around towns (km; cities x2)"), NUM_DOUBLE, 1.0, minValue=0)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.NODE_RADIUS, tr("AI scan radius around line ends/junctions (m)"), NUM_DOUBLE, 400, minValue=0)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.KNOWN_RADIUS, tr("AI scan radius around known substations (m)"), NUM_DOUBLE, 250, minValue=0)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.AI_MAX_TILES, tr("Safety limit: tiles per AI segment"), NUM_INT, 2500, minValue=1)))

        self.addOutput(QgsProcessingOutputFile("GRID_GPKG", tr("Country grid GeoPackage")))
        self.addOutput(QgsProcessingOutputFile("STATE_DB", tr("Resume-state SQLite database")))
        self.addOutput(QgsProcessingOutputFile("SEGMENTS_GEOJSON", tr("Segment status GeoJSON")))
        self.addOutput(QgsProcessingOutputString("WORKSPACE_OUT", tr("Workspace")))
        for key in ("SEGMENTS_TOTAL", "OSM_DONE", "OSM_PENDING", "OSM_RETRY", "OSM_FAILED",
                    "AI_DONE", "AI_PENDING", "AI_RETRY", "AI_FAILED", "ASSETS", "AI_DETECTIONS", "AI_TILES"):
            self.addOutput(QgsProcessingOutputNumber(key, key))

    def prepareAlgorithm(self, parameters, context, feedback):  # noqa: N802
        self._ai_kind = AI_OPTIONS[self.parameterAsEnum(parameters, self.AI_MODE, context)]
        self._child = None
        self._imagery = None
        self._imagery_name = ""
        self._ai_skip_reason = None
        if self._ai_kind == "none":
            return True

        if self._ai_kind == "local":
            model = self.parameterAsFile(parameters, self.MODEL, context)
            if not model:
                reg = default_registry_dir(QgsApplication.qgisSettingsDirPath())
                model = active_model_path(reg)
            if not model:
                self._ai_skip_reason = (
                    "AI queue skipped: no local model selected and no active model is registered. "
                    "OSM progress will still be saved; register/select a model and resume later."
                )
                return True
        elif self._ai_kind == "roboflow":
            key = self.parameterAsString(parameters, self.API_KEY, context) or os.environ.get("ROBOFLOW_API_KEY", "")
            if not key:
                self._ai_skip_reason = (
                    "AI queue skipped: Roboflow was selected but no API key is configured. "
                    "OSM progress will still be saved; add a key and resume later."
                )
                return True

        choice = self.parameterAsEnum(parameters, self.IMAGERY, context)
        name, url, zmax = IMAGERY[choice]
        if url is None:
            lyr = self.parameterAsRasterLayer(parameters, self.IMAGERY_LAYER, context)
            if lyr is None:
                self._ai_skip_reason = "AI queue skipped: no imagery layer was selected."
                return True
            self._imagery = lyr.clone()
            self._imagery_name = lyr.name()
        else:
            self._imagery = xyz_layer(name, url, zmax)
            self._imagery_name = name
        if self._imagery is None or not self._imagery.isValid():
            self._ai_skip_reason = f"AI queue skipped: could not open imagery '{self._imagery_name}'."
            return True

        self._child = DetectSubstationsTFLite().create() if self._ai_kind == "local" else DetectSubstationsRoboflow().create()
        self._child._layer = self._imagery
        self._child._layer_crs = self._imagery.crs()
        self._child._layer_extent = self._imagery.extent()
        self._child._layer_name = self._imagery_name
        self._child._transform_context = context.transformContext()
        return True

    def _resolve_area(self, parameters, context, feedback):
        aoi = self.parameterAsSource(parameters, self.AOI, context)
        if aoi is not None:
            to_wgs = QgsCoordinateTransform(aoi.sourceCrs(), WGS84, context.transformContext())
            geoms = []
            for f in aoi.getFeatures():
                g = QgsGeometry(f.geometry())
                g.transform(to_wgs)
                geoms.append(g)
            area = QgsGeometry.unaryUnion(geoms).makeValid()
            return "custom area", area
        place_text = self.parameterAsString(parameters, self.PLACE, context).strip()
        if not place_text:
            raise QgsProcessingException(tr("Type a country/place or provide an AOI polygon."))
        feedback.pushInfo(tr(f"Looking up '{place_text}' ..."))
        p = osm.geocode(place_text)
        area = geometry_from_geojson(p.get("geojson"))
        if area is None:
            w, s, e, n = p["bbox"]
            area = QgsGeometry.fromRect(QgsRectangle(w, s, e, n))
        return p["name"], area.makeValid()

    def _adaptive_osm_fetch(self, row, assets, cache_dir, feedback):
        bbox = _bbox_with_overlap(row, self._overlap_km)
        place = {"osm_type": None, "osm_id": 0, "bbox": bbox}
        core_flags = {
            "want_lines": "lines" in assets,
            "want_substations": "substations" in assets,
            "want_plants": "plants" in assets,
            "want_towers": False,
            "want_poles": False,
            "want_towns": True,
        }
        elements = []
        query = osm.build_query(place, timeout_s=120, **core_flags)
        try:
            data = osm.overpass(query, feedback, cache_dir, f"core grid for {row['segment_id']}",
                                attempts_per_mirror=1, retry_wait_s=0, http_timeout_ms=150000)
            elements.extend(data.get("elements", []))
        except Exception as combined_exc:  # noqa: BLE001
            feedback.pushWarning(tr(
                f"Combined OSM query failed for {row['segment_id']}; splitting by asset type. {combined_exc}"))
            pieces = [
                ("lines", {"want_lines": True, "want_substations": False, "want_plants": False, "want_towns": False}),
                ("substations", {"want_lines": False, "want_substations": True, "want_plants": False, "want_towns": False}),
                ("plants", {"want_lines": False, "want_substations": False, "want_plants": True, "want_towns": False}),
                ("towns", {"want_lines": False, "want_substations": False, "want_plants": False, "want_towns": True}),
            ]
            errors = []
            success = 0
            for name, flags in pieces:
                if name != "towns" and name not in assets:
                    continue
                q = osm.build_query(place, timeout_s=90, want_towers=False, want_poles=False, **flags)
                try:
                    d = osm.overpass(q, feedback, cache_dir, f"{name} for {row['segment_id']}",
                                     attempts_per_mirror=1, retry_wait_s=0, http_timeout_ms=120000)
                    elements.extend(d.get("elements", []))
                    success += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{name}: {exc}")
            if success == 0:
                raise RuntimeError("All split OSM queries failed: " + " | ".join(errors[-4:])) from combined_exc
            if errors:
                feedback.pushWarning(tr(f"Partial OSM result for {row['segment_id']}: " + " | ".join(errors)))

        for value in ("tower", "pole"):
            if (value + "s") not in assets:
                continue
            q = osm.build_power_node_bbox_query(bbox, value, timeout_s=75 if value == "tower" else 60)
            try:
                d = osm.overpass(q, feedback, cache_dir, f"{value}s for {row['segment_id']}",
                                 attempts_per_mirror=1, retry_wait_s=0, mirror_limit=2, http_timeout_ms=90000)
                elements.extend(d.get("elements", []))
            except Exception as exc:  # noqa: BLE001
                feedback.pushWarning(tr(f"Optional {value}s skipped in {row['segment_id']}: {exc}"))

        dedup = {}
        for el in elements:
            if el.get("id") is not None:
                dedup[(el.get("type"), el.get("id"))] = el
        return {"elements": list(dedup.values())}

    def _write_osm_elements(self, gpkg, store, segment_id, elements, area, context):
        area_engine = QgsGeometry.createGeometryEngine(area.constGet())
        area_engine.prepareGeometry()
        centroid = area.centroid().asPoint()
        metric = utm_for(centroid.x(), centroid.y())
        to_m = QgsCoordinateTransform(WGS84, metric, context.transformContext())
        written = 0
        for el in elements:
            kind = osm.classify(el)
            if kind not in ("line", "substation", "plant", "town", "tower", "pole"):
                continue
            coords = osm.element_coords(el)
            if not coords:
                continue
            tags = el.get("tags") or {}
            asset_key = f"osm:{el.get('type','?')}:{el.get('id')}:{kind}"
            if store.asset_seen(asset_key):
                continue
            if kind == "line":
                if len(coords) < 2:
                    continue
                geom = QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in coords])
                if not area_engine.intersects(geom.constGet()):
                    continue
                geom = geom.intersection(area)
                if geom.isEmpty():
                    continue
                # GPKG layer is LineString; write multipart components separately with stable suffixes.
                parts = geom.asGeometryCollection() if geom.isMultipart() else [geom]
                gm = QgsGeometry(geom)
                gm.transform(to_m)
                km = gm.length() / 1000.0
                attrs = {
                    "asset_key": asset_key, "osm_id": int(el["id"]), "osm_type": el.get("type"),
                    "power": tags.get("power"), "voltage_kv": osm.voltage_kv(tags), "voltage": tags.get("voltage"),
                    "name": tags.get("name"), "operator": tags.get("operator"), "circuits": tags.get("circuits"),
                    "cables": tags.get("cables"), "length_km": round(km, 3), "segment_id": segment_id,
                    "source": "OpenStreetMap",
                }
                for idx, part in enumerate(parts):
                    a = dict(attrs)
                    a["asset_key"] = asset_key if len(parts) == 1 else f"{asset_key}:part{idx + 1}"
                    gpkg.add_wkt("lines", part.asWkt(), a)
                store.register_asset(asset_key, kind, segment_id)
                written += 1
                continue

            pt = _element_point(el, coords)
            if pt.isEmpty() or not area_engine.intersects(pt.constGet()):
                continue
            base = {"asset_key": asset_key, "osm_id": int(el["id"]), "osm_type": el.get("type"),
                    "segment_id": segment_id, "source": "OpenStreetMap"}
            if kind == "substation":
                attrs = dict(base, name=tags.get("name"), voltage_kv=osm.voltage_kv(tags),
                             voltage=tags.get("voltage"), operator=tags.get("operator"), kind=tags.get("substation"))
                layer = "substations"
            elif kind == "plant":
                attrs = dict(base, name=tags.get("name"), plant_source=tags.get("plant:source"),
                             output=tags.get("plant:output:electricity"), operator=tags.get("operator"))
                layer = "plants"
            elif kind == "town":
                attrs = dict(base, name=tags.get("name"), place=tags.get("place"))
                layer = "towns"
            else:
                attrs = dict(base, ref=tags.get("ref"), operator=tags.get("operator"))
                layer = "towers" if kind == "tower" else "poles"
            gpkg.add_wkt(layer, pt.asWkt(), attrs)
            store.register_asset(asset_key, kind, segment_id)
            written += 1
        return written

    def _scan_zones_for_segment(self, row, elements, context):
        rect = QgsGeometry.fromRect(QgsRectangle(row["west"], row["south"], row["east"], row["north"]))
        c = rect.centroid().asPoint()
        metric = utm_for(c.x(), c.y())
        to_m = QgsCoordinateTransform(WGS84, metric, context.transformContext())
        rect_m = QgsGeometry(rect)
        rect_m.transform(to_m)
        towns, subs, node_degree = [], [], {}
        for el in elements:
            kind = osm.classify(el)
            coords = osm.element_coords(el)
            if not kind or not coords:
                continue
            tags = el.get("tags") or {}
            if kind == "line" and len(coords) >= 2:
                for end in (coords[0], coords[-1]):
                    key = (round(end[0], 5), round(end[1], 5))
                    deg, volts = node_degree.get(key, (0, set()))
                    volts.add(tags.get("voltage", ""))
                    node_degree[key] = (deg + 1, volts)
            elif kind == "town":
                towns.append((_element_point(el, coords), tags.get("place")))
            elif kind == "substation":
                subs.append(_element_point(el, coords))

        buffers = []
        town_r = self._town_radius * 1000.0

        def add(pt, radius):
            if radius <= 0 or pt is None or pt.isEmpty():
                return
            g = QgsGeometry(pt)
            g.transform(to_m)
            buffers.append(g.buffer(radius, 10))

        for pt, place_kind in towns:
            add(pt, town_r * (2 if place_kind == "city" else 1))
        for (x, y), (degree, volts) in node_degree.items():
            if degree != 2 or len(volts) > 1:
                add(QgsGeometry.fromPointXY(QgsPointXY(x, y)), self._node_radius)
        for pt in subs:
            add(pt, self._known_radius)
        if not buffers:
            return None, metric
        zones = QgsGeometry.unaryUnion(buffers).intersection(rect_m)
        if zones.isEmpty():
            return None, metric
        layer = QgsVectorLayer(f"Polygon?crs={metric.authid()}", f"zones_{row['segment_id']}", "memory")
        parts = zones.asGeometryCollection() if zones.isMultipart() else [zones]
        feats = []
        for part in parts:
            f = QgsFeature()
            f.setGeometry(QgsGeometry(part))
            feats.append(f)
        layer.dataProvider().addFeatures(feats)
        return layer, metric

    def _run_ai_segment(self, row, raw, gpkg, store, parameters, context, feedback):
        zones, _metric = self._scan_zones_for_segment(row, raw.get("elements", []), context)
        if zones is None:
            feedback.pushInfo(tr(f"{row['segment_id']}: no likely substation scan zones; AI segment complete with 0 tiles."))
            return 0, 0
        child_params = {
            "AOI": zones,
            "GSD": self.parameterAsDouble(parameters, self.GSD, context),
            "TILE_SIZE": 750,
            "OVERLAP": 25,
            "SCORE": self.parameterAsDouble(parameters, self.SCORE, context),
            "NMS_IOU": 0.3,
            "MAX_TILES": self.parameterAsInt(parameters, self.AI_MAX_TILES, context),
            "SKIP_BLANK": True,
            "OUTPUT": "TEMPORARY_OUTPUT",
        }
        if self._ai_kind == "local":
            model = self.parameterAsFile(parameters, self.MODEL, context)
            if model:
                child_params["MODEL"] = model
        else:
            child_params.update({
                "MODEL_ID": self.parameterAsString(parameters, self.RF_MODEL, context),
                "VERSION": self.parameterAsInt(parameters, self.RF_VERSION, context),
                "API_KEY": self.parameterAsString(parameters, self.API_KEY, context),
                "TASK": 0,
            })
        child_res = self._child.processAlgorithm(child_params, context, feedback)
        det_layer = QgsProcessingUtils.mapLayerFromString(child_res["OUTPUT"], context)
        if det_layer is None:
            raise RuntimeError("AI detector returned no output layer")
        to_wgs = QgsCoordinateTransform(det_layer.crs(), WGS84, context.transformContext())
        n = 0
        for f in det_layer.getFeatures():
            g = QgsGeometry(f.geometry())
            g.transform(to_wgs)
            if g.isEmpty():
                continue
            digest = hashlib.sha1(g.asWkt(5).encode("utf-8")).hexdigest()[:16]  # noqa: S324
            key = f"ai:{row['segment_id']}:{digest}"
            if store.asset_seen(key):
                continue
            names = det_layer.fields().names()
            vals = dict(zip(names, f.attributes()))
            attrs = {
                "asset_key": key, "segment_id": row["segment_id"], "label": vals.get("label", "substation"),
                "score": vals.get("score"), "model": vals.get("model"), "imagery": vals.get("imagery"),
                "review": vals.get("review", "unreviewed"), "source": "Grid Mapper AI",
            }
            gpkg.add_wkt("ai_substations", g.asWkt(), attrs)
            store.register_asset(key, "ai_substation", row["segment_id"])
            n += 1
        return int(child_res.get("TILES", 0)), n

    @staticmethod
    def _sleep(seconds, feedback):
        remaining = float(seconds)
        while remaining > 0 and not feedback.isCanceled():
            step = min(0.25, remaining)
            time.sleep(step)
            remaining -= step

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        place_name, area = self._resolve_area(parameters, context, feedback)
        segment_km = self.parameterAsDouble(parameters, self.SEGMENT_KM, context)
        self._overlap_km = self.parameterAsDouble(parameters, self.OVERLAP_KM, context)
        self._town_radius = self.parameterAsDouble(parameters, self.TOWN_RADIUS, context)
        self._node_radius = self.parameterAsDouble(parameters, self.NODE_RADIUS, context)
        self._known_radius = self.parameterAsDouble(parameters, self.KNOWN_RADIUS, context)
        workspace = os.path.abspath(self.parameterAsFile(parameters, self.WORKSPACE, context))
        if not workspace:
            raise QgsProcessingException(tr("Choose a persistent workspace folder."))
        os.makedirs(workspace, exist_ok=True)
        raw_dir = os.path.join(workspace, "osm_segments")
        cache_dir = os.path.join(workspace, "http_cache")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        state_path = os.path.join(workspace, "scan_state.sqlite")
        gpkg_path = os.path.join(workspace, "country_grid.gpkg")
        segments_path = os.path.join(workspace, "segments.geojson")
        summary_path = os.path.join(workspace, "summary.json")

        segments = build_segments(area, segment_km)
        if not segments:
            raise QgsProcessingException(tr("The country/AOI produced no scan segments."))
        metadata = {
            "signature": _signature(area, segment_km, self._overlap_km),
            "place_name": place_name,
            "segment_km": segment_km,
            "overlap_km": self._overlap_km,
            "plugin_version": "1.5.0",
        }
        assets = {ASSET_OPTIONS[i] for i in self.parameterAsEnums(parameters, self.ASSETS, context)}
        max_segments = self.parameterAsInt(parameters, self.MAX_SEGMENTS, context)
        max_ai_segments = self.parameterAsInt(parameters, self.MAX_AI_SEGMENTS, context)
        max_attempts = self.parameterAsInt(parameters, self.MAX_ATTEMPTS, context)
        retry_failed = self.parameterAsBool(parameters, self.RETRY_FAILED, context)
        delay = self.parameterAsDouble(parameters, self.DELAY, context)

        with CountryScanStore(state_path) as store, CountryGeoPackage(gpkg_path) as gpkg:
            try:
                store.initialize(metadata, segments)
            except ValueError as exc:
                raise QgsProcessingException(tr(str(exc))) from exc
            interrupted = store.reset_interrupted()
            if interrupted:
                feedback.pushWarning(tr(f"Recovered {interrupted} interrupted segment(s); they will be retried."))
            start = store.summary()
            feedback.pushInfo(tr(
                f"Country scan: {place_name}; {start['total']:,} segments; "
                f"OSM {start['osm_done']:,}/{start['total']:,} complete ({start['osm_progress_pct']:.1f}%)."))

            queue = store.next_segments(max_segments, retry_failed, max_attempts)
            for idx, row in enumerate(queue, start=1):
                if feedback.isCanceled():
                    break
                sid = row["segment_id"]
                feedback.pushInfo(tr(f"OSM segment {idx}/{len(queue)} this run: {sid}"))
                store.mark_running(sid)
                try:
                    raw = self._adaptive_osm_fetch(row, assets, cache_dir, feedback)
                    raw_path = os.path.join(raw_dir, sid + ".json")
                    with open(raw_path, "w", encoding="utf-8") as fh:
                        json.dump(raw, fh)
                    written = self._write_osm_elements(gpkg, store, sid, raw.get("elements", []), area, context)
                    store.mark_done(sid, written)
                    feedback.pushInfo(tr(f"{sid}: complete; {written} new unique grid assets committed."))
                except Exception as exc:  # noqa: BLE001
                    attempts = int(row.get("attempts", 0)) + 1
                    terminal = attempts >= max_attempts
                    store.mark_retry(sid, exc, terminal=terminal)
                    feedback.pushWarning(tr(
                        f"{sid}: {'failed' if terminal else 'deferred for retry'} after attempt {attempts}: {exc}"))
                _write_segments_geojson(segments_path, store.rows(), place_name)
                self._sleep(delay, feedback)

            # Independent AI queue: only OSM-complete segments are eligible. Missing model/imagery
            # leaves the queue untouched, so adding a model later truly resumes AI from 0/N.
            if self._ai_kind != "none":
                if self._ai_skip_reason:
                    feedback.pushWarning(tr(self._ai_skip_reason))
                elif self._child is not None:
                    ai_queue = store.next_ai_segments(max_ai_segments, retry_failed, max_attempts)
                    feedback.pushInfo(tr(f"AI queue: {len(ai_queue)} eligible segment(s) selected for this run."))
                    for idx, row in enumerate(ai_queue, start=1):
                        if feedback.isCanceled():
                            break
                        sid = row["segment_id"]
                        raw_path = os.path.join(raw_dir, sid + ".json")
                        feedback.pushInfo(tr(f"AI segment {idx}/{len(ai_queue)} this run: {sid}"))
                        if not os.path.isfile(raw_path):
                            store.mark_ai_retry(sid, "Raw OSM segment JSON is missing")
                            continue
                        store.mark_ai_running(sid)
                        try:
                            with open(raw_path, encoding="utf-8") as fh:
                                raw = json.load(fh)
                            tiles, dets = self._run_ai_segment(row, raw, gpkg, store, parameters, context, feedback)
                            store.mark_ai_done(sid, tiles, dets)
                            feedback.pushInfo(tr(f"{sid}: AI complete; {tiles} tiles, {dets} new detections."))
                        except Exception as exc:  # noqa: BLE001
                            attempts = int(row.get("attempts", 0)) + 1
                            terminal = attempts >= max_attempts
                            store.mark_ai_retry(sid, exc, terminal=terminal)
                            feedback.pushWarning(tr(
                                f"{sid}: AI {'failed' if terminal else 'deferred for retry'}: {exc}"))
                        _write_segments_geojson(segments_path, store.rows(), place_name)

            summary = store.summary()
            _write_segments_geojson(segments_path, store.rows(), place_name)
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(dict(summary, place_name=place_name, workspace=workspace,
                               gpkg=gpkg_path, state_db=state_path), fh, indent=2)

        feedback.pushInfo(tr(
            f"OSM progress: {summary['osm_done']:,}/{summary['total']:,} "
            f"({summary['osm_progress_pct']:.1f}%). AI progress: {summary['ai_done']:,}/{summary['total']:,} "
            f"({summary['ai_progress_pct']:.1f}%). Re-run this tool with the same workspace to continue."))
        feedback.pushInfo(tr(f"Country GeoPackage: {gpkg_path}"))
        feedback.pushInfo(tr(f"Resume database: {state_path}"))

        feedback.pushInfo(tr(
            "Open country_grid.gpkg and segments.geojson in QGIS to inspect persistent results. "
            "The files are updated in place on every resume run."))

        return {
            "GRID_GPKG": gpkg_path,
            "STATE_DB": state_path,
            "SEGMENTS_GEOJSON": segments_path,
            "WORKSPACE_OUT": workspace,
            "SEGMENTS_TOTAL": summary["total"],
            "OSM_DONE": summary["osm_done"], "OSM_PENDING": summary["osm_pending"],
            "OSM_RETRY": summary["osm_retry"], "OSM_FAILED": summary["osm_failed"],
            "AI_DONE": summary["ai_done"], "AI_PENDING": summary["ai_pending"],
            "AI_RETRY": summary["ai_retry"], "AI_FAILED": summary["ai_failed"],
            "ASSETS": summary["assets"], "AI_DETECTIONS": summary["ai_detections"],
            "AI_TILES": summary["ai_tiles"],
        }
