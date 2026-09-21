"""'Map the grid' - one-click grid mapping for a named place (country, province, district)."""
import math
import urllib.parse

from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFields,
    QgsFillSymbol,
    QgsGeometry,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsPointXY,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingMultiStepFeedback,
    QgsProcessingOutputNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessingUtils,
    QgsRasterLayer,
    QgsRectangle,
    QgsRendererCategory,
    QgsSingleSymbolRenderer,
    QgsSpatialIndex,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QIcon

from ..core import osm
from ..core.compat import (FAST_INSERT, FILE_BEHAVIOR, NUM_DOUBLE, NUM_INT, SRC_POLYGON, WKB_LINESTRING,
                           WKB_POINT, WKB_POLYGON, advanced, make_field)
from .base_detect import ICON, tr
from .detect_roboflow import DetectSubstationsRoboflow
from .detect_tflite import DetectSubstationsTFLite

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

LAYER_OPTIONS = ["substations", "lines", "towers", "plants"]
IMAGERY = [
    ("Esri World Imagery (latest)",
     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 19),
    ("Google Satellite", "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 20),
    ("Bing Aerial", "https://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1", 19),
    ("Use the imagery layer selected below", None, None),
]
DETECTORS = ["tflite", "roboflow", "none"]


def xyz_layer(name, url, zmax):
    uri = f"type=xyz&url={urllib.parse.quote(url, safe=':/')}&zmax={zmax}&zmin=0"
    return QgsRasterLayer(uri, name, "wms")


def utm_for(lon, lat):
    zone = max(1, min(60, int((lon + 180) // 6) + 1))
    return QgsCoordinateReferenceSystem(f"EPSG:{(32600 if lat >= 0 else 32700) + zone}")


def geometry_from_geojson(gj):
    if not gj:
        return None
    import json
    from osgeo import ogr
    g = ogr.CreateGeometryFromJson(json.dumps(gj))
    if g is None:
        return None
    geom = QgsGeometry.fromWkt(g.ExportToWkt())
    return geom if geom and not geom.isEmpty() else None


class _StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Applies a readable default style when an output layer is loaded."""
    instances = []  # keep Python objects alive until QGIS calls them

    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    @classmethod
    def create(cls, kind):
        obj = cls(kind)
        cls.instances.append(obj)
        return obj

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        if not isinstance(layer, QgsVectorLayer):
            return
        k = self.kind
        if k == "lines":
            classes = [("400+ kV", ">= 300", "#7a0177", 1.4), ("220-330 kV", ">= 200", "#c51b8a", 1.1),
                       ("88-132 kV", ">= 60", "#f768a1", 0.8), ("33-66 kV", ">= 20", "#fa9fb5", 0.5),
                       ("< 33 kV", ">= 0", "#999999", 0.3)]
            expr = "CASE " + " ".join(f"WHEN \"voltage_kv\" {cond} THEN '{lbl}'" for lbl, cond, _c, _w in classes) \
                   + " ELSE 'unknown' END"
            cats = [QgsRendererCategory(lbl, QgsLineSymbol.createSimple(
                {"line_color": col, "line_width": str(w)}), lbl) for lbl, _cond, col, w in classes]
            cats.append(QgsRendererCategory("unknown", QgsLineSymbol.createSimple(
                {"line_color": "#666666", "line_width": "0.4", "line_style": "dash"}), "voltage unknown"))
            layer.setRenderer(QgsCategorizedSymbolRenderer(expr, cats))
        elif k == "ai":
            cats = [QgsRendererCategory("new", QgsFillSymbol.createSimple(
                {"color": "255,0,0,60", "outline_color": "#ff0000", "outline_width": "0.8"}), "AI - not in OSM (new)"),
                QgsRendererCategory("in OSM", QgsFillSymbol.createSimple(
                    {"color": "255,200,0,60", "outline_color": "#ffb300", "outline_width": "0.8"}), "AI - confirms OSM")]
            layer.setRenderer(QgsCategorizedSymbolRenderer("osm_match", cats))
        elif k == "zones":
            layer.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(
                {"color": "0,0,0,0", "outline_color": "#00a0ff", "outline_width": "0.4", "outline_style": "dash"})))
        else:
            props = {
                "osm_subs": {"name": "square", "color": "#1f78b4", "size": "2.6", "outline_color": "white"},
                "towers": {"name": "circle", "color": "#555555", "size": "0.9", "outline_style": "no"},
                "plants": {"name": "triangle", "color": "#33a02c", "size": "3.2", "outline_color": "white"},
            }.get(k)
            if props:
                layer.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple(props)))
        layer.triggerRepaint()


class MapGrid(QgsProcessingAlgorithm):
    PLACE = "PLACE"
    AOI = "AOI"
    LAYERS = "LAYERS"
    IMAGERY = "IMAGERY"
    IMAGERY_LAYER = "IMAGERY_LAYER"
    DETECTOR = "DETECTOR"
    MODEL = "MODEL"
    RF_MODEL = "RF_MODEL"
    RF_VERSION = "RF_VERSION"
    API_KEY = "API_KEY"
    TOWN_RADIUS = "TOWN_RADIUS"
    NODE_RADIUS = "NODE_RADIUS"
    KNOWN_RADIUS = "KNOWN_RADIUS"
    GSD = "GSD"
    SCORE = "SCORE"
    MAX_TILES = "MAX_TILES"
    ESTIMATE_ONLY = "ESTIMATE_ONLY"
    OUT_AI = "AI_SUBSTATIONS"
    OUT_OSM_SUBS = "OSM_SUBSTATIONS"
    OUT_LINES = "LINES"
    OUT_TOWERS = "TOWERS"
    OUT_PLANTS = "PLANTS"
    OUT_ZONES = "SCAN_ZONES"

    def name(self):
        return "map_grid"

    def displayName(self):  # noqa: N802
        return tr("Map the grid (type a place)")

    def group(self):
        return tr("Grid mapping")

    def groupId(self):  # noqa: N802
        return "grid"

    def icon(self):
        return QIcon(ICON)

    def createInstance(self):  # noqa: N802
        return MapGrid()

    def shortHelpString(self):  # noqa: N802
        return tr(
            "<p>Type a place - a country (<i>Zambia</i>), province (<i>Copperbelt Province</i>) or "
            "district - pick the grid layers and press <b>Run</b>. The tool:</p><ol>"
            "<li>finds the boundary of the place (OpenStreetMap Nominatim);</li>"
            "<li>downloads the known grid for it from OpenStreetMap - transmission lines with voltage, "
            "substations, towers / poles and power plants;</li>"
            "<li>builds <b>scan zones</b> where substations are likely: around cities and towns, at line "
            "ends and junctions, and around known substations;</li>"
            "<li>scans those zones on the latest satellite imagery with your deep-learning model "
            "(TFLite or Roboflow) and flags every detected substation as <i>new</i> or <i>in OSM</i>.</li></ol>"
            "<p>Run first with <b>Estimate only</b> ticked to see how many image tiles the scan needs "
            "(a country can need several thousand; roughly 1 s per tile). Lower the zone radii or pick a "
            "province to go faster.</p>"
            "<p>Lines, towers and plants come from OpenStreetMap; the AI model finds substations. "
            "OSM coverage varies, so treat all layers as a starting point for field / utility "
            "verification. Respect the imagery provider's terms of use.</p>")

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterString(self.PLACE, tr("Place to map (country, province, district)"),
                                                       "Zambia"))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.AOI, tr("…or use my own area polygon instead (optional)"), [SRC_POLYGON], optional=True))
        self.addParameter(QgsProcessingParameterEnum(
            self.LAYERS, tr("Grid layers to map"),
            [tr("Substations (AI scan + OSM)"), tr("Transmission / distribution lines"),
             tr("Towers and poles"), tr("Power plants")],
            allowMultiple=True, defaultValue=[0, 1, 2, 3]))
        self.addParameter(QgsProcessingParameterEnum(
            self.IMAGERY, tr("Satellite imagery"), [i[0] for i in IMAGERY], defaultValue=0))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.IMAGERY_LAYER, tr("Imagery layer (only if 'Use the imagery layer selected below')"),
            optional=True))
        self.addParameter(QgsProcessingParameterEnum(
            self.DETECTOR, tr("Substation detector"),
            [tr("TFLite model (offline, custom_model_lite.zip)"), tr("Roboflow hosted model (API key)"),
             tr("No AI scan - OpenStreetMap data only")], defaultValue=0))
        self.addParameter(QgsProcessingParameterFile(
            self.MODEL, tr("TFLite model (custom_model_lite.zip or detect.tflite)"),
            behavior=FILE_BEHAVIOR, fileFilter="TFLite model (*.zip *.tflite)", optional=True))
        self.addParameter(QgsProcessingParameterString(self.API_KEY, tr("Roboflow API key"), "", optional=True))
        self.addParameter(QgsProcessingParameterBoolean(
            self.ESTIMATE_ONLY, tr("Estimate only (download OSM data and count tiles, no AI scan)"), False))
        self.addParameter(advanced(QgsProcessingParameterString(self.RF_MODEL, tr("Roboflow model ID"), "ss-2")))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.RF_VERSION, tr("Roboflow model version"), NUM_INT, 1, minValue=1)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.TOWN_RADIUS, tr("Scan radius around towns (km, cities x2)"), NUM_DOUBLE, 1.0, minValue=0)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.NODE_RADIUS, tr("Scan radius around line ends / junctions (m)"), NUM_DOUBLE, 400, minValue=0)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.KNOWN_RADIUS, tr("Scan radius around known OSM substations (m)"), NUM_DOUBLE, 250, minValue=0)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.GSD, tr("Ground resolution (m/px)"), NUM_DOUBLE, 0.6, minValue=0.1, maxValue=5)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.SCORE, tr("Minimum AI confidence"), NUM_DOUBLE, 0.5, minValue=0, maxValue=1)))
        self.addParameter(advanced(QgsProcessingParameterNumber(
            self.MAX_TILES, tr("Safety limit: maximum image tiles"), NUM_INT, 10000, minValue=1)))

        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_AI, tr("Substations detected by AI"), optional=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_OSM_SUBS, tr("Substations (OpenStreetMap)"), optional=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_LINES, tr("Power lines (OpenStreetMap)"), optional=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_TOWERS, tr("Towers and poles (OpenStreetMap)"), optional=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_PLANTS, tr("Power plants (OpenStreetMap)"), optional=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_ZONES, tr("AI scan zones"), optional=True))
        for key in ("N_AI", "N_AI_NEW", "N_OSM_SUBSTATIONS", "N_LINES", "LINE_KM", "N_TOWERS", "N_PLANTS", "TILES"):
            self.addOutput(QgsProcessingOutputNumber(key, key))

    # ------------------------------------------------------------------
    def prepareAlgorithm(self, parameters, context, feedback):  # noqa: N802
        """Main thread: build the imagery layer and the detector instance."""
        self._detector_kind = DETECTORS[self.parameterAsEnum(parameters, self.DETECTOR, context)]
        choice = self.parameterAsEnum(parameters, self.IMAGERY, context)
        name, url, zmax = IMAGERY[choice]
        if url is None:
            lyr = self.parameterAsRasterLayer(parameters, self.IMAGERY_LAYER, context)
            if lyr is None:
                raise QgsProcessingException(tr("Select an imagery layer, or choose a satellite basemap."))
            self._imagery = lyr.clone()
            self._imagery_name = lyr.name()
        else:
            self._imagery = xyz_layer(name, url, zmax)
            self._imagery_name = name
        if not self._imagery.isValid():
            raise QgsProcessingException(tr(f"Could not open imagery '{self._imagery_name}'"))
        self._child = None
        if self._detector_kind == "tflite":
            self._child = DetectSubstationsTFLite().create()
        elif self._detector_kind == "roboflow":
            self._child = DetectSubstationsRoboflow().create()
        if self._child is not None:
            self._child._layer = self._imagery
            self._child._layer_crs = self._imagery.crs()
            self._child._layer_extent = self._imagery.extent()
            self._child._layer_name = self._imagery_name
            self._child._transform_context = context.transformContext()
        return True

    def _sink(self, parameters, key, context, fields, wkb, crs):
        if parameters.get(key) in (None, ""):
            return None, None
        return self.parameterAsSink(parameters, key, context, fields, wkb, crs)

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        layers = {LAYER_OPTIONS[i] for i in self.parameterAsEnums(parameters, self.LAYERS, context)}
        estimate_only = self.parameterAsBool(parameters, self.ESTIMATE_ONLY, context)
        want_ai = "substations" in layers and self._child is not None and not estimate_only
        multi = QgsProcessingMultiStepFeedback(3, feedback)
        tctx = context.transformContext()

        # 1 -- the area ----------------------------------------------------
        aoi_src = self.parameterAsSource(parameters, self.AOI, context)
        if aoi_src is not None:
            to_wgs = QgsCoordinateTransform(aoi_src.sourceCrs(), WGS84, tctx)
            geoms = []
            for f in aoi_src.getFeatures():
                g = QgsGeometry(f.geometry())
                g.transform(to_wgs)
                geoms.append(g)
            area = QgsGeometry.unaryUnion(geoms)
            bb = area.boundingBox()
            place = {"name": "custom area", "osm_type": None, "osm_id": 0,
                     "bbox": (bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum())}
        else:
            text = self.parameterAsString(parameters, self.PLACE, context).strip()
            if not text:
                raise QgsProcessingException(tr("Type a place name or give an area polygon."))
            feedback.pushInfo(tr(f"Looking up '{text}' …"))
            try:
                place = osm.geocode(text)
            except Exception as exc:  # noqa: BLE001
                raise QgsProcessingException(tr(f"Place lookup failed: {exc}")) from exc
            area = geometry_from_geojson(place.get("geojson"))
            if area is None or area.type() != QgsGeometry.fromRect(QgsRectangle(0, 0, 1, 1)).type():
                w, s, e, n = place["bbox"]
                area = QgsGeometry.fromRect(QgsRectangle(w, s, e, n))
        area = area.makeValid()
        feedback.pushInfo(tr(f"Area: {place['name']}"))
        c = area.centroid().asPoint()
        metric = utm_for(c.x(), c.y())
        to_m = QgsCoordinateTransform(WGS84, metric, tctx)
        area_m = QgsGeometry(area)
        area_m.transform(to_m)
        feedback.pushInfo(tr(f"Area size: {area_m.area() / 1e6:,.0f} km² (metric CRS {metric.authid()})"))
        if feedback.isCanceled():
            return {}

        # 2 -- OpenStreetMap grid data ---------------------------------------
        multi.setCurrentStep(0)
        need_lines = "lines" in layers or want_ai or estimate_only
        query = osm.build_query(place, want_lines=need_lines,
                                want_substations="substations" in layers,
                                want_plants="plants" in layers, want_towers="towers" in layers,
                                want_towns=want_ai or estimate_only)
        try:
            data = osm.overpass(query, feedback)
        except Exception as exc:  # noqa: BLE001
            raise QgsProcessingException(tr(f"OpenStreetMap download failed: {exc}")) from exc
        elements = data.get("elements", [])
        feedback.pushInfo(tr(f"{len(elements):,} OpenStreetMap features downloaded"))
        clip_needed = place.get("osm_type") is None  # bbox query -> clip to the polygon
        self._area_geom = area  # the engine keeps a pointer to it
        engine = QgsGeometry.createGeometryEngine(self._area_geom.constGet())
        engine.prepareGeometry()

        def inside(geom):
            return (not clip_needed) or engine.intersects(geom.constGet())

        line_f = QgsFields()
        for n, k, ln in (("osm_id", "long", 0), ("power", "string", 16), ("voltage_kv", "double", 8),
                         ("voltage", "string", 40), ("name", "string", 120), ("operator", "string", 80),
                         ("circuits", "string", 10), ("cables", "string", 10), ("length_km", "double", 10),
                         ("source", "string", 20)):
            line_f.append(make_field(n, k, ln, 2 if k == "double" else 0))
        sub_f = QgsFields()
        for n, k, ln in (("osm_id", "long", 0), ("osm_type", "string", 10), ("name", "string", 120),
                         ("voltage_kv", "double", 8), ("voltage", "string", 40), ("operator", "string", 80),
                         ("kind", "string", 40), ("source", "string", 20)):
            sub_f.append(make_field(n, k, ln, 1 if k == "double" else 0))
        plant_f = QgsFields()
        for n, k, ln in (("osm_id", "long", 0), ("name", "string", 120), ("plant_source", "string", 40),
                         ("output", "string", 40), ("operator", "string", 80), ("source", "string", 20)):
            plant_f.append(make_field(n, k, ln))
        tower_f = QgsFields()
        for n, k, ln in (("osm_id", "long", 0), ("power", "string", 10), ("ref", "string", 40),
                         ("operator", "string", 80), ("source", "string", 20)):
            tower_f.append(make_field(n, k, ln))

        s_lines, d_lines = self._sink(parameters, self.OUT_LINES, context, line_f, WKB_LINESTRING, WGS84) \
            if "lines" in layers else (None, None)
        s_subs, d_subs = self._sink(parameters, self.OUT_OSM_SUBS, context, sub_f, WKB_POINT, WGS84) \
            if "substations" in layers else (None, None)
        s_plants, d_plants = self._sink(parameters, self.OUT_PLANTS, context, plant_f, WKB_POINT, WGS84) \
            if "plants" in layers else (None, None)
        s_towers, d_towers = self._sink(parameters, self.OUT_TOWERS, context, tower_f, WKB_POINT, WGS84) \
            if "towers" in layers else (None, None)

        n_lines = n_subs = n_plants = n_towers = 0
        line_km = 0.0
        towns, osm_sub_pts = [], []
        node_degree = {}
        for i, el in enumerate(elements):
            if feedback.isCanceled():
                return {}
            if i % 5000 == 0:
                multi.setProgress(100.0 * i / max(1, len(elements)))
            kind = osm.classify(el)
            tags = el.get("tags") or {}
            coords = osm.element_coords(el)
            if not kind or not coords:
                continue
            if kind == "line":
                if len(coords) < 2:
                    continue
                geom = QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in coords])
                if not inside(geom):
                    continue
                for end in (coords[0], coords[-1]):
                    key = (round(end[0], 5), round(end[1], 5))
                    deg, volts = node_degree.get(key, (0, set()))
                    volts.add(tags.get("voltage", ""))
                    node_degree[key] = (deg + 1, volts)
                gm = QgsGeometry(geom)
                gm.transform(to_m)
                km = gm.length() / 1000.0
                if s_lines is not None:
                    f = QgsFeature(line_f)
                    f.setGeometry(geom)
                    f.setAttributes([el["id"], tags.get("power"), osm.voltage_kv(tags), tags.get("voltage"),
                                     tags.get("name"), tags.get("operator"), tags.get("circuits"),
                                     tags.get("cables"), round(km, 2), "OpenStreetMap"])
                    s_lines.addFeature(f, FAST_INSERT)
                    n_lines += 1
                    line_km += km
                continue
            if kind == "town":
                pt = QgsGeometry.fromPointXY(QgsPointXY(*coords[0]))
                if inside(pt):
                    towns.append((pt, tags.get("place")))
                continue
            if el.get("type") == "way" and len(coords) >= 4 and coords[0] == coords[-1]:
                pt = QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in coords]]).pointOnSurface()
            elif len(coords) == 1:
                pt = QgsGeometry.fromPointXY(QgsPointXY(*coords[0]))
            else:
                pt = QgsGeometry.fromMultiPointXY([QgsPointXY(x, y) for x, y in coords]).centroid()
            if not inside(pt):
                continue
            if kind == "substation":
                osm_sub_pts.append((pt, tags.get("name")))
                if s_subs is not None:
                    f = QgsFeature(sub_f)
                    f.setGeometry(pt)
                    f.setAttributes([el["id"], el["type"], tags.get("name"), osm.voltage_kv(tags),
                                     tags.get("voltage"), tags.get("operator"), tags.get("substation"),
                                     "OpenStreetMap"])
                    s_subs.addFeature(f, FAST_INSERT)
                    n_subs += 1
            elif kind == "plant" and s_plants is not None:
                f = QgsFeature(plant_f)
                f.setGeometry(pt)
                f.setAttributes([el["id"], tags.get("name"), tags.get("plant:source"),
                                 tags.get("plant:output:electricity"), tags.get("operator"), "OpenStreetMap"])
                s_plants.addFeature(f, FAST_INSERT)
                n_plants += 1
            elif kind == "tower" and s_towers is not None:
                f = QgsFeature(tower_f)
                f.setGeometry(pt)
                f.setAttributes([el["id"], tags.get("power"), tags.get("ref"), tags.get("operator"),
                                 "OpenStreetMap"])
                s_towers.addFeature(f, FAST_INSERT)
                n_towers += 1
        feedback.pushInfo(tr(
            f"OSM grid: {n_lines:,} line sections ({line_km:,.0f} km), {len(osm_sub_pts):,} substations, "
            f"{n_plants:,} plants, {n_towers:,} towers/poles, {len(towns):,} towns"))

        results = self._results = {"N_LINES": n_lines, "LINE_KM": round(line_km, 1), "N_OSM_SUBSTATIONS": len(osm_sub_pts),
                   "N_TOWERS": n_towers, "N_PLANTS": n_plants, "N_AI": 0, "N_AI_NEW": 0, "TILES": 0}
        for key, dest in ((self.OUT_LINES, d_lines), (self.OUT_OSM_SUBS, d_subs), (self.OUT_PLANTS, d_plants),
                          (self.OUT_TOWERS, d_towers)):
            if dest:
                results[key] = dest

        if "substations" not in layers:
            return results

        # 3 -- scan zones ---------------------------------------------------
        multi.setCurrentStep(1)
        town_r = self.parameterAsDouble(parameters, self.TOWN_RADIUS, context) * 1000.0
        node_r = self.parameterAsDouble(parameters, self.NODE_RADIUS, context)
        known_r = self.parameterAsDouble(parameters, self.KNOWN_RADIUS, context)
        buffers = []

        def add_buffer(pt_wgs, radius):
            if radius <= 0:
                return
            g = QgsGeometry(pt_wgs)
            g.transform(to_m)
            buffers.append(g.buffer(radius, 12))

        for pt, kind in towns:
            add_buffer(pt, town_r * (2 if kind == "city" else 1))
        # substations sit where lines end, several lines meet, or the voltage changes
        nodes = [k for k, (d, volts) in node_degree.items() if d != 2 or len(volts) > 1]
        for x, y in nodes:
            add_buffer(QgsGeometry.fromPointXY(QgsPointXY(x, y)), node_r)
        for pt, _n in osm_sub_pts:
            add_buffer(pt, known_r)
        feedback.pushInfo(tr(f"Scan zones from {len(towns)} towns, {len(nodes)} line ends/junctions/voltage changes, "
                             f"{len(osm_sub_pts)} known substations"))
        if not buffers:
            feedback.pushWarning(tr("No towns, lines or substations found to build scan zones."))
            return results
        zones_m = QgsGeometry.unaryUnion(buffers).intersection(area_m)
        zone_km2 = zones_m.area() / 1e6
        gsd = self.parameterAsDouble(parameters, self.GSD, context)
        tile_m = 750 * gsd
        est = int(math.ceil(zone_km2 * 1e6 / (tile_m * 0.75) ** 2 * 1.15)) + len(buffers) // 4
        feedback.pushInfo(tr(f"Scan zones cover {zone_km2:,.0f} km² ≈ {est:,} image tiles "
                             f"(≈ {est / 60:,.0f} min at ~1 s per tile)"))

        zones_layer = QgsVectorLayer(f"Polygon?crs={metric.authid()}", "zones", "memory")
        zf = QgsFeature()
        zf.setGeometry(zones_m)
        zones_layer.dataProvider().addFeatures([zf])
        s_z, d_z = self._sink(parameters, self.OUT_ZONES, context, QgsFields(), WKB_POLYGON, WGS84)
        if s_z is not None:
            to_wgs_m = QgsCoordinateTransform(metric, WGS84, tctx)
            parts = zones_m.asGeometryCollection() if zones_m.isMultipart() else [zones_m]
            for part in parts:
                g = QgsGeometry(part)
                g.transform(to_wgs_m)
                f = QgsFeature()
                f.setGeometry(g)
                s_z.addFeature(f, FAST_INSERT)
            results[self.OUT_ZONES] = d_z

        if not want_ai:
            if estimate_only:
                feedback.pushInfo(tr("Estimate only - untick 'Estimate only' to run the AI scan."))
            elif self._child is None:
                feedback.pushInfo(tr("No AI detector selected - OpenStreetMap layers only."))
            results["TILES"] = est
            return results
        max_tiles = self.parameterAsInt(parameters, self.MAX_TILES, context)
        if est > max_tiles * 1.2:
            raise QgsProcessingException(tr(
                f"The AI scan needs about {est:,} tiles, above the safety limit of {max_tiles:,}. "
                "Map a province or district, reduce the zone radii (Advanced parameters) or raise the limit. "
                "The OpenStreetMap layers above were still produced."))

        # 4 -- AI scan ------------------------------------------------------
        multi.setCurrentStep(2)
        child_params = {
            "AOI": zones_layer, "GSD": gsd, "TILE_SIZE": 750, "OVERLAP": 25,
            "SCORE": self.parameterAsDouble(parameters, self.SCORE, context), "NMS_IOU": 0.3,
            "MAX_TILES": max_tiles, "SKIP_BLANK": True, "OUTPUT": "TEMPORARY_OUTPUT",
        }
        if self._detector_kind == "tflite":
            model = self.parameterAsFile(parameters, self.MODEL, context)
            if not model:
                raise QgsProcessingException(tr("Select the TFLite model (custom_model_lite.zip), or choose "
                                                 "'No AI scan'."))
            child_params["MODEL"] = model
        else:
            child_params.update({"MODEL_ID": self.parameterAsString(parameters, self.RF_MODEL, context),
                                 "VERSION": self.parameterAsInt(parameters, self.RF_VERSION, context),
                                 "API_KEY": self.parameterAsString(parameters, self.API_KEY, context),
                                 "TASK": 0})
        child_res = self._child.processAlgorithm(child_params, context, multi)
        det_layer = QgsProcessingUtils.mapLayerFromString(child_res["OUTPUT"], context)
        results["TILES"] = child_res.get("TILES", 0)

        # compare with OSM substations
        idx, sub_geoms = QgsSpatialIndex(), {}
        for i, (pt, nm) in enumerate(osm_sub_pts):
            g = QgsGeometry(pt)
            g.transform(to_m)
            f = QgsFeature(i)
            f.setGeometry(g)
            idx.addFeature(f)
            sub_geoms[i] = (g, nm)
        ai_fields = QgsFields(det_layer.fields())
        ai_fields.append(make_field("osm_match", "string", 10))
        ai_fields.append(make_field("osm_name", "string", 120))
        ai_fields.append(make_field("osm_dist_m", "double", 10, 1))
        s_ai, d_ai = self._sink(parameters, self.OUT_AI, context, ai_fields, WKB_POLYGON, WGS84)
        det_to_m = QgsCoordinateTransform(det_layer.crs(), metric, tctx)
        det_to_wgs = QgsCoordinateTransform(det_layer.crs(), WGS84, tctx)
        n_ai = n_new = 0
        match_r = max(known_r, 250.0)
        for f in det_layer.getFeatures():
            gm = QgsGeometry(f.geometry())
            gm.transform(det_to_m)
            best, best_d = None, None
            for i in idx.intersects(gm.boundingBox().buffered(match_r)):
                d = gm.distance(sub_geoms[i][0])
                if d <= match_r and (best_d is None or d < best_d):
                    best, best_d = i, d
            status = "in OSM" if best is not None else "new"
            n_ai += 1
            n_new += status == "new"
            if s_ai is not None:
                g = QgsGeometry(f.geometry())
                g.transform(det_to_wgs)
                nf = QgsFeature(ai_fields)
                nf.setGeometry(g)
                nf.setAttributes(f.attributes() + [status, sub_geoms[best][1] if best is not None else None,
                                                   round(best_d, 1) if best_d is not None else None])
                s_ai.addFeature(nf, FAST_INSERT)
        feedback.pushInfo(tr(f"AI found {n_ai} substations: {n_new} not in OpenStreetMap, "
                             f"{n_ai - n_new} confirming OSM."))
        results.update({"N_AI": n_ai, "N_AI_NEW": n_new})
        if d_ai:
            results[self.OUT_AI] = d_ai
        self._results = results
        return results

    def postProcessAlgorithm(self, context, feedback):  # noqa: N802
        styles = {self.OUT_LINES: "lines", self.OUT_OSM_SUBS: "osm_subs", self.OUT_TOWERS: "towers",
                  self.OUT_PLANTS: "plants", self.OUT_AI: "ai", self.OUT_ZONES: "zones"}
        for layer_id in list(context.layersToLoadOnCompletion().keys()):
            details = context.layerToLoadOnCompletionDetails(layer_id)
            kind = styles.get(details.outputName)
            if kind:
                details.setPostProcessor(_StylePostProcessor.create(kind))
        return getattr(self, "_results", {})
