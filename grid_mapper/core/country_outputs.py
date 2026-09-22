"""GeoPackage output helpers for resumable country scans."""
from __future__ import annotations

import os


def _field(ogr, name, kind, width=0, precision=0):
    mapping = {"string": ogr.OFTString, "int": ogr.OFTInteger, "int64": ogr.OFTInteger64, "real": ogr.OFTReal}
    f = ogr.FieldDefn(name, mapping[kind])
    if width:
        f.SetWidth(int(width))
    if precision:
        f.SetPrecision(int(precision))
    return f


LAYER_SCHEMAS = {
    "lines": ("LineString", [
        ("asset_key", "string", 80, 0), ("osm_id", "int64", 0, 0), ("osm_type", "string", 12, 0),
        ("power", "string", 20, 0), ("voltage_kv", "real", 0, 1), ("voltage", "string", 50, 0),
        ("name", "string", 160, 0), ("operator", "string", 120, 0), ("circuits", "string", 20, 0),
        ("cables", "string", 20, 0), ("length_km", "real", 0, 3), ("segment_id", "string", 32, 0),
        ("source", "string", 30, 0),
    ]),
    "substations": ("Point", [
        ("asset_key", "string", 80, 0), ("osm_id", "int64", 0, 0), ("osm_type", "string", 12, 0),
        ("name", "string", 160, 0), ("voltage_kv", "real", 0, 1), ("voltage", "string", 50, 0),
        ("operator", "string", 120, 0), ("kind", "string", 50, 0), ("segment_id", "string", 32, 0),
        ("source", "string", 30, 0),
    ]),
    "plants": ("Point", [
        ("asset_key", "string", 80, 0), ("osm_id", "int64", 0, 0), ("osm_type", "string", 12, 0),
        ("name", "string", 160, 0), ("plant_source", "string", 60, 0), ("output", "string", 60, 0),
        ("operator", "string", 120, 0), ("segment_id", "string", 32, 0), ("source", "string", 30, 0),
    ]),
    "towns": ("Point", [
        ("asset_key", "string", 80, 0), ("osm_id", "int64", 0, 0), ("osm_type", "string", 12, 0),
        ("name", "string", 160, 0), ("place", "string", 30, 0), ("segment_id", "string", 32, 0),
        ("source", "string", 30, 0),
    ]),
    "towers": ("Point", [
        ("asset_key", "string", 80, 0), ("osm_id", "int64", 0, 0), ("osm_type", "string", 12, 0),
        ("ref", "string", 60, 0), ("operator", "string", 120, 0), ("segment_id", "string", 32, 0),
        ("source", "string", 30, 0),
    ]),
    "poles": ("Point", [
        ("asset_key", "string", 80, 0), ("osm_id", "int64", 0, 0), ("osm_type", "string", 12, 0),
        ("ref", "string", 60, 0), ("operator", "string", 120, 0), ("segment_id", "string", 32, 0),
        ("source", "string", 30, 0),
    ]),
    "ai_substations": ("Polygon", [
        ("asset_key", "string", 100, 0), ("segment_id", "string", 32, 0), ("label", "string", 80, 0),
        ("score", "real", 0, 4), ("model", "string", 160, 0), ("imagery", "string", 160, 0),
        ("review", "string", 30, 0), ("source", "string", 30, 0),
    ]),
}


class CountryGeoPackage:
    def __init__(self, path):
        from osgeo import ogr, osr
        self.ogr = ogr
        self.osr = osr
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        drv = ogr.GetDriverByName("GPKG")
        if os.path.isfile(self.path):
            self.ds = drv.Open(self.path, 1)
        else:
            self.ds = drv.CreateDataSource(self.path)
        if self.ds is None:
            raise RuntimeError(f"Could not open GeoPackage: {self.path}")
        self.srs = osr.SpatialReference()
        self.srs.ImportFromEPSG(4326)
        self._ensure_layers()

    def close(self):
        self.ds = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ensure_layers(self):
        geom_types = {
            "Point": self.ogr.wkbPoint,
            "LineString": self.ogr.wkbLineString,
            "Polygon": self.ogr.wkbPolygon,
        }
        for name, (geom_name, fields) in LAYER_SCHEMAS.items():
            layer = self.ds.GetLayerByName(name)
            if layer is None:
                layer = self.ds.CreateLayer(name, self.srs, geom_types[geom_name])
                if layer is None:
                    raise RuntimeError(f"Could not create GeoPackage layer {name}")
                for fname, kind, width, precision in fields:
                    layer.CreateField(_field(self.ogr, fname, kind, width, precision))

    def add_wkt(self, layer_name, wkt, attrs):
        layer = self.ds.GetLayerByName(layer_name)
        feat = self.ogr.Feature(layer.GetLayerDefn())
        geom = self.ogr.CreateGeometryFromWkt(wkt)
        if geom is None:
            raise ValueError("Invalid WKT geometry")
        feat.SetGeometry(geom)
        for key, value in attrs.items():
            if value is not None and feat.GetFieldIndex(key) >= 0:
                feat.SetField(key, value)
        rc = layer.CreateFeature(feat)
        feat = None
        if rc != 0:
            raise RuntimeError(f"OGR failed writing {layer_name} feature (code {rc})")
        try:
            layer.SyncToDisk()
        except AttributeError:
            pass


def gpkg_layer_uri(path, layer_name):
    return f"{os.path.abspath(path)}|layername={layer_name}"
