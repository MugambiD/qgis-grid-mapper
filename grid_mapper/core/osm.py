"""Place lookup (Nominatim) and existing power-grid data (Overpass / OpenStreetMap)."""
import os

from . import net

NOMINATIM_URL = os.environ.get("GRIDMAPPER_NOMINATIM_URL", "https://nominatim.openstreetmap.org")
OVERPASS_URLS = [u for u in os.environ.get(
    "GRIDMAPPER_OVERPASS_URL",
    "https://overpass-api.de/api/interpreter,https://overpass.kumi.systems/api/interpreter",
).split(",") if u]


def geocode(place):
    """Return dict(name, osm_type, osm_id, bbox=(W,S,E,N), geojson) for the best boundary match."""
    results = net.get_json(f"{NOMINATIM_URL.rstrip('/')}/search", {
        "q": place, "format": "jsonv2", "limit": 5, "polygon_geojson": 1,
        "polygon_threshold": 0.005,
    })
    if not results:
        raise RuntimeError(f"Place '{place}' not found")
    # prefer administrative boundaries (countries, provinces, districts)
    results.sort(key=lambda r: (r.get("osm_type") != "relation",
                                r.get("category", r.get("class")) != "boundary",
                                -float(r.get("importance") or 0)))
    r = results[0]
    s, n, w, e = [float(v) for v in r["boundingbox"]]
    return {"name": r.get("display_name", place), "osm_type": r.get("osm_type"),
            "osm_id": int(r.get("osm_id")), "bbox": (w, s, e, n), "geojson": r.get("geojson")}


def area_id(osm_type, osm_id):
    if osm_type == "relation":
        return 3600000000 + osm_id
    if osm_type == "way":
        return 2400000000 + osm_id
    return None


def build_query(place_info, want_lines=True, want_substations=True, want_plants=True,
                want_towers=True, want_towns=True):
    aid = area_id(place_info["osm_type"], place_info["osm_id"])
    if aid:
        head, flt = f"area(id:{aid})->.a;\n", "(area.a)"
    else:
        w, s, e, n = place_info["bbox"]
        head, flt = "", f"({s},{w},{n},{e})"
    parts = []
    if want_lines:
        parts.append(f'way["power"~"^(line|minor_line|cable)$"]{flt};')
    if want_substations:
        parts.append(f'nwr["power"="substation"]{flt};')
    if want_plants:
        parts.append(f'nwr["power"="plant"]{flt};')
    if want_towers:
        parts.append(f'node["power"~"^(tower|pole)$"]{flt};')
    if want_towns:
        parts.append(f'node["place"~"^(city|town)$"]{flt};')
    return f"[out:json][timeout:900][maxsize:1073741824];\n{head}(\n  " + "\n  ".join(parts) + "\n);\nout geom tags;"


def overpass(query, feedback=None):
    last = None
    for url in OVERPASS_URLS:
        try:
            if feedback:
                feedback.pushInfo(f"Querying OpenStreetMap via {url.split('/')[2]} …")
            return net.post_form_json(url, {"data": query})
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last = exc
            if feedback:
                feedback.pushWarning(f"Overpass mirror failed: {exc}")
    raise RuntimeError(f"All Overpass servers failed: {last}")


def classify(element):
    tags = element.get("tags") or {}
    power, place = tags.get("power"), tags.get("place")
    if power in ("line", "minor_line", "cable"):
        return "line"
    if power == "substation":
        return "substation"
    if power == "plant":
        return "plant"
    if power in ("tower", "pole"):
        return "tower"
    if place in ("city", "town"):
        return "town"
    return None


def element_coords(element):
    """Return list of (lon, lat) for the element (node: 1 point, way: vertices, relation: all member vertices)."""
    t = element.get("type")
    if t == "node":
        return [(element["lon"], element["lat"])]
    if t == "way":
        return [(p["lon"], p["lat"]) for p in element.get("geometry") or []]
    pts = []
    for m in element.get("members") or []:
        if m.get("type") == "node" and "lon" in m:
            pts.append((m["lon"], m["lat"]))
        for p in m.get("geometry") or []:
            pts.append((p["lon"], p["lat"]))
    if not pts and element.get("bounds"):
        b = element["bounds"]
        pts = [((b["minlon"] + b["maxlon"]) / 2, (b["minlat"] + b["maxlat"]) / 2)]
    return pts


def voltage_kv(tags):
    v = str((tags or {}).get("voltage", "")).split(";")[0].strip()
    try:
        return round(float(v) / 1000.0, 1)
    except ValueError:
        return None
