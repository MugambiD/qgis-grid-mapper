"""Place lookup (Nominatim) and existing power-grid data (Overpass / OpenStreetMap)."""
import math
import os

from . import net

NOMINATIM_URL = os.environ.get("GRIDMAPPER_NOMINATIM_URL", "https://nominatim.openstreetmap.org")
OVERPASS_URLS = [u.strip() for u in os.environ.get(
    "GRIDMAPPER_OVERPASS_URL",
    ",".join([
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]),
).split(",") if u.strip()]
RETRY_STATUS = (0, 429, 502, 503, 504)
CACHE_DAYS = 7


def geocode(place):
    """Return dict(name, osm_type, osm_id, bbox=(W,S,E,N), geojson) for the best boundary match."""
    results = net.get_json(f"{NOMINATIM_URL.rstrip('/')}/search", {
        "q": place, "format": "jsonv2", "limit": 5, "polygon_geojson": 1,
        "polygon_threshold": 0.005,
    })
    if not results:
        raise RuntimeError(f"Place '{place}' not found")
    # Prefer administrative boundaries (countries, provinces, districts).
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


def _area_filter(place_info):
    aid = area_id(place_info["osm_type"], place_info["osm_id"])
    if aid:
        return f"area(id:{aid})->.a;\n", "(area.a)"
    w, s, e, n = place_info["bbox"]
    return "", f"({s},{w},{n},{e})"


def build_query(place_info, want_lines=True, want_substations=True, want_plants=True,
                want_towers=False, want_poles=False, want_towns=True, timeout_s=180):
    """Build a compact Overpass QL query for the requested grid asset types.

    Towers and especially distribution poles are intentionally optional because large
    node queries are a common cause of timeouts on public Overpass instances. The
    public-facing Map the grid tool normally fetches those structures separately in
    small bounding-box tiles.
    """
    head, flt = _area_filter(place_info)
    parts = []
    if want_lines:
        parts.append(f'way["power"~"^(line|minor_line|cable)$"]{flt};')
    if want_substations:
        parts.append(f'nwr["power"="substation"]{flt};')
    if want_plants:
        parts.append(f'nwr["power"="plant"]{flt};')
    if want_towers:
        parts.append(f'node["power"="tower"]{flt};')
    if want_poles:
        parts.append(f'node["power"="pole"]{flt};')
    if want_towns:
        parts.append(f'node["place"~"^(city|town)$"]{flt};')
    if not parts:
        return None
    return f"[out:json][timeout:{int(timeout_s)}];\n{head}(\n  " + "\n  ".join(parts) + "\n);\nout geom tags;"


def build_power_node_bbox_query(bbox, power_value, timeout_s=60):
    """Build a lightweight bbox query for ``power=tower`` or ``power=pole`` nodes."""
    if power_value not in ("tower", "pole"):
        raise ValueError("power_value must be 'tower' or 'pole'")
    w, s, e, n = bbox
    return (f'[out:json][timeout:{int(timeout_s)}];\n('
            f'node["power"="{power_value}"]({s},{w},{n},{e});\n);\nout body;')


def bbox_tiles(place_info, cell_deg=0.35):
    """Return deterministic lon/lat tiles covering the place bounding box.

    ``cell_deg`` is deliberately approximate. These tiles are only used to make heavy
    OSM point queries smaller and more reliable; final features are clipped to the true
    place geometry by the processing algorithm.
    """
    if cell_deg <= 0:
        raise ValueError("cell_deg must be positive")
    w, s, e, n = [float(v) for v in place_info["bbox"]]
    nx = max(1, int(math.ceil(max(0.0, e - w) / cell_deg)))
    ny = max(1, int(math.ceil(max(0.0, n - s) / cell_deg)))
    dx = (e - w) / nx if nx else 0.0
    dy = (n - s) / ny if ny else 0.0
    tiles = []
    for row in range(ny):
        south = s + row * dy
        north = n if row == ny - 1 else s + (row + 1) * dy
        for col in range(nx):
            west = w + col * dx
            east = e if col == nx - 1 else w + (col + 1) * dx
            tiles.append((row, col, (west, south, east, north)))
    return tiles


def _cache_path(cache_dir, query):
    import hashlib
    return os.path.join(cache_dir, hashlib.sha256(query.encode()).hexdigest()[:32] + ".json")


def overpass(query, feedback=None, cache_dir=None, label="grid data", attempts_per_mirror=2,
             retry_wait_s=4, mirror_limit=None, http_timeout_ms=240000, allow_stale_cache=True):
    """Run one query against public Overpass mirrors with retries and a short disk cache.

    Optional/large layers can use ``attempts_per_mirror=1`` and ``mirror_limit=2`` so a
    failed layer never holds up the whole grid-mapping operation for several minutes.
    """
    import json
    import time

    from .net import HttpError, post_form_json

    if not query:
        return {"elements": []}
    stale_data = None
    stale_age_days = None
    if cache_dir:
        path = _cache_path(cache_dir, query)
        if os.path.isfile(path):
            try:
                age = time.time() - os.path.getmtime(path)
                with open(path, encoding="utf-8") as fh:
                    cached = json.load(fh)
                if age < CACHE_DAYS * 86400:
                    if feedback:
                        feedback.pushInfo(f"Using cached OpenStreetMap {label} (less than {CACHE_DAYS} days old)")
                    return cached
                stale_data = cached
                stale_age_days = age / 86400.0
            except (OSError, ValueError):
                stale_data = None
    errors = []
    urls = OVERPASS_URLS[:mirror_limit] if mirror_limit else OVERPASS_URLS
    attempts_per_mirror = max(1, int(attempts_per_mirror))
    for url in urls:
        host = url.split("/")[2]
        for attempt in range(attempts_per_mirror):
            if feedback and feedback.isCanceled():
                raise RuntimeError("Cancelled")
            if feedback:
                feedback.pushInfo(f"Downloading OpenStreetMap {label} from {host}"
                                  + (" (retry)" if attempt else "") + " …")
            try:
                data = post_form_json(url, {"data": query}, timeout_ms=http_timeout_ms)
            except HttpError as exc:
                errors.append(f"{host}: {exc}")
                if feedback:
                    feedback.pushWarning(f"{host} failed: {exc}")
                if exc.status in RETRY_STATUS and attempt + 1 < attempts_per_mirror and retry_wait_s > 0:
                    for _ in range(int(retry_wait_s)):
                        if feedback and feedback.isCanceled():
                            raise RuntimeError("Cancelled") from exc
                        time.sleep(1)
                    continue
                break
            remark = data.get("remark", "")
            if "runtime error" in remark.lower():
                errors.append(f"{host}: {remark}")
                if feedback:
                    feedback.pushWarning(f"{host}: {remark}")
                break
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                with open(_cache_path(cache_dir, query), "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
            return data
    if allow_stale_cache and stale_data is not None:
        if feedback:
            feedback.pushWarning(
                f"All live Overpass requests failed for {label}; using stale cached data "
                f"({stale_age_days:.1f} days old).")
        return stale_data
    raise RuntimeError("All Overpass servers tried for this request failed. Public servers may be busy. "
                       "Try again later or use a smaller area. Details: " + " | ".join(errors[-4:]))


def fetch_power_nodes_tiled(place_info, power_value, feedback=None, cache_dir=None,
                            cell_deg=None, max_cells=None):
    """Best-effort tiled fetch for transmission towers or distribution poles.

    Individual tile failures are skipped, successful cells are kept, and duplicate OSM
    node IDs are removed. This makes optional structure layers non-fatal on busy public
    Overpass servers.
    """
    if power_value == "tower":
        cell_deg = 0.40 if cell_deg is None else cell_deg
        max_cells = 200 if max_cells is None else max_cells
        timeout_s = 75
    elif power_value == "pole":
        cell_deg = 0.15 if cell_deg is None else cell_deg
        max_cells = 120 if max_cells is None else max_cells
        timeout_s = 60
    else:
        raise ValueError("power_value must be 'tower' or 'pole'")

    tiles = bbox_tiles(place_info, cell_deg)
    if len(tiles) > int(max_cells):
        raise RuntimeError(
            f"{len(tiles)} OSM cells would be required for {power_value}s (limit {max_cells}). "
            "Use a province/district or smaller AOI for this optional layer."
        )

    dedup = {}
    failed = 0
    total = len(tiles)
    for idx, (row, col, bbox) in enumerate(tiles, start=1):
        if feedback and feedback.isCanceled():
            raise RuntimeError("Cancelled")
        label = f"{power_value}s cell {idx}/{total}"
        query = build_power_node_bbox_query(bbox, power_value, timeout_s)
        try:
            data = overpass(query, feedback, cache_dir, label,
                            attempts_per_mirror=1, retry_wait_s=0,
                            mirror_limit=2, http_timeout_ms=90000)
        except Exception as exc:  # noqa: BLE001 - optional layer; continue with other cells
            failed += 1
            if feedback:
                feedback.pushWarning(f"Skipping OSM {label}: {exc}")
            continue
        for el in data.get("elements", []):
            if el.get("type") == "node" and el.get("id") is not None:
                dedup[(el["type"], el["id"])] = el

    if total and failed == total:
        raise RuntimeError(f"Every OSM {power_value} cell failed")
    if feedback:
        if failed:
            feedback.pushWarning(
                f"OSM {power_value}s are partial: {total - failed}/{total} query cells succeeded."
            )
        else:
            feedback.pushInfo(f"OSM {power_value}s: all {total} query cells succeeded.")
    return {"elements": list(dedup.values()), "cells": total, "failed_cells": failed}


def classify(element):
    tags = element.get("tags") or {}
    power, place = tags.get("power"), tags.get("place")
    if power in ("line", "minor_line", "cable"):
        return "line"
    if power == "substation":
        return "substation"
    if power == "plant":
        return "plant"
    if power == "tower":
        return "tower"
    if power == "pole":
        return "pole"
    if place in ("city", "town"):
        return "town"
    return None


def element_coords(element):
    """Return list of (lon, lat) for node/way/relation elements."""
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
