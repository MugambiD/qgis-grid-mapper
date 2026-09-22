"""Versioned model registry with conservative promotion gates.

The registry keeps candidate models separate from the active model. A candidate
is promoted only when its exported metadata contains a usable validation metric
and the configured gates pass. This avoids silently replacing a good model with
an overfit one.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone

REGISTRY_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_registry_dir(qgis_settings_dir):
    return os.path.join(os.path.abspath(qgis_settings_dir), "grid_mapper", "model_registry")


def _safe(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "model"


def read_model_metadata(model_path):
    """Read Grid Mapper model metadata from sidecar JSON or a model ZIP."""
    model_path = os.path.abspath(model_path)
    if zipfile.is_zipfile(model_path):
        try:
            with zipfile.ZipFile(model_path) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".json") and not n.endswith("/")]
                names.sort(key=lambda n: ("model" not in os.path.basename(n).lower(), len(n)))
                if names:
                    return json.loads(zf.read(names[0]).decode("utf-8"))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            return {}
    stem = os.path.splitext(model_path)[0]
    for p in (stem + ".json", os.path.join(os.path.dirname(model_path), "model.json")):
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                pass
    return {}


def _find_metric(metrics, preferred=None):
    if not isinstance(metrics, dict):
        return None, None
    lower = {str(k).lower(): (k, v) for k, v in metrics.items()}
    order = []
    if preferred:
        order.append(str(preferred).lower())
    order += ["f1", "f1_score", "map", "map50", "ap50", "ap", "coco_map"]
    seen = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        if name in lower:
            key, val = lower[name]
            try:
                return str(key), float(val)
            except (TypeError, ValueError):
                continue
    return None, None


def metric_from_metadata(meta):
    if not isinstance(meta, dict):
        return None, None
    preferred = meta.get("primary_metric")
    return _find_metric(meta.get("metrics") or {}, preferred)


def _geo_metric_map(meta, preferred=None):
    result = {}
    for geo, metrics in (meta.get("geography_metrics") or {}).items() if isinstance(meta, dict) else []:
        _name, value = _find_metric(metrics, preferred)
        if value is not None:
            result[str(geo)] = value
    return result


def promotion_decision(current_meta, candidate_meta, min_gain=0.005, max_geo_regression=0.03,
                       min_first_metric=0.0):
    """Return a structured candidate promotion decision."""
    cand_name, cand = metric_from_metadata(candidate_meta)
    if cand is None:
        return {"passed": False, "reason": "candidate has no recognised validation metric",
                "candidate_metric": None, "current_metric": None, "metric_name": None,
                "geo_regressions": {}, "missing_geographies": []}
    cur_name, cur = metric_from_metadata(current_meta or {})
    metric_name = cand_name or cur_name or "metric"
    if cur is not None and cur_name and cand_name and str(cur_name).lower() != str(cand_name).lower():
        return {"passed": False,
                "reason": f"candidate metric {cand_name} is not comparable to active metric {cur_name}",
                "candidate_metric": cand, "current_metric": cur, "metric_name": metric_name,
                "geo_regressions": {}, "missing_geographies": []}
    if cur is None:
        passed = cand >= float(min_first_metric)
        return {"passed": passed,
                "reason": "first scored model" if passed else "candidate below first-model metric floor",
                "candidate_metric": cand, "current_metric": None, "metric_name": metric_name,
                "geo_regressions": {}, "missing_geographies": []}

    gain = cand - cur
    if gain < float(min_gain):
        return {"passed": False,
                "reason": f"overall {metric_name} gain {gain:+.4f} is below required {float(min_gain):+.4f}",
                "candidate_metric": cand, "current_metric": cur, "metric_name": metric_name,
                "geo_regressions": {}, "missing_geographies": []}

    preferred = candidate_meta.get("primary_metric") or current_meta.get("primary_metric") or metric_name
    old_geo = _geo_metric_map(current_meta, preferred)
    new_geo = _geo_metric_map(candidate_meta, preferred)
    missing = sorted(set(old_geo) - set(new_geo))
    if missing:
        return {"passed": False,
                "reason": "candidate is missing one or more geography validation groups from the active model",
                "candidate_metric": cand, "current_metric": cur, "metric_name": metric_name,
                "geo_regressions": {}, "missing_geographies": missing}
    regressions = {}
    for geo in sorted(set(old_geo) & set(new_geo)):
        drop = old_geo[geo] - new_geo[geo]
        if drop > float(max_geo_regression):
            regressions[geo] = {"current": old_geo[geo], "candidate": new_geo[geo], "drop": drop}
    if regressions:
        return {"passed": False, "reason": "one or more geography validation sets regressed beyond the gate",
                "candidate_metric": cand, "current_metric": cur, "metric_name": metric_name,
                "geo_regressions": regressions, "missing_geographies": []}
    return {"passed": True, "reason": "candidate passed overall and geography regression gates",
            "candidate_metric": cand, "current_metric": cur, "metric_name": metric_name,
            "geo_regressions": {}, "missing_geographies": []}


def _registry_path(root):
    return os.path.join(root, "registry.json")


def load_registry(root):
    path = _registry_path(root)
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                obj = json.load(fh)
            if isinstance(obj, dict):
                obj.setdefault("schema_version", REGISTRY_VERSION)
                obj.setdefault("models", [])
                obj.setdefault("active_model_id", None)
                return obj
        except (OSError, ValueError):
            pass
    return {"schema_version": REGISTRY_VERSION, "active_model_id": None, "models": []}


def save_registry(root, registry):
    os.makedirs(root, exist_ok=True)
    path = _registry_path(root)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def _active_entry(registry):
    aid = registry.get("active_model_id")
    return next((m for m in registry.get("models", []) if m.get("model_id") == aid), None)


def active_model_path(root):
    reg = load_registry(root)
    entry = _active_entry(reg)
    if not entry:
        return None
    p = entry.get("stored_path")
    return p if p and os.path.isfile(p) else None


def register_candidate(root, candidate_path, model_id=None, min_gain=0.005,
                       max_geo_regression=0.03, promote_if_passed=True, min_first_metric=0.0):
    os.makedirs(root, exist_ok=True)
    candidate_path = os.path.abspath(candidate_path)
    if not os.path.isfile(candidate_path):
        raise FileNotFoundError(candidate_path)
    meta = read_model_metadata(candidate_path)
    derived = model_id or meta.get("model_id") or meta.get("version") or os.path.splitext(os.path.basename(candidate_path))[0]
    model_id = _safe(derived)
    reg = load_registry(root)
    current = _active_entry(reg)
    current_meta = current.get("metadata", {}) if current else {}
    decision = promotion_decision(current_meta, meta, min_gain, max_geo_regression, min_first_metric)

    model_dir = os.path.join(root, "models", model_id)
    os.makedirs(model_dir, exist_ok=True)
    dst = os.path.join(model_dir, os.path.basename(candidate_path))
    shutil.copy2(candidate_path, dst)
    stem = os.path.splitext(candidate_path)[0]
    side = stem + ".json"
    if os.path.isfile(side):
        shutil.copy2(side, os.path.join(model_dir, os.path.basename(side)))

    entry = {
        "model_id": model_id,
        "registered_utc": utc_now(),
        "source_path": candidate_path,
        "stored_path": dst,
        "metadata": meta,
        "promotion_decision": decision,
        "active": False,
    }
    reg["models"] = [m for m in reg.get("models", []) if m.get("model_id") != model_id]
    reg["models"].append(entry)
    promoted = bool(decision.get("passed") and promote_if_passed)
    if promoted:
        reg["active_model_id"] = model_id
    for m in reg["models"]:
        m["active"] = m.get("model_id") == reg.get("active_model_id")
    save_registry(root, reg)
    return entry, promoted, reg
