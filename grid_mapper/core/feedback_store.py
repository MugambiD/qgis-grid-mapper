"""Persistent human-feedback dataset for Grid Mapper continual learning.

This module deliberately has no QGIS dependency so dataset bookkeeping can be
unit-tested outside QGIS and reused by the Colab training workflow.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone

SCHEMA_VERSION = 1
MANIFEST_NAME = "feedback.jsonl"

_POSITIVE = {"tp", "true positive", "true_positive", "positive", "confirmed", "confirm", "yes", "correct"}
_NEGATIVE = {"fp", "false positive", "false_positive", "negative", "rejected", "reject", "no"}
_CORRECTED = {"corrected", "edited", "corrected box", "corrected_box", "corrected geometry", "corrected_geometry"}
_MISSED = {"missed", "fn", "false negative", "false_negative"}
_UNCERTAIN = {"uncertain", "unsure", "maybe", "needs review", "needs_review", "review"}
_EMPTY = {"", "unreviewed", "none", "null", "na", "n/a"}


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_review(value):
    """Map common QA values onto the stable feedback vocabulary."""
    text = str(value or "").strip().lower()
    if text in _POSITIVE:
        return "positive"
    if text in _NEGATIVE:
        return "negative"
    if text in _CORRECTED:
        return "corrected"
    if text in _MISSED:
        return "missed"
    if text in _UNCERTAIN:
        return "uncertain"
    if text in _EMPTY:
        return "unreviewed"
    return "unreviewed"


def sample_id_for(*parts):
    raw = "\x1f".join(str(p or "") for p in parts).encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()[:20]


def stable_split(key, valid_pct=10.0, test_pct=10.0):
    """Assign a sample to a fixed split without reshuffling older samples.

    The split is hash-based, so adding new feedback never changes the split of
    previously captured examples. This is important for honest model promotion.
    """
    valid_pct, test_pct = float(valid_pct), float(test_pct)
    if valid_pct < 0 or test_pct < 0 or valid_pct + test_pct >= 100:
        raise ValueError("Validation/test percentages must be >= 0 and sum to less than 100")
    bucket = int(hashlib.sha256(str(key).encode("utf-8", "replace")).hexdigest()[:8], 16) % 10000 / 100.0
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + valid_pct:
        return "valid"
    return "train"


def active_learning_priority(score, review="unreviewed", threshold=0.5):
    """Return (0..1 priority, reason) for human review.

    Uncertain predictions closest to the decision threshold are most useful.
    Already reviewed samples receive zero priority.
    """
    status = normalize_review(review)
    if status in {"positive", "negative", "corrected", "missed"}:
        return 0.0, "already reviewed"
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = float(threshold)
    score = min(1.0, max(0.0, score))
    threshold = min(0.999, max(0.001, float(threshold)))
    scale = max(threshold, 1.0 - threshold)
    uncertainty = max(0.0, 1.0 - abs(score - threshold) / scale)
    if status == "uncertain":
        return max(0.85, uncertainty), "marked uncertain"
    return uncertainty, "near decision threshold" if uncertainty >= 0.5 else "lower information value"


class FeedbackStore:
    def __init__(self, root):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.samples_dir = os.path.join(self.root, "samples")
        self.snapshots_dir = os.path.join(self.root, "snapshots")
        self.manifest_path = os.path.join(self.root, MANIFEST_NAME)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self._ids = None

    def records(self):
        if not os.path.isfile(self.manifest_path):
            return []
        out = []
        with open(self.manifest_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
        return out

    def ids(self):
        if self._ids is None:
            self._ids = {str(r.get("sample_id")) for r in self.records() if r.get("sample_id")}
        return self._ids

    def has(self, sample_id):
        return str(sample_id) in self.ids()

    def append(self, record):
        rec = dict(record)
        sid = str(rec.get("sample_id") or "").strip()
        if not sid:
            raise ValueError("feedback record requires sample_id")
        if self.has(sid):
            return False
        rec.setdefault("schema_version", SCHEMA_VERSION)
        rec.setdefault("created_utc", utc_now())
        with open(self.manifest_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        self.ids().add(sid)
        return True

    def summary(self):
        recs = self.records()
        return {
            "records": len(recs),
            "by_status": dict(Counter(r.get("status", "unknown") for r in recs)),
            "by_split": dict(Counter(r.get("split", "unknown") for r in recs)),
            "by_geography": dict(Counter(r.get("geography") or "unspecified" for r in recs)),
        }


def export_snapshot(root, snapshot_name, class_name="substation"):
    """Export reviewed feedback as a fixed-split COCO dataset snapshot."""
    store = FeedbackStore(root)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(snapshot_name).strip()) or "snapshot"
    out_root = os.path.join(store.snapshots_dir, safe)
    if os.path.exists(out_root):
        raise FileExistsError(f"Snapshot already exists: {out_root}")
    os.makedirs(out_root)

    statuses = {"positive", "negative", "corrected", "missed"}
    records = [r for r in store.records() if r.get("status") in statuses]
    next_img = 1
    next_ann = 1
    stats = defaultdict(Counter)
    geo_stats = defaultdict(Counter)

    for split in ("train", "valid", "test"):
        split_dir = os.path.join(out_root, split)
        os.makedirs(split_dir, exist_ok=True)
        images, annotations = [], []
        for rec in records:
            if rec.get("split") != split:
                continue
            src_rel = rec.get("image")
            if not src_rel:
                continue
            src = os.path.join(store.root, src_rel)
            if not os.path.isfile(src):
                continue
            ext = os.path.splitext(src)[1] or ".jpg"
            dst_name = f"{next_img:07d}_{rec['sample_id']}{ext.lower()}"
            shutil.copy2(src, os.path.join(split_dir, dst_name))
            size = int(rec.get("chip_size") or 0)
            width = int(rec.get("width") or size or 750)
            height = int(rec.get("height") or size or 750)
            img_id = next_img
            next_img += 1
            images.append({
                "id": img_id,
                "file_name": dst_name,
                "width": width,
                "height": height,
                "gridmapper_sample_id": rec.get("sample_id"),
                "gridmapper_geography": rec.get("geography") or "unspecified",
                "gridmapper_status": rec.get("status"),
                "gridmapper_source_model": rec.get("model") or "",
                "gridmapper_created_utc": rec.get("created_utc") or "",
            })
            for ann in rec.get("annotations") or []:
                bbox = [float(v) for v in ann.get("bbox", [])]
                if len(bbox) != 4 or bbox[2] <= 1 or bbox[3] <= 1:
                    continue
                seg = ann.get("segmentation") or []
                annotations.append({
                    "id": next_ann,
                    "image_id": img_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "segmentation": seg,
                })
                next_ann += 1
            status = rec.get("status", "unknown")
            geography = rec.get("geography") or "unspecified"
            stats[split][status] += 1
            geo_stats[geography][split] += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 0, "name": class_name + "s", "supercategory": "none"},
                {"id": 1, "name": class_name, "supercategory": class_name + "s"},
            ],
            "info": {
                "description": "Grid Mapper human-reviewed continual-learning snapshot",
                "schema_version": SCHEMA_VERSION,
                "created_utc": utc_now(),
                "split_is_fixed": True,
            },
        }
        with open(os.path.join(split_dir, "_annotations.coco.json"), "w", encoding="utf-8") as fh:
            json.dump(coco, fh, indent=2)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "snapshot": safe,
        "created_utc": utc_now(),
        "source_manifest": os.path.abspath(store.manifest_path),
        "counts": {k: dict(v) for k, v in stats.items()},
        "geography_counts": {k: dict(v) for k, v in geo_stats.items()},
        "total_records": sum(sum(v.values()) for v in stats.values()),
        "fixed_splits": True,
        "notes": "Negative images intentionally have no annotations and are retained for hard-negative training.",
    }
    with open(os.path.join(out_root, "dataset_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return out_root, summary
