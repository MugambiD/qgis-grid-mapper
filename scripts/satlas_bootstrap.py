"""SatlasPretrain -> Grid Mapper RF-DETR bootstrap converter.

The converter intentionally ingests only ``power_substation`` annotations from the
high-resolution SatlasPretrain static labels and copies only NAIP 512x512 chips
that are actually selected for training/validation/test.

Satlas raw storage is distributed in large tar archives, so this module cannot
avoid downloading whichever source archive(s) the user chooses to obtain.  Once
those archives are extracted, however, it does *not* copy the rest of Satlas into
the Grid Mapper training dataset.

No QGIS dependency.  It is safe to run in Colab or a normal Python environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image

SATLAS_PARENT_PIXELS = 8192
SATLAS_CHILD_PIXELS = 512
SATLAS_CHILD_FACTOR = SATLAS_PARENT_PIXELS // SATLAS_CHILD_PIXELS  # 16 (z13 -> z17)
DEFAULT_CLASS = "power_substation"


def _stable_fraction(key: str) -> float:
    return int(hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _tile_key(col: int, row: int) -> str:
    return f"{int(col)}_{int(row)}"


def _load_split(path: Path) -> set[Tuple[int, int]]:
    if not path.is_file():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {(int(x[0]), int(x[1])) for x in raw}


def _clip_edge(points: Sequence[Tuple[float, float]], inside, intersect):
    if not points:
        return []
    out = []
    prev = points[-1]
    prev_in = inside(prev)
    for cur in points:
        cur_in = inside(cur)
        if cur_in:
            if not prev_in:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect(prev, cur))
        prev, prev_in = cur, cur_in
    return out


def clip_polygon_to_rect(points: Sequence[Tuple[float, float]], xmin: float, ymin: float,
                         xmax: float, ymax: float) -> List[Tuple[float, float]]:
    """Sutherland-Hodgman clipping for one exterior polygon ring."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return []

    def ix(xval):
        def f(a, b):
            ax, ay = a; bx, by = b
            if bx == ax:
                return (xval, ay)
            t = (xval - ax) / (bx - ax)
            return (xval, ay + t * (by - ay))
        return f

    def iy(yval):
        def f(a, b):
            ax, ay = a; bx, by = b
            if by == ay:
                return (ax, yval)
            t = (yval - ay) / (by - ay)
            return (ax + t * (bx - ax), yval)
        return f

    pts = _clip_edge(pts, lambda p: p[0] >= xmin, ix(xmin))
    pts = _clip_edge(pts, lambda p: p[0] <= xmax, ix(xmax))
    pts = _clip_edge(pts, lambda p: p[1] >= ymin, iy(ymin))
    pts = _clip_edge(pts, lambda p: p[1] <= ymax, iy(ymax))
    # Drop consecutive duplicates caused by clipping.
    cleaned = []
    for p in pts:
        if not cleaned or abs(p[0] - cleaned[-1][0]) > 1e-6 or abs(p[1] - cleaned[-1][1]) > 1e-6:
            cleaned.append(p)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned if len(cleaned) >= 3 else []


def _polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1] -
                   points[(i + 1) % len(points)][0] * points[i][1]
                   for i in range(len(points))) / 2.0)


def _bbox(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _image_name_from_vector(data: dict) -> Optional[str]:
    meta = data.get("metadata") or {}
    for key in ("ImageName", "image_name", "imagename"):
        if meta.get(key):
            return str(meta[key])
    return None


class NaipLocator:
    """Find high-resolution NAIP tiles, preferring Satlas' ImageName metadata."""
    def __init__(self, satlas_root: os.PathLike):
        self.root = Path(satlas_root)
        self.naip = self.root / "naip"
        if not self.naip.exists() and (self.root / "naip_small").exists():
            self.naip = self.root / "naip_small"
        self.cache: Dict[Tuple[Optional[str], int, int], Optional[Path]] = {}

    def find(self, image_name: Optional[str], col17: int, row17: int) -> Optional[Path]:
        key = (image_name, int(col17), int(row17))
        if key in self.cache:
            return self.cache[key]
        fname = f"{col17}_{row17}.png"
        if image_name:
            p = self.naip / image_name / "tci" / fname
            if p.is_file():
                self.cache[key] = p
                return p
        # Fallback for static labels without usable metadata or partial NAIP roots.
        matches = sorted(self.naip.glob(f"*/tci/{fname}")) if self.naip.exists() else []
        p = matches[-1] if matches else None
        self.cache[key] = p
        return p


@dataclass
class SatlasSample:
    image_path: Path
    split: str
    parent_tile: Tuple[int, int]
    child_tile: Tuple[int, int]
    polygons: List[List[float]]
    image_name: Optional[str]
    positive: bool

    @property
    def stable_key(self) -> str:
        return f"satlas:{self.parent_tile[0]}:{self.parent_tile[1]}:{self.child_tile[0]}:{self.child_tile[1]}"


def iter_satlas_samples(satlas_root: os.PathLike, class_name: str = DEFAULT_CLASS,
                        negative_ratio: float = 2.0, max_positive_chips: int = 4000,
                        max_negative_chips: int = 8000, valid_pct: float = 10.0,
                        include_official_test: bool = False,
                        min_polygon_area_px: float = 16.0) -> Iterator[SatlasSample]:
    """Yield selected high-resolution Satlas chips with substation labels.

    Only static label folders where ``class_name`` is explicitly present are
    considered.  This is important: a missing class key means the class was not
    annotated, while ``"power_substation": []`` is a valid hard negative.

    Satlas' zoom-13 8192x8192 annotation coordinates are clipped into the
    corresponding 16x16 zoom-17 NAIP chips (512x512 each).
    """
    root = Path(satlas_root)
    static_root = root / "static"
    meta_root = root / "metadata"
    if not static_root.is_dir():
        raise FileNotFoundError(f"Satlas static labels not found: {static_root}")
    locator = NaipLocator(root)
    if not locator.naip.is_dir():
        raise FileNotFoundError(f"Satlas NAIP imagery not found under {root}/naip or {root}/naip_small")

    train_tiles = _load_split(meta_root / "train_highres.json")
    test_tiles = _load_split(meta_root / "test_highres.json")

    positives: List[SatlasSample] = []
    negative_candidates: List[SatlasSample] = []

    for vf in sorted(static_root.glob("*/vector.json")):
        try:
            col13, row13 = (int(x) for x in vf.parent.name.split("_")[:2])
        except Exception:
            continue
        data = json.loads(vf.read_text(encoding="utf-8"))
        if class_name not in data:
            continue  # not annotated for this class; never assume it is negative
        parent = (col13, row13)
        if parent in test_tiles and not include_official_test:
            continue
        if parent in test_tiles:
            split = "test"
        else:
            # Preserve official training domain, then carve a deterministic validation set.
            frac = _stable_fraction(f"satlas-valid:{col13}:{row13}") * 100.0
            split = "valid" if frac < float(valid_pct) else "train"

        image_name = _image_name_from_vector(data)
        per_child: Dict[Tuple[int, int], List[List[float]]] = defaultdict(list)
        features = data.get(class_name) or []
        for feat in features:
            geom = (feat or {}).get("Geometry") or {}
            if str(geom.get("Type", "")).lower() != "polygon":
                continue
            rings = geom.get("Polygon") or []
            if not rings:
                continue
            outer = [(float(x), float(y)) for x, y in rings[0]]
            if len(outer) < 3:
                continue
            bx1, by1, bx2, by2 = _bbox(outer)
            sx0 = max(0, min(15, int(math.floor(bx1 / SATLAS_CHILD_PIXELS))))
            sy0 = max(0, min(15, int(math.floor(by1 / SATLAS_CHILD_PIXELS))))
            sx1 = max(0, min(15, int(math.floor(max(0.0, bx2 - 1e-9) / SATLAS_CHILD_PIXELS))))
            sy1 = max(0, min(15, int(math.floor(max(0.0, by2 - 1e-9) / SATLAS_CHILD_PIXELS))))
            for sy in range(sy0, sy1 + 1):
                for sx in range(sx0, sx1 + 1):
                    xmin, ymin = sx * SATLAS_CHILD_PIXELS, sy * SATLAS_CHILD_PIXELS
                    clipped = clip_polygon_to_rect(outer, xmin, ymin,
                                                   xmin + SATLAS_CHILD_PIXELS,
                                                   ymin + SATLAS_CHILD_PIXELS)
                    if len(clipped) < 3 or _polygon_area(clipped) < min_polygon_area_px:
                        continue
                    local = []
                    for x, y in clipped:
                        local.extend([min(512.0, max(0.0, x - xmin)),
                                      min(512.0, max(0.0, y - ymin))])
                    per_child[(sx, sy)].append(local)

        # Positive child chips.
        for (sx, sy), polygons in per_child.items():
            c17, r17 = col13 * SATLAS_CHILD_FACTOR + sx, row13 * SATLAS_CHILD_FACTOR + sy
            img = locator.find(image_name, c17, r17)
            if img:
                positives.append(SatlasSample(img, split, parent, (c17, r17), polygons, image_name, True))

        # Any child chip in an explicitly annotated parent tile that contains no
        # substation polygon is a valid negative candidate.
        positive_children = set(per_child)
        for sy in range(SATLAS_CHILD_FACTOR):
            for sx in range(SATLAS_CHILD_FACTOR):
                if (sx, sy) in positive_children:
                    continue
                c17, r17 = col13 * SATLAS_CHILD_FACTOR + sx, row13 * SATLAS_CHILD_FACTOR + sy
                img = locator.find(image_name, c17, r17)
                if img:
                    negative_candidates.append(SatlasSample(img, split, parent, (c17, r17), [], image_name, False))

    # Deterministic cap so repeated training runs see the same bootstrap set.
    positives.sort(key=lambda s: hashlib.sha1(s.stable_key.encode()).hexdigest())
    if max_positive_chips and len(positives) > max_positive_chips:
        positives = positives[:max_positive_chips]

    wanted_neg = int(round(len(positives) * max(0.0, negative_ratio)))
    if max_negative_chips:
        wanted_neg = min(wanted_neg, int(max_negative_chips))
    negative_candidates.sort(key=lambda s: hashlib.sha1(("neg:" + s.stable_key).encode()).hexdigest())
    negatives = negative_candidates[:wanted_neg]

    # Stable ordering across positives/negatives.
    combined = positives + negatives
    combined.sort(key=lambda s: hashlib.sha1((s.split + ":" + s.stable_key).encode()).hexdigest())
    yield from combined


def selection_summary(samples: Sequence[SatlasSample]) -> dict:
    by_split = Counter(s.split for s in samples)
    pos_split = Counter(s.split for s in samples if s.positive)
    neg_split = Counter(s.split for s in samples if not s.positive)
    return {
        "samples": len(samples),
        "positive_chips": sum(s.positive for s in samples),
        "negative_chips": sum(not s.positive for s in samples),
        "by_split": dict(by_split),
        "positive_by_split": dict(pos_split),
        "negative_by_split": dict(neg_split),
    }


def export_coco(samples: Sequence[SatlasSample], output_dir: os.PathLike,
                copy_mode: str = "copy", class_display_name: str = "substation") -> dict:
    """Export selected Satlas samples as an RF-DETR compatible COCO dataset."""
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    stores = {s: {"images": [], "annotations": []} for s in ("train", "valid", "test")}
    img_id = ann_id = 0
    for sample in samples:
        split = sample.split
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        img_id += 1
        dst_name = f"satlas_{sample.parent_tile[0]}_{sample.parent_tile[1]}_{sample.child_tile[0]}_{sample.child_tile[1]}.jpg"
        dst = split_dir / dst_name
        with Image.open(sample.image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            im.save(dst, quality=95)
        stores[split]["images"].append({
            "id": img_id, "file_name": dst_name, "width": w, "height": h,
            "gridmapper_source": "satlaspretrain_power_substation",
            "gridmapper_geography": "satlas_naip",
            "gridmapper_satlas_parent": _tile_key(*sample.parent_tile),
            "gridmapper_satlas_child": _tile_key(*sample.child_tile),
        })
        for poly in sample.polygons:
            xs, ys = poly[0::2], poly[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            ann_id += 1
            stores[split]["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                "segmentation": [list(map(float, poly))],
            })

    cats = [{"id": 0, "name": class_display_name + "s", "supercategory": "none"},
            {"id": 1, "name": class_display_name, "supercategory": class_display_name + "s"}]
    for split, store in stores.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        (out / split / "_annotations.coco.json").write_text(json.dumps({
            "images": store["images"], "annotations": store["annotations"], "categories": cats,
            "info": {
                "description": "Grid Mapper SatlasPretrain power_substation bootstrap",
                "source": "AllenAI SatlasPretrain high-resolution NAIP",
                "license_notice": "Satlas contains mixed-source labels; preserve source attribution and review SatlasPretrain source licenses.",
            },
        }, indent=2), encoding="utf-8")
    manifest = selection_summary(samples)
    manifest.update({
        "class": DEFAULT_CLASS,
        "source": "SatlasPretrain",
        "notes": [
            "Only tiles explicitly annotated for power_substation were used.",
            "An absent power_substation key was never treated as a negative.",
            "NAIP imagery is public domain; Satlas annotations include mixed source licenses including ODbL and ODC-BY.",
        ],
    })
    (out / "satlas_bootstrap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def scan_label_catalog(satlas_root: os.PathLike, class_name: str = DEFAULT_CLASS) -> dict:
    """Scan labels without needing imagery; useful before downloading NAIP archives."""
    root = Path(satlas_root)
    static_root = root / "static"
    result = {"annotated_tiles": 0, "positive_parent_tiles": 0, "negative_parent_tiles": 0,
              "instances": 0, "image_names": [], "years": {}}
    image_names = set(); years = Counter()
    for vf in sorted(static_root.glob("*/vector.json")):
        data = json.loads(vf.read_text(encoding="utf-8"))
        if class_name not in data:
            continue
        result["annotated_tiles"] += 1
        feats = data.get(class_name) or []
        if feats:
            result["positive_parent_tiles"] += 1
            result["instances"] += len(feats)
        else:
            result["negative_parent_tiles"] += 1
        name = _image_name_from_vector(data)
        if name:
            image_names.add(name)
            tail = name.rsplit("_", 1)[-1]
            year = tail[:4] if len(tail) >= 4 and tail[:4].isdigit() else "unknown"
            years[year] += 1
    result["image_names"] = sorted(image_names)
    result["years"] = dict(sorted(years.items()))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--satlas-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--negative-ratio", type=float, default=2.0)
    ap.add_argument("--max-positive", type=int, default=4000)
    ap.add_argument("--max-negative", type=int, default=8000)
    ap.add_argument("--valid-pct", type=float, default=10.0)
    ap.add_argument("--include-official-test", action="store_true")
    ap.add_argument("--catalog-only", action="store_true")
    args = ap.parse_args(argv)
    if args.catalog_only:
        summary = scan_label_catalog(args.satlas_root)
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "image_names"}, indent=2))
        return 0
    samples = list(iter_satlas_samples(
        args.satlas_root, negative_ratio=args.negative_ratio,
        max_positive_chips=args.max_positive, max_negative_chips=args.max_negative,
        valid_pct=args.valid_pct, include_official_test=args.include_official_test,
    ))
    manifest = export_coco(samples, args.out)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
