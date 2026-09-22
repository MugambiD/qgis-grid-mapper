"""Validate and build the official QGIS plugin ZIP without generated junk."""
from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "grid_mapper"
REQUIRED = ("metadata.txt", "__init__.py", "LICENSE")
EXCLUDE_PARTS = {"__pycache__", ".git", ".idea", ".vscode", "tests", "notebooks"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def metadata():
    cp = configparser.ConfigParser()
    cp.read(PLUGIN / "metadata.txt", encoding="utf-8")
    return cp["general"]


def validate():
    missing = [name for name in REQUIRED if not (PLUGIN / name).is_file()]
    if missing:
        raise SystemExit(f"Missing required plugin files: {', '.join(missing)}")
    md = metadata()
    for key in ("name", "qgisMinimumVersion", "description", "version", "author", "email", "about", "repository"):
        if not md.get(key, "").strip():
            raise SystemExit(f"metadata.txt missing required value: {key}")
    if md.get("qgisMaximumVersion") != "4.99":
        raise SystemExit("qgisMaximumVersion must be 4.99 for the QGIS 3/4 public package")
    if any(k.lower() == "supportsqt6" for k in md):
        raise SystemExit("Remove obsolete supportsQt6 metadata before release")
    binaries = []
    for p in PLUGIN.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".exe", ".dll", ".so", ".dylib", ".onnx", ".tflite"}:
            binaries.append(str(p.relative_to(PLUGIN)))
    if binaries:
        raise SystemExit("Public plugin package must not bundle binaries/model weights: " + ", ".join(binaries))
    return md


def included_files():
    for p in sorted(PLUGIN.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(PLUGIN)
        if any(part in EXCLUDE_PARTS for part in rel.parts):
            continue
        if p.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        yield p, rel


def build(md):
    out_dir = ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"grid_mapper-{md['version']}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p, rel in included_files():
            zf.write(p, Path("grid_mapper") / rel)
    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f"ZIP integrity failure at {bad}")
        names = set(zf.namelist())
        for name in REQUIRED:
            expected = f"grid_mapper/{name}"
            if expected not in names:
                raise SystemExit(f"ZIP missing {expected}")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    md = validate()
    if not args.check_only:
        build(md)


if __name__ == "__main__":
    main()
