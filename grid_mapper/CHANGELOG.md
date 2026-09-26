# Changelog

## 1.5.1

Security-scan hardening, in response to the official QGIS plugin repository's automated scan (Bandit and detect-secrets). No behaviour of the mapping tools changes.

- Replaced the `xml.sax.saxutils` import with a small built-in XML text escaper (`core/xmltext.py`): the plugin writes label XML but never parses untrusted XML.
- Cache and dataset keys now use SHA-256 instead of MD5/SHA-1 (OSM query cache, model zip cache, AI detection dedup, feedback sample ids and dataset splits).
- Country-scan queue lookups are now fixed SQL statements with bound parameters only; no query text is assembled from variables.
- TFLite backend probing collects and reports why each runtime failed instead of silently continuing.
- Renamed the Roboflow key class constants to `KEY_PARAM`; the Processing parameter id stays `API_KEY`, so existing models and scripts keep working.

Note: because sample ids and split buckets are hashed differently, feedback captured by 1.5.1 will not deduplicate against samples captured by 1.5.0. Existing records keep the split recorded with them.

## 1.5.0

Country-scale, resumable scanning:

- Added **Scan an entire country (resumable)** to the Processing toolbox and Grid Mapper toolbar.
- Divides a country or user AOI into deterministic approximate-kilometre cells (25 km default) and requests each OSM segment independently.
- Stores scan state in SQLite and resumes from the first unfinished segment after QGIS closes, crashes or connectivity is lost.
- Commits successful grid assets immediately to `country_grid.gpkg`; stores raw OSM JSON per segment for audit/reuse and `segments.geojson` for progress mapping.
- OSM and AI use independent queues: finish the country grid first, then register an RF-DETR/ONNX model and continue AI segment-by-segment later.
- Adds per-run limits (`20` OSM segments and `2` AI segments by default), request throttling and per-segment retry/failure state.
- Combined OSM requests automatically fall back to smaller asset-specific requests when a public Overpass server times out.
- Expired OSM cache can be used as a last-resort fallback when every live Overpass mirror fails.
- Map the grid now returns the OSM result successfully when a local model or Roboflow key is missing instead of marking the entire algorithm failed.
- Fixed the Python 3.12 `datetime.utcnow()` deprecation warning in continual-learning snapshot export.

## 1.4.1

Public-distribution hardening and first-run reliability:

- OSM-only mapping is now the default, so a fresh install works without a model or AI runtime.
- Imagery is only initialised when an AI detector is selected.
- Transmission towers and distribution poles are separate optional layers; poles remain off by default.
- Heavy OSM structure queries are split into small bbox cells, deduplicated and clipped to the true AOI.
- Optional tower/pole timeouts are non-fatal and can return partial successful cells.
- Existing layer-index compatibility is preserved for saved Processing models from 1.4.0.
- QGIS 4 metadata updated: the obsolete `supportsQt6` flag is removed; `qgisMaximumVersion=4.99` remains the compatibility signal.
- External dependency, privacy, network-service and public-release documentation expanded.
- QGIS network classes remain the default; direct urllib fallback inside QGIS is now explicit opt-in only.

## 1.4.0

- Added active-learning review prioritisation (`al_priority`, `al_reason`).
- Added an append-only human-feedback dataset capturing confirmed detections, false positives, corrected geometry and missed substations.
- Added deterministic train/valid/test assignment so existing validation examples never reshuffle when new feedback arrives.
- Added immutable COCO snapshot export with hard-negative images and geography metadata.
- Extended the RF-DETR Colab notebook to combine baseline + feedback snapshots, optionally warm-start from the previous checkpoint, calculate global and per-geography precision/recall/F1, and export promotion metadata.
- Added a versioned local model registry with conservative promotion gates. Candidate models that fail validation remain registered but cannot replace the active model.
- Local detection and Map the grid can use the safely promoted active model when no explicit model file is selected.
- Preserved the v1.3 ONNX RF-DETR / YOLO backend, RF-DETR segmentation footprints, thesis TFLite models and Roboflow support.

## 1.3.0

- Added ONNX Runtime inference for RF-DETR detection/segmentation and YOLO exports.
- Added RF-DETR training/export notebook.
- Added mask-to-polygon support for segmentation models.
