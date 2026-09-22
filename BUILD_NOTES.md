# Grid Mapper 1.5.0 build notes

Built on v1.4.1 public-distribution hardening and the v1.4 continual-learning/model-registry work.

## v1.5 country-scale architecture

- New **Scan an entire country (resumable)** Processing algorithm and toolbar action.
- Country/AOI is divided into deterministic approximate-kilometre cells (25 km default); the plugin never needs one giant country-wide Overpass query.
- Every OSM cell has persistent `pending/running/retry/failed/done` state in SQLite.
- Successful segments are committed immediately to `country_grid.gpkg` and raw OSM JSON is retained per segment.
- Re-running the same workspace resumes; interrupted `running` cells automatically become `retry`.
- OSM and AI are independent queues, allowing a national OSM database to be completed before a model exists.
- Optional AI processes only OSM-complete segments, with a conservative default of two segments per run.
- Per-run limits and request delay make long scans intentionally slow and restartable.
- Combined OSM calls fall back to smaller asset-specific queries after timeout; stale cached OSM data can be used when all live mirrors fail.
- Missing AI model/API key leaves the AI queue pending and never discards OSM progress.
- `Map the grid` now returns successful OSM results if AI was requested but no model/key is configured.
- Fixed Python 3.12 `datetime.utcnow()` deprecation in feedback snapshot export.

## Runtime testing still required

This build was syntax/unit/package tested outside QGIS. Before official publication, smoke-test in QGIS 4.0.1:

1. `Map the grid` OSM-only on Maputo/Harare;
2. `Scan an entire country` with a small AOI and 2-3 segments;
3. close QGIS, reopen, and verify the same workspace resumes;
4. force/cancel a segment and verify it becomes `retry` on the next run;
5. open `country_grid.gpkg` and `segments.geojson` in QGIS;
6. test a single local ONNX AI segment after registering a model;
7. test Roboflow only with permitted imagery and a test key.
