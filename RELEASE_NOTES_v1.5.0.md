# Grid Mapper v1.5.0 — resumable country scanning

Grid Mapper can now build a national grid dataset without asking OpenStreetMap/Overpass for an entire country in one request.

## Highlights

- New **Scan an entire country (resumable)** tool.
- 25 km cells by default, configurable from 5–100 km.
- Process only a chosen number of cells per run; `0` means continue until the queue finishes.
- SQLite checkpointing and automatic recovery of interrupted cells.
- Persistent GeoPackage plus segment-status GeoJSON and raw per-cell OSM JSON.
- Independent OSM and AI queues: build the national grid now, run RF-DETR later.
- Adaptive Overpass splitting, retries, cache reuse and stale-cache fallback.
- Towers optional; distribution poles remain off by default.
- `Map the grid` no longer fails the whole job merely because no AI model/API key is configured.
- Python 3.12 UTC deprecation warning fixed.

## Recommended national-scan settings

Start with lines + substations + plants, 25 km segments, 20 OSM segments per run, a 2 second delay, and no AI. Once OSM reaches 100%, enable a validated local RF-DETR/ONNX model and process 1–5 AI segments per run.

## Important

This release candidate still needs a live QGIS 4.0.1 smoke test for the new country scanner before official repository publication.
