# Resumable country scanning

Grid Mapper 1.5 adds a persistent country scanner designed for unreliable public APIs and long-running African grid-mapping jobs.

## Why segments

A country-wide Overpass query can time out or overload a public mirror. The country scanner geocodes the boundary once, divides it into approximate-kilometre cells, and requests one cell at a time. A failed cell is deferred while successful cells are committed immediately.

## Workspace

```text
workspace/
  scan_state.sqlite
  country_grid.gpkg
  segments.geojson
  summary.json
  osm_segments/
  http_cache/
```

`scan_state.sqlite` is the source of truth for resume/retry state. `country_grid.gpkg` is the persistent vector database. `segments.geojson` can be styled in QGIS by `osm_status` and `ai_status` to show progress.

## Two independent queues

OSM collection and AI detection are independent. You can complete the national OSM grid while no model is installed. Later, select/register a local ONNX model and run only the AI queue over OSM-complete cells.

AI remains a smart scan: likely substation zones are generated around towns, known substations and transmission line ends/junctions/voltage changes. It does not blindly render every square metre of a country by default.

## Recommended settings

For a first Zambia/Zimbabwe/Malawi/Mozambique run: 25 km OSM cells, 20 cells per run, 2 seconds between cells, lines + substations + plants, towers optional and poles off. For AI: 1–5 cells per run with Esri/licensed imagery or a local raster.

## Resuming

Use exactly the same workspace, country/AOI, segment size and overlap. Interrupted `running` cells are changed to `retry` automatically. Completed OSM and AI cells are skipped. A workspace signature prevents accidentally mixing two different countries or segmentation geometries.
