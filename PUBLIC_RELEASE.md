# Grid Mapper 1.5.0 public release checklist

This source tree is prepared as a release candidate for public distribution. Live QGIS smoke testing is still required before official QGIS Plugin Repository submission.

1. Install `grid_mapper-1.5.0.zip` into a clean QGIS 4.0.1 profile and confirm Plugin Manager shows **1.5.0**.
2. Run default OSM-only `Map the grid`; it must succeed without a model/runtime.
3. Run the new resumable country scanner over a small polygon, then re-run the same workspace and verify completed cells are skipped.
4. Verify `scan_state.sqlite`, `country_grid.gpkg`, `segments.geojson`, `summary.json` and `osm_segments/*.json` are created.
5. Test local ONNX and/or Roboflow AI country segments separately.
6. Update the public GitHub repository to this exact source revision and tag `v1.5.0`.
7. Upload `grid_mapper-1.5.0.zip` to the official QGIS Plugin Repository only after smoke tests pass.

The install ZIP contains no model weights, compiled AI runtimes, secrets, notebooks or test suite.
