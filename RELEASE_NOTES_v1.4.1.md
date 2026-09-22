# Grid Mapper v1.4.1

Grid Mapper 1.4.1 is the public-distribution hardening release.

## Highlights

- **Works on a fresh install without an AI model.** `Map the grid` now defaults to OpenStreetMap-only mode, so users can map substations, lines and power plants without installing ONNX/TFLite runtimes.
- **Fixes the Overpass timeout pattern seen in Harare.** Transmission towers and distribution poles are no longer requested together in one large query. They are separate optional layers and are fetched in smaller cells.
- **Distribution poles are off by default.** Pole datasets can be extremely large and are rarely necessary for utility-scale renewable-energy screening.
- **Optional OSM layers cannot kill the core run.** Failed tower/pole cells are skipped; successful cells and the core grid layers are retained.
- **True AOI clipping.** Bbox-tiled tower/pole results are clipped back to the named administrative boundary or supplied AOI.
- **QGIS 4 publication metadata updated.** `qgisMaximumVersion=4.99` remains; the obsolete `supportsQt6` flag is removed.
- **Public-package disclosure improved.** Optional dependencies, network services, Roboflow imagery uploads, API-key handling and model/image licensing considerations are documented.
- **No bundled model binaries.** The official ZIP contains Python/source documentation only.

## Existing v1.4 capabilities retained

- ONNX RF-DETR and YOLO inference;
- RF-DETR segmentation footprints;
- thesis TFLite model support;
- Roboflow inference;
- active-learning prioritisation;
- append-only TP/FP/corrected/missed feedback capture;
- fixed train/validation/test snapshot export;
- geography-aware validation metadata;
- safe candidate-model registry and promotion gates.

## Recommended first public test

Install `grid_mapper-1.4.1.zip` in a clean QGIS profile and run:

- Place: `Harare`
- Layers: Substations + Transmission/distribution lines + Power plants
- Detector: `No AI scan - OpenStreetMap data only`

Then test transmission towers separately. Test distribution poles only on a small district/AOI.
