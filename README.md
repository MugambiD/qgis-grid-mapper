# Grid Mapper – Substation Detector (QGIS plugin)

QGIS Processing tools that turn the MSc Data Science thesis
*"Investigating the use of Deep learning tools to Map substations in Kenya"*
(Danson Mugambi, University of East London, 2023) into a repeatable grid-mapping workflow.

| Tool (Processing Toolbox → Grid Mapper) | What it does | Thesis step |
|---|---|---|
| **Map the grid (type a place)** | Type *Zambia* (or a province/district) → power lines, substations, towers, plants from OpenStreetMap + AI scan of the latest satellite imagery for substations missing from the map | Whole workflow |
| **Extract training chips around substations** | Renders a 750 × 750 px chip around every known substation, with world file + CRS, and Pascal VOC or YOLO labels from footprint polygons | Data extraction, annotation (ch. 3) |
| **Detect substations (TFLite model)** | Tiles any imagery (drone GeoTIFF, Google/Bing/Esri XYZ) and runs the Colab-trained SSD-MobileNet-V2-FPNLite-320 or EfficientDet model offline | Models 3 & 4 |
| **Detect substations (Roboflow API)** | Same tiling, inference on the hosted Roboflow models `ss-2/1` (object detection, 85.6 % mAP) and `ss-ovfnj/1` (SegFormer, 79 % mIoU) | Models 1 & 2 |
| **Validate detections against reference substations** | TP / FP / FN, precision, recall, F1 against `Substations.shp`, KETRACO/KPLC data or OSM | Evaluation |

Results are polygon (and optional point) layers with `score`, `area_m2`, `length_m`,
`width_m`, source tile, model, imagery and a `review` field for manual QA.

---

## 1. Install the plugin

Download `grid_mapper-x.y.z.zip` from the [Releases](https://github.com/MugambiD/qgis-grid-mapper/releases) page (or zip the `grid_mapper` folder yourself).

1. QGIS 3.22 – 4.x (Qt5 and Qt6 builds; tested on 3.34 LTR). If an older copy is installed, uninstall it first.
2. *Plugins → Manage and Install Plugins → Install from ZIP* → choose the downloaded zip → **Install Plugin**.
3. A **Grid Mapper** toolbar button and *Plugins → Grid Mapper* menu appear; all tools are also in the Processing Toolbox.

## 2. One-time: TFLite runtime (only for the offline TFLite tool)

QGIS' Python doesn't include TensorFlow Lite. Close QGIS, then:

**Windows** – open **OSGeo4W Shell** from the Start menu:

```
python -c "import numpy; print(numpy.__version__)"
python -m pip install ai-edge-litert "numpy==<the version printed above>"
```

Pinning numpy stops pip from upgrading the numpy QGIS was built against (which can break QGIS).
If the plugin can't find a runtime it shows this exact command, with your numpy version filled in.

**macOS / Linux** – same command with the Python QGIS uses (`python3 -m pip install --user …`).

Alternatives the plugin also accepts: `tflite-runtime`, or the full `tensorflow` package.
The Roboflow, chip-extraction and validation tools need nothing extra.

## 3. Get the models

* **SSD-MobileNet-V2-FPNLite-320** (best offline model): `Thesis/3_SSD-Mobilenet V2-FPN Model/custom_model_lite.zip` on Google Drive.
* **EfficientDet**: `Thesis/4_Efficientdet Model/custom_model_lite (2).zip`.

Download the zip and select it directly in the tool. The plugin extracts `detect.tflite` and
`labelmap.txt` itself (cached in the QGIS profile folder). A loose `.tflite` file also works.

**Roboflow**: model IDs `ss-2` (version 1) and `ss-ovfnj` (version 1) in the `substations` workspace.
Put your private API key (Roboflow → Settings → API Keys) in the tool, or set the
`ROBOFLOW_API_KEY` environment variable.

## 4. Map the grid – one click

Toolbar → **Map the grid**:

1. **Place to map**: `Zambia`, `Copperbelt Province`, `Lusaka District`, … (or your own polygon).
2. **Grid layers**: substations, lines, towers/poles, power plants.
3. **Satellite imagery**: Esri World Imagery (default), Google Satellite or Bing Aerial, or your own raster.
4. **Substation detector**: your TFLite zip, the Roboflow model (API key), or *OpenStreetMap only*.
5. Tick **Estimate only** for the first run. It downloads the OSM grid and reports how many image tiles the AI scan needs.

What happens:

* The boundary comes from OpenStreetMap Nominatim, and the existing grid from the Overpass API.
* **Scan zones** are drawn where substations are likely: around towns (1 km, cities 2 km), at line ends, junctions and voltage changes (400 m), and around known substations (250 m).
* The AI scans only those zones and labels each detection **new** (not in OSM, red) or **in OSM** (confirms, amber).
* The layers load already styled, with lines coloured by voltage class.

Rough effort for a whole country such as Zambia is several thousand tiles (≈ 1 s each, so a few hours). Run a province first, or shrink the zone radii under *Advanced parameters*. Lines, towers and plants come from OpenStreetMap; the current AI model detects substations only.

Self-hosted Nominatim / Overpass servers can be used by setting the `GRIDMAPPER_NOMINATIM_URL` / `GRIDMAPPER_OVERPASS_URL` environment variables.

## 5. Typical workflow (individual tools)

1. **Add imagery** – Browser → XYZ Tiles → New connection, e.g.
   Google Satellite `https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}` or
   Esri World Imagery `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`.
   Drone orthomosaics / GeoTIFFs work too and give the best results.
2. **Detect** – *Detect substations (TFLite model)*:
   * Imagery layer: the basemap or raster
   * Area to scan: draw on the canvas, or give an AOI polygon layer (a county, a corridor buffer)
   * Model: `custom_model_lite.zip`
   * Ground resolution **0.5–0.6 m/px**, tile **750 px**, overlap **25 %**, confidence **0.5**
3. **Review** – style by `score`, set `review` to `TP`/`FP` as you check each site.
4. **Validate** – *Validate detections…* with the thesis `Substations.shp` (clipped to your AOI) as reference, match distance 100–150 m.
5. **Improve** – feed missed sites (the *Missed* output) and false positives back into
   *Extract training chips…*, label them in Roboflow/LabelImg, and retrain in Colab.

### Parameter tips

| Parameter | Guidance |
|---|---|
| Ground resolution | Keep it close to the training imagery (≈ 0.3–0.6 m/px, XYZ zoom 18–19). A coarser value means fewer tiles but smaller-looking substations. |
| Tile size | 750 px matches the thesis chips. The TFLite model resizes each tile to 320 × 320 internally. |
| Overlap | 20–30 % ensures every substation appears whole in at least one tile. Duplicates are merged (NMS) and clipped fragments are dropped. |
| Safety limit | Stops runs that would request thousands of basemap tiles. A 1 × 1 km area at 0.6 m/px with 25 % overlap = 9 tiles. |
| Save rendered tiles | Writes every tile as a georeferenced JPG (+ .jgw) so you can reuse them as training data. |

## 6. Notes & limits

* Model accuracy is the thesis accuracy: SSD-MobileNet mAP ≈ 51 %, EfficientDet ≈ 17 %,
  Roboflow OD 85.6 %. Treat the output as candidates for review, not as final asset data.
* Scanning large areas of Google/Bing tiles may conflict with their terms of use. For
  production mapping, use licensed imagery (e.g. Esri, Maxar, Planet, own drone data).
* The Roboflow tool uploads every tile to Roboflow, so it uses API credits.
* Roboflow semantic segmentation (`ss-ovfnj`) masks are vectorised with GDAL; the
  score is fixed at 1.0 because the API returns no per-object confidence.

## 7. Files

```
.github/workflows/release.yml   builds the installable zip on every v* tag
grid_mapper/
  metadata.txt, __init__.py, plugin.py, provider.py
  core/tiling.py            rendering, tiling, georeferencing
  core/tflite_detector.py   TFLite loader (zip/tflite), output parsing, preprocessing
  core/roboflow_client.py   hosted inference client (uses QGIS proxy settings)
  core/postprocess.py       cross-tile NMS
  core/osm.py, core/net.py  Nominatim / Overpass access
  core/compat.py            QGIS 3 / QGIS 4 (Qt6) compatibility
  algorithms/               the Processing algorithms (map_grid.py = Map the grid)
```

Licence: GPL-2.0-or-later (as required for QGIS plugins).
