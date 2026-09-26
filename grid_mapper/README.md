# Grid Mapper – Substation Detector (QGIS plugin)

**Version: v1.5.1** — security hardening for resumable country scanning and AI mapping. See [release notes](https://github.com/MugambiD/qgis-grid-mapper/blob/cleanup-v1.5.1/RELEASE_NOTES_v1.5.1.md).

QGIS Processing tools that turn the MSc Data Science thesis
*"Investigating the use of Deep learning tools to Map substations in Kenya"*
(Danson Mugambi, University of East London, 2023) into a repeatable grid-mapping workflow.

| Tool (Processing Toolbox → Grid Mapper) | What it does | Thesis step |
|---|---|---|
| **Map the grid (type a place)** | Type *Zambia* (or a province/district) → substations, power lines and plants from OpenStreetMap; optional towers/poles; optional AI scan for candidate unmapped substations | Whole workflow |
| **Scan an entire country (resumable)** | Divides a country/AOI into cells, saves each successful segment immediately, retries failures independently, resumes across QGIS sessions, and can run a later AI queue over OSM-complete cells | New in 1.5 |
| **Extract training chips around substations** | Renders a 750 × 750 px chip around every known substation, with world file + CRS, and Pascal VOC or YOLO labels from footprint polygons | Data extraction, annotation (ch. 3) |
| **Detect substations (local model)** | Tiles any imagery (drone GeoTIFF, Google/Bing/Esri XYZ) and runs a model offline: **RF-DETR** (ONNX, boxes or true footprints), YOLO (ONNX), or the thesis SSD-MobileNet / EfficientDet (TFLite) | Models 3 & 4, upgraded |
| **Detect substations (Roboflow API)** | Same tiling, inference on the hosted Roboflow models `ss-2/1` (object detection, 85.6 % mAP) and `ss-ovfnj/1` (SegFormer, 79 % mIoU) | Models 1 & 2 |
| **Validate detections against reference substations** | TP / FP / FN, precision, recall, F1 against reference substations | Evaluation |
| **Prioritise detections for active-learning review** | Adds `al_priority` so uncertain, unreviewed detections are checked first | New in 1.4 |
| **Capture reviewed feedback for continual learning** | Saves TP/FP/corrected/missed examples as an append-only georeferenced training store | New in 1.4 |
| **Export continual-learning dataset snapshot** | Produces immutable COCO train/valid/test snapshots with fixed splits and hard negatives | New in 1.4 |
| **Register / safely promote a candidate model** | Versions candidate models and activates only those that pass validation gates | New in 1.4 |

Results are polygon (and optional point) layers with `score`, `area_m2`, `length_m`,
`width_m`, source tile, model, imagery and a `review` field for manual QA.

---

## What changed in v1.5.1

- XML label output uses a small text escaper without importing an XML parser.
- Plugin cache keys, detection deduplication, feedback IDs and dataset split buckets use SHA-256.
- Country-scan queue queries use fixed SQL with bound parameters.
- TFLite runtime probing reports why a backend could not load.
- Roboflow tools retain the `API_KEY` Processing parameter for existing models and scripts.

**Upgrading from v1.5.0:** existing feedback records keep their stored split. Newly captured feedback uses different sample IDs and will not deduplicate against v1.5.0 records. Older cache entries may be regenerated.

## 1. Install the plugin

Download the installable plugin ZIP from [Releases](https://github.com/MugambiD/qgis-grid-mapper/releases). To build v1.5.1 from this source checkout, run the following from the repository root:

```sh
python scripts/build_release.py --check-only
python scripts/build_release.py
```

The output is `dist/grid_mapper-1.5.1.zip`, containing one top-level `grid_mapper/` folder. GitHub's **Code → Download ZIP** archive is a source checkout; build the plugin ZIP before installing it in QGIS.

1. QGIS 3.22 – 4.x. The metadata declares compatibility through QGIS 4.x with `qgisMaximumVersion=4.99`. If an older copy is installed, uninstall it first.
2. *Plugins → Manage and Install Plugins → Install from ZIP* → choose the downloaded zip → **Install Plugin**.
3. A **Grid Mapper** toolbar button and *Plugins → Grid Mapper* menu appear; all tools are also in the Processing Toolbox.

## 2. One-time: model runtime (only for the offline model tools)

QGIS' Python doesn't include a deep-learning runtime. Close QGIS, open **OSGeo4W Shell** (Windows) and install the one you need, pinning numpy to the version QGIS ships so pip can't upgrade it (upgrading can break QGIS):

```
python -c "import numpy; print(numpy.__version__)"
python -m pip install onnxruntime "numpy==<version printed above>"      # RF-DETR / YOLO (.onnx)  – recommended
python -m pip install ai-edge-litert "numpy==<version printed above>"   # thesis TFLite models
```

If a runtime is missing the plugin shows this command with your numpy version filled in. macOS / Linux: same commands with the Python QGIS uses (`python3 -m pip install --user …`). The Roboflow, Map the grid (OSM-only), chip-extraction and validation tools need nothing extra.

## 3. Get or train a model

**Recommended – train an Apache-designated RF-DETR model with the training notebook in the source repository**
[`GridMapper_RF-DETR_training.ipynb`](https://github.com/MugambiD/qgis-grid-mapper/blob/main/notebooks/GridMapper_RF-DETR_training.ipynb) – open it in Google Colab (GPU runtime) and *Run all*. The notebook is kept in the source repository rather than the official QGIS install ZIP. It:

1. optionally bootstraps from AllenAI **SatlasPretrain** high-resolution `power_substation` polygons and hard negatives, then combines them with your Roboflow project (`ss-2`), thesis Drive folders, older chips and QGIS feedback snapshots;
2. converts Satlas zoom-13/8192 px polygons into the corresponding 512 px NAIP child chips without treating unannotated tiles as negatives;
3. fine-tunes RF-DETR with the aerial-imagery augmentation preset (`TASK = "detect"` for boxes or `"segment"` for true footprints);
4. reports built-in RF-DETR metrics plus fixed-threshold precision/recall/F1 and per-geography F1;
5. saves `gridmapper_…zip` (ONNX + class names + promotion metadata) and the best `.pth` checkpoint to Google Drive.

For later training rounds, keep the baseline sources enabled, add the latest immutable QGIS feedback snapshot to `GRIDMAPPER_SNAPSHOT_DIRS`, and optionally warm-start with `RESUME_CHECKPOINT`. This replay + fine-tune approach reduces catastrophic forgetting. Select the candidate zip directly in QGIS – no unzipping needed.

For Satlas setup and provenance/licensing notes see [`docs/SATLAS_BOOTSTRAP.md`](https://github.com/MugambiD/qgis-grid-mapper/blob/main/docs/SATLAS_BOOTSTRAP.md). The raw Satlas distribution is archive-based; Grid Mapper only ingests selected substation chips after extraction.

**Thesis TFLite models**
* SSD-MobileNet-V2-FPNLite-320: `Thesis/3_SSD-Mobilenet V2-FPN Model/custom_model_lite.zip` on Google Drive.
* EfficientDet: `Thesis/4_Efficientdet Model/custom_model_lite (2).zip`.

**Other ONNX models** – Ultralytics YOLO exports (`format="onnx"`, no NMS) also load; check the licence of the exact model/code you use. RF-DETR core/open-source package and Apache-designated model weights are Apache-2.0, while RF-DETR Plus components such as XL/2XL detection models have different licensing. Model licences are separate from Grid Mapper's GPL licence.

**Roboflow**: model IDs `ss-2` (version 1) and `ss-ovfnj` (version 1) in the `substations` workspace.
Prefer setting the `ROBOFLOW_API_KEY` environment variable. The Processing framework can record parameter values in local logs, so avoid pasting long-lived secrets into ordinary processing-string fields when possible.

## 4. Map the grid – one click

Toolbar → **Map the grid**:

1. **Place to map**: `Zambia`, `Copperbelt Province`, `Lusaka District`, … (or your own polygon).
2. **Grid layers**: the public default is substations + lines + power plants. Transmission towers and distribution poles are optional; poles are deliberately off by default because they can be very large OSM datasets.
3. **Substation detector**: the public default is **No AI scan - OpenStreetMap data only**. This means a fresh installation works without a model, imagery request, ONNX/TFLite runtime or API key.
4. To run AI, explicitly choose a local model or Roboflow. Only then is imagery initialised.
5. Use **Estimate only** before a large AI run to calculate the likely tile count.

What happens:

* The boundary comes from OpenStreetMap Nominatim, and the existing grid from Overpass.
* The core grid query is independent of optional tower/pole downloads. A tower/pole timeout no longer aborts the run.
* Towers and poles are fetched in smaller bbox cells, deduplicated, then clipped back to the true AOI. Partial successful cells are kept.
* If AI is selected, **scan zones** are drawn around towns, line ends/junctions/voltage changes and known substations.
* The AI scans only those zones and labels each detection **new** (not in OSM) or **in OSM**.
* Output layers load with default styling, including voltage classes for lines.

A country-scale AI scan can still require thousands of imagery tiles. Run a province/district first, or shrink scan-zone radii. For optional tower/pole layers, country-scale requests may be skipped when they would create too many public-Overpass cells; use a smaller AOI instead. The AI model detects substations only.

Self-hosted Nominatim / Overpass servers can be used by setting the `GRIDMAPPER_NOMINATIM_URL` / `GRIDMAPPER_OVERPASS_URL` environment variables.

### Public-server behaviour in v1.4.1

Grid Mapper treats lines/substations/plants as the core OSM result. Transmission towers and distribution poles are optional best-effort layers. If one or more structure cells time out, the successful cells are kept and the core grid result continues. This is specifically designed to avoid the 504 timeout pattern seen when public Overpass servers are asked for every pole across a large city or region.

## 4A. Country scanner – resumable national mapping (v1.5)

Use **Grid Mapper → Scan an entire country (resumable)** when the AOI is too large for one Overpass request. A Zambia-style run can use 25 km segments and process only 10–25 segments per QGIS session. Each successful cell is committed before the next request.

The workspace contains:

```text
my_country_scan/
  scan_state.sqlite          # resume/retry state for OSM and AI queues
  country_grid.gpkg          # persistent lines/substations/plants/towers/poles/AI detections
  segments.geojson           # progress map with OSM + AI status per cell
  summary.json               # machine-readable progress summary
  osm_segments/              # successful raw OSM JSON by segment
  http_cache/                # request cache; stale data can be used if every live mirror fails
```

Recommended first run: **25 km segments**, **20 OSM segments per run**, **2 second delay**, lines + substations + plants, and **No AI**. Re-run with the same workspace until OSM reaches 100%. Then select a local RF-DETR/ONNX model (or Roboflow) and process a few AI segments per run. The AI queue is independent, so registering a model later does not discard the completed country grid.

Distribution poles remain off by default because they can dominate public Overpass workloads. Country scanning is intentionally slow and polite; it is designed to finish over repeated sessions, not to flood public servers.

## 5. Typical workflow (individual tools)

1. **Add imagery** – Browser → XYZ Tiles → New connection, e.g.
   Google Satellite `https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}` or
   Esri World Imagery `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`.
   Drone orthomosaics / GeoTIFFs work too and give the best results.
2. **Detect** – *Detect substations (local model)*:
   * Imagery layer: the basemap or raster
   * Area to scan: draw on the canvas, or give an AOI polygon layer (a county, a corridor buffer)
   * Model: your `gridmapper_…zip` (RF-DETR) or `custom_model_lite.zip` (TFLite)
   * Ground resolution **0.5–0.6 m/px**, tile **750 px**, overlap **25 %**, confidence **0.5**
3. **Prioritise** – run *Prioritise detections for active-learning review* and sort by `al_priority` descending.
4. **Review** – edit a detection geometry if the box/footprint is wrong and set `review` to `TP`, `FP`, `corrected` or `uncertain`.
5. **Validate** – run *Validate detections…* against a reference layer; its *Missed* output contains false negatives.
6. **Capture** – run *Capture reviewed feedback for continual learning*, supplying the same imagery plus the optional *Missed* layer. TP/corrected/missed examples become positives; FP examples become hard negatives.
7. **Snapshot** – run *Export continual-learning dataset snapshot*. Each sample keeps a permanent train/valid/test assignment; adding future samples never reshuffles old validation examples.
8. **Retrain** – open the included RF-DETR Colab notebook, add the snapshot to `GRIDMAPPER_SNAPSHOT_DIRS`, keep older/baseline data enabled, and optionally use the previous `.pth` checkpoint.
9. **Promote safely** – run *Register / safely promote a candidate model*. A candidate becomes active only if its overall validation metric improves enough and no shared geography regresses beyond the configured gate. When no model file is selected, local detection and *Map the grid* use the approved active model automatically.

### Parameter tips

| Parameter | Guidance |
|---|---|
| Ground resolution | Keep it close to the training imagery (≈ 0.3–0.6 m/px, XYZ zoom 18–19). A coarser value means fewer tiles but smaller-looking substations. |
| Tile size | 750 px matches the thesis chips. Models resize each tile to their own input size (e.g. 320 px TFLite, 384–576 px RF-DETR). |
| Overlap | 20–30 % ensures every substation appears whole in at least one tile. Duplicates are merged (NMS) and clipped fragments are dropped. |
| Safety limit | Stops runs that would request thousands of basemap tiles. A 1 × 1 km area at 0.6 m/px with 25 % overlap = 9 tiles. |
| Save rendered tiles | Writes every tile as a georeferenced JPG (+ .jgw) so you can reuse them as training data. |

## 6. Continual-learning design

Grid Mapper does **not** train on its own unchecked predictions. Learning is human-in-the-loop:

```text
Detect → prioritise → human review/correction → capture feedback → fixed snapshot
      → GPU retraining in Colab → validation by geography → promote or reject → active model
```

The feedback store is append-only and contains georeferenced image chips, metadata, annotations and hard negatives. Re-running the same reviewed feature is deduplicated. New samples receive a stable hash-based split, so the validation/test sets remain comparable from model version to model version.

The model registry lives inside the QGIS profile under `grid_mapper/model_registry/`. Candidate packages are copied into versioned folders. The promotion gate prefers the model metadata's `primary_metric` (the supplied notebook exports `f1`) and also checks `geography_metrics`. A failed candidate stays registered for inspection but never replaces the current active model.

See [`docs/CONTINUAL_LEARNING.md`](docs/CONTINUAL_LEARNING.md) for the detailed workflow.

## 7. Network services, privacy and licences

Grid Mapper has no telemetry and does not send project data to the plugin author. Depending on the options selected, it contacts OpenStreetMap Nominatim/Overpass, the chosen imagery provider, and — only in Roboflow mode — Roboflow. Roboflow mode uploads rendered image tiles for inference. Local ONNX/TFLite inference stays on the user's computer. See [`docs/NETWORK_AND_PRIVACY.md`](docs/NETWORK_AND_PRIVACY.md).

The plugin ZIP contains no model weights or compiled AI runtimes. Optional Python runtimes are installed separately into QGIS Python. Users are responsible for checking the licence and usage terms of their chosen models, imagery and hosted services.

## 8. Notes & limits

* Thesis model accuracy: SSD-MobileNet mAP ≈ 51 %, EfficientDet ≈ 17 %, Roboflow OD 85.6 %. RF-DETR models report their own test mAP in the notebook and in the model's `.json`. Treat all output as candidates for review, not as final asset data.
* Scanning large areas of Google/Bing tiles may conflict with their terms of use. For
  production mapping, use licensed imagery (e.g. Esri, Maxar, Planet, own drone data).
* The Roboflow tool uploads every tile to Roboflow, so it uses API credits.
* Roboflow semantic segmentation (`ss-ovfnj`) masks are vectorised with GDAL; the
  score is fixed at 1.0 because the API returns no per-object confidence.

## 9. Files

```
.github/workflows/quality.yml   runs tests, Bandit and secrets checks
.github/workflows/release.yml   checks and builds the installable ZIP on every v* tag
scripts/build_release.py       validates and packages grid_mapper/
scripts/check_secrets.py       fails CI when the secrets scan reports findings
docs/                         Satlas bootstrap guide
notebooks/                      Colab notebook to train RF-DETR and export for the plugin
grid_mapper/
  metadata.txt, __init__.py, plugin.py, provider.py
  core/tiling.py            rendering, tiling, georeferencing
  core/tflite_detector.py   model loader (zip/tflite/onnx) + TFLite SSD parsing
  core/onnx_detector.py     RF-DETR (detect/segment) and YOLO ONNX inference
  core/feedback_store.py     append-only reviewed dataset + fixed COCO snapshots
  core/model_registry.py     candidate versioning + safe promotion gates
  core/masks.py             mask → polygon (GDAL)
  core/roboflow_client.py   hosted inference client (uses QGIS proxy settings)
  core/postprocess.py       cross-tile NMS
  core/osm.py, core/net.py  Nominatim / Overpass access
  core/compat.py            QGIS 3 / QGIS 4 (Qt6) compatibility
  core/xmltext.py           XML label text escaping
  algorithms/               detection, active-learning, feedback, snapshot and model-registry tools
  docs/                     continual-learning workflow and model lifecycle
```

Licence: GPL-2.0-or-later (as required for QGIS plugins).
