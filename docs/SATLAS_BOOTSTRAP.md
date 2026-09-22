# SatlasPretrain bootstrap for Grid Mapper RF-DETR

Grid Mapper's RF-DETR training notebook can optionally bootstrap from the high-resolution **SatlasPretrain** `power_substation` labels before continual learning on African review data.

## Why Satlas

SatlasPretrain contains high-resolution aerial imagery at roughly 0.5–2 m/pixel and explicitly includes `power_substation` and `power_tower` labels. That makes it a useful generic remote-sensing bootstrap before fine-tuning on Grid Mapper's Kenya/Zambia/Zimbabwe/Malawi/Mozambique examples.

The RF-DETR weights are **not** initialized from the Satlas Swin model directly; the architectures differ. Instead, Grid Mapper converts selected Satlas substation examples into the same COCO dataset used for RF-DETR training.

## Safety rules in the converter

The converter does not treat every unlabeled Satlas tile as a negative. It only uses a tile when `power_substation` is explicitly present in `vector.json`. An empty value (`"power_substation": []`) is a valid hard negative; a missing key means the class may not have been annotated and the tile is ignored.

Satlas high-resolution labels use an 8192×8192 coordinate system inside a zoom-13 parent tile. NAIP imagery is stored as 512×512 zoom-17 children. `scripts/satlas_bootstrap.py` clips every substation polygon into the corresponding 16×16 child grid and carries the polygon or its bounding box into COCO.

By default Satlas' official high-resolution test split is **not** mixed into Grid Mapper's promotion test. Satlas is used for training/validation bootstrap while fixed African tests and QGIS feedback remain the primary promotion evidence. Set `SATLAS_INCLUDE_OFFICIAL_TEST=True` only when you intentionally want a Satlas/NAIP test group.

## Data preparation

Set in the notebook:

```python
USE_SATLAS = True
SATLAS_ROOT = "/content/drive/MyDrive/GridMapper/SatlasPretrain"
```

The extracted root should contain:

```text
SatlasPretrain/
  static/
  metadata/
  naip/          # full NAIP years, or
  naip_small/    # small Satlas sample for converter testing
```

The notebook contains opt-in helpers for the official static-label, metadata and small-NAIP archives. Full NAIP year archives are large, so `SATLAS_NAIP_YEARS_TO_DOWNLOAD` is empty by default. Run the label-catalog cell first to see which image years are referenced, then add only the years you deliberately want to stage.

The official raw distribution is archive-based. The Grid Mapper converter can copy only relevant training chips after extraction, but it cannot reduce the byte size of an upstream tar archive that you choose to download.

## Recommended first experiment

Use RF-DETR Small with aerial augmentation and no scale jitter:

```python
TASK = "detect"
MODEL_SIZE = "small"
USE_AERIAL_AUGMENTATION = True
SCALE_JITTER = False
SATLAS_NEGATIVE_RATIO = 2.0
SATLAS_MAX_POSITIVE_CHIPS = 4000
SATLAS_MAX_NEGATIVE_CHIPS = 8000
```

Keep the thesis/Roboflow source enabled as replay data. Add immutable QGIS feedback snapshots as they become available. Compare each candidate on the same fixed African test examples before promotion.

## Command-line converter

The same converter can be used outside Colab:

```bash
python scripts/satlas_bootstrap.py \
  --satlas-root /data/SatlasPretrain \
  --out /data/gridmapper_satlas_coco \
  --negative-ratio 2 \
  --max-positive 4000 \
  --max-negative 8000
```

To inspect labels before staging imagery:

```bash
python scripts/satlas_bootstrap.py \
  --satlas-root /data/SatlasPretrain \
  --out satlas_catalog.json \
  --catalog-only
```

## Provenance and licensing

SatlasPretrain combines imagery and labels from several sources. Its documentation lists NAIP imagery as public domain and includes source-specific label/data licences such as ODbL and ODC-BY. Preserve provenance in derived datasets and review the official SatlasPretrain source/licensing notes before redistribution or commercial dataset publication.
