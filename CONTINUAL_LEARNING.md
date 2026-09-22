# Grid Mapper continual learning

Grid Mapper 1.4 adds a **human-in-the-loop** learning cycle. It does not allow the model to label its own predictions as truth. A person confirms, rejects or corrects detections, and only that reviewed data enters the training store.

## 1. Review the useful detections first

Run **Prioritise detections for active-learning review**. The tool adds:

- `al_priority` — 0 to 1, highest for unreviewed detections closest to the chosen decision threshold;
- `al_reason` — why the feature was prioritised.

Sort `al_priority` descending. Very confident predictions can still be sampled occasionally, but the uncertain band is usually the best use of review time.

## 2. Record human feedback

The detection output already contains a `review` field. Recommended values are:

- `TP` / `confirmed` — correct substation;
- `FP` / `rejected` — false detection;
- `corrected` — correct object after you edited the geometry;
- `uncertain` — do not train on it yet.

If a true substation was missed, use the **Missed reference substations** output from the validation tool.

Run **Capture reviewed feedback for continual learning** with the imagery used for detection. It writes an append-only dataset folder:

```text
feedback_dataset/
  feedback.jsonl
  samples/
    <sample_id>.jpg
    <sample_id>.jgw
    <sample_id>.jpg.aux.xml
  snapshots/
```

TP/corrected/missed examples contain a substation annotation. FP examples are retained as hard-negative images with no annotation. The sample ID is deterministic enough to prevent duplicate capture of the same reviewed feature.

Each sample receives a permanent `train`, `valid` or `test` split using a stable hash. Adding later examples cannot reshuffle the old validation/test samples.

## 3. Export an immutable training snapshot

Run **Export continual-learning dataset snapshot**. It creates:

```text
snapshots/feedback_YYYYMMDD/
  train/
    images...
    _annotations.coco.json
  valid/
    images...
    _annotations.coco.json
  test/
    images...
    _annotations.coco.json
  dataset_manifest.json
```

The COCO image records retain Grid Mapper fields such as geography, source model and review status. Negative examples are valid COCO images with zero annotations.

Do not overwrite an old snapshot. Use a new name for each training round so results remain reproducible.

## 4. Retrain outside QGIS

Open `notebooks/GridMapper_RF-DETR_training.ipynb` from the source repository in Google Colab with a GPU runtime. The training notebook is not bundled in the official QGIS install ZIP.

- Put the newest snapshot in `GRIDMAPPER_SNAPSHOT_DIRS`.
- Keep the original thesis/Roboflow/older-chip sources enabled as a replay buffer.
- Optionally set `RESUME_CHECKPOINT` to the previous RF-DETR `.pth` checkpoint.
- Run all cells.

The notebook calculates a fixed-threshold precision/recall/F1 on the fixed test set and, where geography metadata exists, the same metrics by geography. It exports these metrics inside the model package JSON.

Training on only the newest geography can make that geography improve while older geographies get worse. Replaying old data and checking geography metrics are the two safeguards against that failure mode.

## 5. Register and promote a candidate

Run **Register / safely promote a candidate model** and select the new model ZIP.

Default gates:

- overall primary validation metric must improve by at least `0.005`;
- no shared geography may drop by more than `0.03`.

The supplied RF-DETR notebook sets `primary_metric` to `f1`. If a model package contains no recognised validation metric, it is not automatically promoted.

A passing candidate becomes the active local model. A failing candidate is still copied into the registry for audit, but the existing active model remains unchanged.

The registry is stored under the QGIS profile:

```text
<QGIS profile>/grid_mapper/model_registry/
  registry.json
  models/
    <model_id>/...
```

When **Detect substations (local model)** or **Map the grid** is run without selecting a model file, Grid Mapper uses the active registered model.

## 6. What still requires a GPU

QGIS can capture, version, validate and run ONNX inference on CPU. RF-DETR fine-tuning still needs a GPU in practice. The intended cycle is therefore **continuous data collection, periodic GPU retraining, gated deployment** rather than training continuously inside QGIS.
