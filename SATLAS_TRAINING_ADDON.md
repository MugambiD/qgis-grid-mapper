# Grid Mapper 1.5 training add-on – SatlasPretrain bootstrap

This source snapshot keeps the QGIS runtime metadata at **1.5.0**. The plugin runtime code is unchanged; the training toolchain has been extended with:

- `notebooks/GridMapper_RF-DETR_training.ipynb`: optional SatlasPretrain `power_substation` bootstrap, aerial augmentation and provenance metadata.
- `scripts/satlas_bootstrap.py`: standalone Satlas high-resolution polygon/chip converter.
- `docs/SATLAS_BOOTSTRAP.md`: setup, split and licensing guidance.
- `grid_mapper/tests/test_satlas_bootstrap.py`: synthetic regression tests for polygon clipping, hard negatives and COCO export.

The converter is conservative about negatives: a missing `power_substation` key is ignored, while an explicit empty list is eligible as a hard negative.
