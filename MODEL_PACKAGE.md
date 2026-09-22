# Grid Mapper model package metadata

A safely promotable model ZIP should contain the inference model and a JSON sidecar. The RF-DETR notebook writes this automatically.

Example:

```json
{
  "model_id": "gridmapper_detect_medium_20260922_1200",
  "arch": "rfdetr",
  "task": "detect",
  "primary_metric": "f1",
  "metrics": {
    "precision": 0.91,
    "recall": 0.87,
    "f1": 0.89
  },
  "geography_metrics": {
    "Kenya": {"precision": 0.92, "recall": 0.90, "f1": 0.91},
    "Zambia": {"precision": 0.88, "recall": 0.86, "f1": 0.87}
  }
}
```

The registry compares the candidate and active model using the declared `primary_metric`. If absent, it looks for common names such as `f1`, `mAP`, `map50` or `AP50`.
