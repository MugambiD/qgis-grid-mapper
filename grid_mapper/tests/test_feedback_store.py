import json
import os
import tempfile
import unittest

from grid_mapper.core.feedback_store import (
    FeedbackStore, active_learning_priority, export_snapshot,
    normalize_review, sample_id_for, stable_split,
)


class FeedbackStoreTests(unittest.TestCase):
    def test_review_normalisation(self):
        self.assertEqual(normalize_review("TP"), "positive")
        self.assertEqual(normalize_review("FP"), "negative")
        self.assertEqual(normalize_review("corrected"), "corrected")
        self.assertEqual(normalize_review("FN"), "missed")
        self.assertEqual(normalize_review("maybe"), "uncertain")

    def test_stable_split_does_not_change(self):
        vals = [stable_split("same-key", 10, 10) for _ in range(10)]
        self.assertEqual(len(set(vals)), 1)

    def test_active_learning_prefers_threshold(self):
        mid, _ = active_learning_priority(0.5, "unreviewed", 0.5)
        high, _ = active_learning_priority(0.99, "unreviewed", 0.5)
        done, _ = active_learning_priority(0.5, "TP", 0.5)
        self.assertGreater(mid, high)
        self.assertEqual(done, 0.0)

    def test_append_deduplicates_and_snapshot_keeps_negative(self):
        with tempfile.TemporaryDirectory() as td:
            store = FeedbackStore(td)
            for idx, status in enumerate(("positive", "negative")):
                sid = sample_id_for(idx, status)
                img_rel = f"samples/{sid}.jpg"
                with open(os.path.join(td, img_rel), "wb") as fh:
                    fh.write(b"fake-jpg")
                rec = {
                    "sample_id": sid, "status": status, "split": "train",
                    "geography": "Kenya", "image": img_rel, "chip_size": 750,
                    "annotations": ([{"class": "substation", "bbox": [10,20,30,40], "segmentation": []}]
                                    if status == "positive" else []),
                }
                self.assertTrue(store.append(rec))
                self.assertFalse(store.append(rec))
            out, summary = export_snapshot(td, "round1")
            with open(os.path.join(out, "train", "_annotations.coco.json"), encoding="utf-8") as fh:
                coco = json.load(fh)
            self.assertEqual(len(coco["images"]), 2)
            self.assertEqual(len(coco["annotations"]), 1)
            self.assertEqual(summary["total_records"], 2)


if __name__ == "__main__":
    unittest.main()
