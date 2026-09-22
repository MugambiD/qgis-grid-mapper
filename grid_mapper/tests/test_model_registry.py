import json
import os
import tempfile
import unittest
import zipfile

from grid_mapper.core.model_registry import (
    active_model_path, promotion_decision, register_candidate,
)


class ModelRegistryTests(unittest.TestCase):
    def test_rejects_unscored_candidate(self):
        d = promotion_decision({}, {})
        self.assertFalse(d["passed"])

    def test_rejects_geo_regression(self):
        old = {"primary_metric":"f1", "metrics":{"f1":0.80},
               "geography_metrics":{"Kenya":{"f1":0.82}, "Zambia":{"f1":0.75}}}
        new = {"primary_metric":"f1", "metrics":{"f1":0.83},
               "geography_metrics":{"Kenya":{"f1":0.78}, "Zambia":{"f1":0.84}}}
        d = promotion_decision(old, new, min_gain=0.005, max_geo_regression=0.03)
        self.assertFalse(d["passed"])
        self.assertIn("Kenya", d["geo_regressions"])

    def test_rejects_missing_previous_geography(self):
        old = {"primary_metric":"f1", "metrics":{"f1":0.80},
               "geography_metrics":{"Kenya":{"f1":0.80}, "Zambia":{"f1":0.78}}}
        new = {"primary_metric":"f1", "metrics":{"f1":0.85},
               "geography_metrics":{"Kenya":{"f1":0.85}}}
        d = promotion_decision(old, new)
        self.assertFalse(d["passed"])
        self.assertEqual(d["missing_geographies"], ["Zambia"])

    def test_promotes_better_candidate_and_preserves_active_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, "good.zip")
            bad = os.path.join(td, "bad.zip")
            def make(path, model_id, f1):
                meta={"model_id":model_id, "primary_metric":"f1", "metrics":{"f1":f1},
                      "geography_metrics":{"Kenya":{"f1":f1}}}
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("model.json", json.dumps(meta))
                    z.writestr("model.onnx", b"stub")
            make(good, "good", 0.80)
            make(bad, "bad", 0.70)
            _, promoted, _ = register_candidate(os.path.join(td,"reg"), good)
            self.assertTrue(promoted)
            active = active_model_path(os.path.join(td,"reg"))
            self.assertTrue(active and active.endswith("good.zip"))
            _, promoted2, _ = register_candidate(os.path.join(td,"reg"), bad)
            self.assertFalse(promoted2)
            self.assertEqual(active_model_path(os.path.join(td,"reg")), active)


if __name__ == "__main__":
    unittest.main()
