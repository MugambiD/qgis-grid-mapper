import importlib.util
import json
import shutil
import tempfile
import unittest
import sys
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "satlas_bootstrap.py"
spec = importlib.util.spec_from_file_location("satlas_bootstrap", SCRIPT)
sb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sb
spec.loader.exec_module(sb)


class SatlasBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="satlas_fixture_"))
        (self.tmp / "static" / "100_200").mkdir(parents=True)
        (self.tmp / "metadata").mkdir(parents=True)
        self.image_name = "m_fixture_20200101"
        vector = {
            "metadata": {"ImageName": self.image_name},
            "power_substation": [{
                "Geometry": {"Type": "polygon", "Polygon": [[
                    [700, 700], [1100, 700], [1100, 900], [700, 900], [700, 700]
                ]]},
                "Properties": {},
            }],
        }
        (self.tmp / "static" / "100_200" / "vector.json").write_text(json.dumps(vector))
        (self.tmp / "metadata" / "train_highres.json").write_text(json.dumps([[100, 200]]))
        (self.tmp / "metadata" / "test_highres.json").write_text("[]")
        for sx, sy in [(1, 1), (2, 1), (0, 0), (3, 3), (4, 4)]:
            col, row = 100 * 16 + sx, 200 * 16 + sy
            p = self.tmp / "naip" / self.image_name / "tci" / f"{col}_{row}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (512, 512), (20 * sx, 20 * sy, 50)).save(p)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_polygon_crossing_parent_children_is_clipped(self):
        samples = list(sb.iter_satlas_samples(
            self.tmp, negative_ratio=1, max_positive_chips=100,
            max_negative_chips=100, valid_pct=0,
        ))
        positives = [s for s in samples if s.positive]
        negatives = [s for s in samples if not s.positive]
        self.assertEqual(2, len(positives))
        self.assertEqual(2, len(negatives))
        for sample in positives:
            self.assertTrue(sample.polygons)
            for poly in sample.polygons:
                self.assertTrue(all(0 <= v <= 512 for v in poly))

    def test_missing_class_is_not_negative(self):
        d = self.tmp / "static" / "101_200"
        d.mkdir()
        (d / "vector.json").write_text(json.dumps({"metadata": {"ImageName": self.image_name}}))
        samples = list(sb.iter_satlas_samples(
            self.tmp, negative_ratio=10, max_positive_chips=100,
            max_negative_chips=100, valid_pct=0,
        ))
        self.assertTrue(all(s.parent_tile != (101, 200) for s in samples))

    def test_export_coco_keeps_hard_negatives(self):
        samples = list(sb.iter_satlas_samples(
            self.tmp, negative_ratio=1, max_positive_chips=100,
            max_negative_chips=100, valid_pct=0,
        ))
        out = self.tmp / "out"
        manifest = sb.export_coco(samples, out)
        coco = json.loads((out / "train" / "_annotations.coco.json").read_text())
        annotated = {a["image_id"] for a in coco["annotations"]}
        self.assertGreater(len(coco["images"]), len(annotated))
        self.assertEqual(2, manifest["positive_chips"])
        self.assertEqual(2, manifest["negative_chips"])


if __name__ == "__main__":
    unittest.main()
