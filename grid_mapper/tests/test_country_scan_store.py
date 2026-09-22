import os
import tempfile
import unittest

from grid_mapper.core.country_scan_store import CountryScanStore


SEGMENTS = [
    {"segment_id": "r0000c0000", "row_idx": 0, "col_idx": 0, "west": 20, "south": -15, "east": 20.2, "north": -14.8},
    {"segment_id": "r0000c0001", "row_idx": 0, "col_idx": 1, "west": 20.2, "south": -15, "east": 20.4, "north": -14.8},
]


class CountryScanStoreTests(unittest.TestCase):
    def test_resume_and_independent_ai_queue(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "scan.sqlite")
            with CountryScanStore(path) as store:
                store.initialize({"signature": "abc", "place_name": "Test"}, SEGMENTS)
                self.assertEqual(len(store.next_segments()), 2)
                store.mark_running("r0000c0000")
                store.mark_done("r0000c0000", 7)
                self.assertEqual(store.summary()["osm_done"], 1)
                ai = store.next_ai_segments()
                self.assertEqual([r["segment_id"] for r in ai], ["r0000c0000"])
                store.mark_ai_running("r0000c0000")
                store.mark_ai_done("r0000c0000", 12, 2)
                s = store.summary()
                self.assertEqual(s["ai_done"], 1)
                self.assertEqual(s["ai_tiles"], 12)

            with CountryScanStore(path) as store:
                self.assertEqual(store.summary()["osm_done"], 1)
                self.assertEqual([r["segment_id"] for r in store.next_segments()], ["r0000c0001"])

    def test_interrupted_segments_become_retry(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "scan.sqlite")
            with CountryScanStore(path) as store:
                store.initialize({"signature": "abc"}, SEGMENTS)
                store.mark_running("r0000c0000")
            with CountryScanStore(path) as store:
                self.assertEqual(store.reset_interrupted(), 1)
                self.assertEqual(store.rows()[0]["status"], "retry")

    def test_signature_prevents_wrong_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            with CountryScanStore(os.path.join(td, "scan.sqlite")) as store:
                store.initialize({"signature": "abc"}, SEGMENTS)
                with self.assertRaises(ValueError):
                    store.initialize({"signature": "different"}, SEGMENTS)

    def test_asset_deduplication(self):
        with tempfile.TemporaryDirectory() as td:
            with CountryScanStore(os.path.join(td, "scan.sqlite")) as store:
                store.initialize({"signature": "abc"}, SEGMENTS)
                self.assertTrue(store.register_asset("osm:way:1:line", "line", "r0000c0000"))
                self.assertFalse(store.register_asset("osm:way:1:line", "line", "r0000c0001"))


if __name__ == "__main__":
    unittest.main()
