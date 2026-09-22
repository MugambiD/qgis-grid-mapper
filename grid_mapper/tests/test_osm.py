import unittest

from grid_mapper.core import osm


class OsmHelperTests(unittest.TestCase):
    def test_bbox_tiles_cover_expected_grid(self):
        place = {"bbox": (30.0, -18.0, 31.0, -17.5)}
        tiles = osm.bbox_tiles(place, 0.25)
        self.assertEqual(len(tiles), 8)
        self.assertEqual(tiles[0][2][0], 30.0)
        self.assertAlmostEqual(tiles[-1][2][2], 31.0)
        self.assertAlmostEqual(tiles[-1][2][3], -17.5)

    def test_structure_queries_are_separate(self):
        q_t = osm.build_power_node_bbox_query((30, -18, 31, -17), "tower")
        q_p = osm.build_power_node_bbox_query((30, -18, 31, -17), "pole")
        self.assertIn('["power"="tower"]', q_t)
        self.assertNotIn('["power"="pole"]', q_t)
        self.assertIn('["power"="pole"]', q_p)

    def test_classify_distinguishes_tower_and_pole(self):
        self.assertEqual(osm.classify({"tags": {"power": "tower"}}), "tower")
        self.assertEqual(osm.classify({"tags": {"power": "pole"}}), "pole")

    def test_core_query_does_not_include_structures_by_default(self):
        place = {"osm_type": None, "osm_id": 0, "bbox": (30, -18, 31, -17)}
        q = osm.build_query(place, True, True, True, False, False, True)
        self.assertIn('power"~"^(line|minor_line|cable)$', q)
        self.assertIn('power"="substation', q)
        self.assertNotIn('power"="tower', q)
        self.assertNotIn('power"="pole', q)


if __name__ == "__main__":
    unittest.main()
