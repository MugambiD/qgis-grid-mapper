import configparser
import os
import unittest


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        root = os.path.dirname(os.path.dirname(__file__))
        self.plugin_root = root
        self.metadata = os.path.join(root, "metadata.txt")

    def test_required_release_files_exist(self):
        for name in ("metadata.txt", "__init__.py", "LICENSE", "README.md"):
            self.assertTrue(os.path.isfile(os.path.join(self.plugin_root, name)), name)

    def test_qgis4_metadata(self):
        cp = configparser.ConfigParser()
        cp.read(self.metadata, encoding="utf-8")
        g = cp["general"]
        self.assertEqual(g.get("version"), "1.5.1")
        self.assertEqual(g.get("qgisMaximumVersion"), "4.99")
        self.assertNotIn("supportsqt6", {k.lower() for k in g.keys()})
        self.assertEqual(g.get("experimental"), "False")

    def test_public_links_present(self):
        cp = configparser.ConfigParser()
        cp.read(self.metadata, encoding="utf-8")
        g = cp["general"]
        self.assertTrue(g.get("repository", "").startswith("https://"))
        self.assertTrue(g.get("tracker", "").startswith("https://"))
        self.assertTrue(g.get("homepage", "").startswith("https://"))


if __name__ == "__main__":
    unittest.main()
