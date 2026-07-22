import tempfile
import unittest
from pathlib import Path

from finframe.main_window import discover_image_files


class ImageDiscoveryTests(unittest.TestCase):
    def test_folder_discovery_is_recursive_sorted_and_case_insensitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nested"
            nested.mkdir()
            (root / "second.PNG").touch()
            (nested / "first.jpg").touch()
            (nested / "notes.txt").touch()

            discovered = discover_image_files(root)

            self.assertEqual({path.name for path in discovered}, {"first.jpg", "second.PNG"})
            self.assertEqual(discovered, sorted(discovered, key=lambda path: str(path).casefold()))


if __name__ == "__main__":
    unittest.main()
