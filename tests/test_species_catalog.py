import hashlib
import tempfile
import unittest
from pathlib import Path

from finframe.database import Database
from finframe.species_catalog import MASTER_SPECIES, MASTER_SPECIES_NAMES


class SpeciesCatalogTests(unittest.TestCase):
    def test_catalog_matches_first_master_sheet_headers(self):
        digest = hashlib.sha256("\n".join(MASTER_SPECIES_NAMES).encode()).hexdigest()

        self.assertEqual(len(MASTER_SPECIES_NAMES), 99)
        self.assertEqual(digest, "6ec818e455c6374caf3455abc6ee418e27de41e52d0b5c42a50551f43e3bfc3a")
        self.assertEqual(MASTER_SPECIES_NAMES[0], "Parma polylepis")
        self.assertEqual(MASTER_SPECIES_NAMES[-1], "Ostorhinchus doederleini")

    def test_catalog_codes_are_stable_and_unique(self):
        codes = [record[2] for record in MASTER_SPECIES]

        self.assertEqual(len(codes), 99)
        self.assertEqual(len(codes), len(set(codes)))
        self.assertTrue(all(code.startswith("MS") for code in codes))

    def test_new_database_contains_master_catalog_and_preserves_custom_species(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "finframe.sqlite3"
            db = Database(path)
            self.assertEqual(len(db.list_species()), 99)
            db.add_species("Custom fish", "Customus fishii", "CUSTOM", "#123456")

            reopened = Database(path)
            species = reopened.list_species()
            self.assertEqual(len(species), 100)
            self.assertIsNotNone(reopened.species_by_code("CUSTOM"))
            self.assertEqual(sum(item["code"].startswith("MS") for item in species), 99)


if __name__ == "__main__":
    unittest.main()
