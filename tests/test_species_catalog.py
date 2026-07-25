import hashlib
import tempfile
import unittest
from pathlib import Path

from finframe.database import Database
from finframe.species_catalog import (
    MASTER_SPECIES,
    MASTER_SPECIES_COMMON_NAMES,
    MASTER_SPECIES_NAMES,
)


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

    def test_every_master_taxon_has_a_distinct_common_name(self):
        self.assertEqual(set(MASTER_SPECIES_COMMON_NAMES), set(MASTER_SPECIES_NAMES))
        self.assertTrue(all(name.strip() for name in MASTER_SPECIES_COMMON_NAMES.values()))
        self.assertTrue(
            all(
                common.casefold() != scientific.casefold()
                for common, scientific, _code, _color in MASTER_SPECIES
            )
        )
        self.assertEqual(MASTER_SPECIES_COMMON_NAMES["Parma polylepis"], "Banded Scalyfin")
        self.assertEqual(MASTER_SPECIES_COMMON_NAMES["Chromis norfolkensis"], "Norfolk Chromis")

    def test_new_database_contains_master_catalog_and_preserves_custom_species(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "finframe.sqlite3"
            db = Database(path)
            self.assertEqual(len(db.list_species()), 99)
            self.assertEqual(db.species_by_code("MSPARPOL")["common_name"], "Banded Scalyfin")
            db.add_species("Custom fish", "Customus fishii", "CUSTOM", "#123456")

            reopened = Database(path)
            species = reopened.list_species()
            self.assertEqual(len(species), 100)
            self.assertIsNotNone(reopened.species_by_code("CUSTOM"))
            self.assertEqual(sum(item["code"].startswith("MS") for item in species), 99)

    def test_legacy_master_names_are_upgraded_without_overwriting_local_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "finframe.sqlite3"
            db = Database(path)
            legacy_code = MASTER_SPECIES[0][2]
            custom_code = MASTER_SPECIES[1][2]
            with db.connect() as connection:
                connection.execute(
                    "UPDATE species SET common_name=scientific_name WHERE code=?",
                    (legacy_code,),
                )
                connection.execute(
                    "UPDATE species SET common_name='My local wrasse name' WHERE code=?",
                    (custom_code,),
                )

            reopened = Database(path)
            self.assertEqual(reopened.species_by_code(legacy_code)["common_name"], "Banded Scalyfin")
            self.assertEqual(
                reopened.species_by_code(custom_code)["common_name"],
                "My local wrasse name",
            )

    def test_custom_species_can_be_corrected_without_changing_its_dataset_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            db = Database(Path(temporary) / "finframe.sqlite3")
            created = db.add_species(
                "Typo fish",
                "Typous fishii",
                "USR-TYPOUS-FISHII",
                "#123456",
            )

            updated = db.update_species(
                created["id"],
                common_name="Corrected fish",
                scientific_name="Correctus fishii",
                color="#654321",
            )

            self.assertEqual(updated["common_name"], "Corrected fish")
            self.assertEqual(updated["scientific_name"], "Correctus fishii")
            self.assertEqual(updated["code"], "USR-TYPOUS-FISHII")
            self.assertEqual(updated["color"], "#654321")

            with self.assertRaisesRegex(ValueError, "already exists"):
                db.add_species(
                    "Duplicate fish",
                    "correctus FISHII",
                    "USR-DUPLICATE",
                    "#abcdef",
                )

    def test_project_import_merges_a_new_species_by_scientific_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            db = Database(Path(temporary) / "finframe.sqlite3")
            existing = db.add_species(
                "Local shared fish",
                "Communis studentii",
                "USR-COMMUNIS-STUDENTII",
                "#123456",
            )
            snapshot = {
                "schemaVersion": "2.0",
                "project": {"name": "Student contribution"},
                "species": [
                    {
                        "id": "source_species",
                        "common_name": "Other common name",
                        "scientific_name": "communis STUDENTII",
                        "code": "A-DIFFERENT-STUDENT-CODE",
                        "color": "#abcdef",
                    }
                ],
                "videos": [],
            }

            db.import_project_snapshot(snapshot)

            self.assertEqual(
                db.species_by_scientific_name("Communis studentii")["id"],
                existing["id"],
            )
            self.assertIsNone(db.species_by_code("A-DIFFERENT-STUDENT-CODE"))


if __name__ == "__main__":
    unittest.main()
