from __future__ import annotations

import unittest

from scripts.check_public_release import check_content


class PublicReleaseTests(unittest.TestCase):
    def test_text_disguised_as_data_artifact_is_rejected(self) -> None:
        self.assertIn("non-source artifact type", check_content("samples.geojson", b"{}"))
        self.assertIn(
            "controlled data/output/dependency directory",
            check_content("data/observations.json", b"{}"),
        )

    def test_binary_disguised_as_source_is_rejected(self) -> None:
        self.assertIn("binary content", check_content("configs/map.json", b"\x00binary"))

    def test_synthetic_config_is_allowed(self) -> None:
        self.assertEqual(
            check_content("configs/synthetic.example.json", b'{"synthetic":true,"x":0,"y":0}'),
            [],
        )

    def test_recognizable_secret_is_rejected_without_reporting_value(self) -> None:
        synthetic_token = "gh" + "p_" + "x" * 36
        result = check_content("config.py", synthetic_token.encode())
        self.assertIn("GitHub credential", result)
        self.assertNotIn(synthetic_token, " ".join(result))

    def test_symlinks_and_path_traversal_are_rejected(self) -> None:
        self.assertTrue(check_content("linked.py", b"", "120000"))
        self.assertIn("non-relative path", check_content("../private.py", b""))


if __name__ == "__main__":
    unittest.main()
