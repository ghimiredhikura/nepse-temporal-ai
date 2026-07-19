"""Regression tests for the public release entry points and utilities."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

import main
from nepse_ai.utils import save_local_checkpoint, sha256_file


class HashingTests(unittest.TestCase):
    def test_streaming_sha256_matches_hashlib(self) -> None:
        payload = b"NEPSE temporal AI\n" * 10_000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(payload)
            self.assertEqual(
                sha256_file(path, chunk_size=257),
                hashlib.sha256(payload).hexdigest(),
            )

    def test_streaming_sha256_rejects_invalid_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            sha256_file(Path("unused"), chunk_size=0)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_write_is_complete_and_leaves_no_temporary_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.pt"
            digest = save_local_checkpoint(
                {"weights": torch.tensor([1.0, 2.0])},
                destination,
            )
            self.assertTrue(destination.is_file())
            self.assertEqual(digest, sha256_file(destination))
            self.assertEqual(
                list(destination.parent.glob(f".{destination.name}.*.tmp")),
                [],
            )


class ReleaseTests(unittest.TestCase):
    def test_headline_table_matches_locked_results(self) -> None:
        table = main.build_main_table()
        self.assertEqual(table.shape, (5, 7))
        temporal = table.loc[table["Model"].eq("Temporal GRU")].iloc[0]
        self.assertAlmostEqual(float(temporal["2024 PR-AUC"]), 0.3211575)
        self.assertAlmostEqual(float(temporal["2025 PR-AUC"]), 0.2679912)
        self.assertAlmostEqual(float(temporal["Pooled PR-AUC"]), 0.2951392)

    def test_processed_sample_matches_manifest(self) -> None:
        manifest = main.validate_sample()
        self.assertEqual(int(manifest["rows"]), 27_476)
        self.assertEqual(int(manifest["securities"]), 24)

    def test_headline_pdf_is_reproducible(self) -> None:
        main.build_main_figure()
        first = sha256_file(main.OUTPUT / "main_result_figure.pdf")
        main.build_main_figure()
        second = sha256_file(main.OUTPUT / "main_result_figure.pdf")
        self.assertEqual(first, second)

if __name__ == "__main__":
    unittest.main()
