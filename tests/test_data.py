"""Small offline checks for dataset restoration and analysis metrics."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

import download_data as D


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, name, content=b"data"):
        path = self.root / "bundle.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            item = tarfile.TarInfo(name)
            item.size = len(content)
            tar.addfile(item, io.BytesIO(content))
        return path

    def test_paths_cannot_escape_project(self):
        for name in ["../escape", "/absolute", "output/../../escape"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                D.safe_target(self.root / "project", name)

    def test_checked_writes_are_repeatable(self):
        data = b"data"
        digest = hashlib.sha256(data).hexdigest()
        D.checked_write(self.root, "results/a", data, digest)
        D.checked_write(self.root, "results/a", data, digest)
        self.assertEqual((self.root / "results/a").read_bytes(), data)

    def test_existing_different_results_are_preserved(self):
        (self.root / "scores.json").write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            D.checked_write(self.root, "scores.json", b"new")
        self.assertEqual((self.root / "scores.json").read_bytes(), b"existing")

    def test_checksum_failure_writes_nothing(self):
        with self.assertRaises(ValueError):
            D.checked_write(self.root, "bad", b"data", "incorrect")
        self.assertFalse((self.root / "bad").exists())

    def test_archive_restores_only_manifest_files(self):
        archive = self.archive("results/a")
        record = {"path": "results/a", "sha256": hashlib.sha256(b"data").hexdigest()}
        D.restore_archive(archive, self.root / "project", {"files": [record]})
        self.assertEqual((self.root / "project/results/a").read_bytes(), b"data")
        with self.assertRaises(ValueError):
            D.restore_archive(archive, self.root / "other", {"files": []})

    def test_missing_archive_members_fail(self):
        archive = self.archive("results/a")
        with self.assertRaises(ValueError):
            D.restore_archive(archive, self.root / "project", {"files": [
                {"path": "results/a", "sha256": hashlib.sha256(b"data").hexdigest()},
                {"path": "results/missing", "sha256": "missing"}]})

    def test_parquet_projection_preserves_samples_and_ignores_other_runs(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = [{"model_size": "m", "train_domain": "d", "id": "q1", "aligned": 10, "coherent": 90},
                {"model_size": "m", "train_domain": "d", "id": "q1", "aligned": None, "coherent": 80},
                {"model_size": "excluded", "train_domain": "d", "id": "q1", "aligned": 0, "coherent": 0}]
        path = self.root / "existing.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        D.restore_parquet(path, self.root / "project", ["output/m/d/scores.json"])
        restored = json.loads((self.root / "project/output/m/d/scores.json").read_text())
        self.assertEqual(restored, [{k: r[k] for k in ["id", "aligned", "coherent"]} for r in rows[:2]])
        self.assertFalse((self.root / "project/output/excluded").exists())


class AnalysisTests(unittest.TestCase):
    def test_reasoning_rate_preserves_all_scored_denominator(self):
        from emtransfer import metrics
        rows = [{"id": "bad_medical_advice_q1", "aligned": a, "coherent": c}
                for a, c in [(10, 90), (90, 90), (0, 10), (None, 100)]]
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                folder = Path("output/m/incorrect_math")
                folder.mkdir(parents=True)
                (folder / "scores.json").write_text(json.dumps(rows))
                folder = Path("output/m/base_model")
                folder.mkdir()
                (folder / "scores.json").write_text(json.dumps([
                    {"id": "bad_medical_advice_q1", "aligned": 90, "coherent": 100}]))
                adjusted = metrics.adjusted(metrics.load_model("m"))["incorrect_math"]["bad_medical_advice"]
                self.assertAlmostEqual(adjusted["mis"], 100 / 3)
                self.assertEqual(adjusted["align"], -40)
                self.assertAlmostEqual(adjusted["coh"], 200 / 3)
            finally:
                os.chdir(previous)

    def test_standard_metric_filters_null_and_incoherent_responses(self):
        from analysis.build_tables import aligned
        rows = [{"aligned": a, "coherent": c} for a, c in [(10, 50), (90, 100), (0, 49), (None, 100), (10, None)]]
        self.assertEqual(aligned(rows), [10, 90])


if __name__ == "__main__":
    unittest.main()
