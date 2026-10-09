import tempfile
from pathlib import Path
import unittest

from benchmark.examples.random_zinc import RandomZincOptimizer, read_smiles


class RandomZincTests(unittest.TestCase):
    def test_seeded_sampling_without_replacement(self):
        pool = [str(i) for i in range(20)]
        a, b = RandomZincOptimizer(pool, 42), RandomZincOptimizer(pool, 42)
        self.assertEqual(a.propose(20), b.propose(20))
        self.assertEqual(a.propose(1), [])
        self.assertEqual(set(a.pool), set(pool))

    def test_csv_and_bad_download_detection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "zinc.csv"
            path.write_text('smiles,logP\nCCO,1\n"c1ccccc1",2\n')
            self.assertEqual(read_smiles(path), ["CCO", "c1ccccc1"])
            path.write_text("<html><h1>403 Forbidden</h1></html>")
            with self.assertRaisesRegex(ValueError, "HTML/error page"):
                read_smiles(path)


if __name__ == "__main__":
    unittest.main()
