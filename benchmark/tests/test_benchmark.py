import json
import hashlib
from itertools import combinations
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from benchmark import BenchmarkRun, BudgetExhausted, load_task
from benchmark.diagnose import diagnose, plateau, text_evidence, validate_trace
from benchmark.metrics import network_metrics, select, molecular_metrics
from benchmark.report import generate_report, summarize


class FakeTask:
    name, seed, options = "dolphin", 42, {}

    def __init__(self, metric="unique_valid", budget=3):
        self.protocol = dict(load_task("dolphin").protocol, budget=budget, budget_metric=metric, selection="fitness")

    def canonicalize(self, candidate):
        return load_task("dolphin").canonicalize(candidate)

    def evaluate_one(self, candidate, canonical):
        return {"valid": True, "fitness": -0.1, "objectives": {"error": 0.1}, "min_objectives": [0.1]}


def rows_for_sets(sets):
    return [{"attempt": i, "canonical": json.dumps(list(s)), "candidate": list(s),
             "valid": True, "duplicate": False, "budget_used": i,
             "fitness": -0.1, "objectives": {"error": 0.1}, "min_objectives": [0.1]}
            for i, s in enumerate(sets, 1)]


class RecordingTests(unittest.TestCase):
    def test_bundled_network_data_and_reference_scores(self):
        cases = [("dolphin", "doublewell", [25, 17, 32, 35], 0.0040707747779881),
                 ("residence_hall", "genereg", [181, 111, 138, 166, 137], 2.38175444209538e-7)]
        for name, dynamics, nodes, expected in cases:
            with self.subTest(task=name):
                task = load_task(name)
                filename = f"fullstate-{name}-{dynamics}.npy"
                manifest = json.loads((task.data_dir / "manifest.json").read_text())
                digest = hashlib.sha256((task.data_dir / "fullstates" / filename).read_bytes()).hexdigest()
                self.assertEqual(digest, manifest[filename]["sha256"])
                result = task.evaluate_one(nodes, task.canonicalize(nodes))
                self.assertAlmostEqual(result["objectives"]["error"], expected, delta=expected * 1e-10)

    def test_real_network_adapter_with_synthetic_data(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "fullstates").mkdir()
            fullstate = np.arange(1, 125, dtype=float).reshape(2, 62)
            np.save(root / "fullstates/fullstate-dolphin-doublewell.npy", fullstate)
            task = load_task("dolphin", data_dir=root)
            run = BenchmarkRun(task, root / "run")
            result = run.evaluate([[1, 2, 3, 4], [4, 3, 2, 1]])
            expected = ((fullstate[:, :4].mean(axis=1) - fullstate.mean(axis=1)) ** 2).sum() / fullstate.mean(axis=1).sum()
            self.assertAlmostEqual(result[0]["objectives"]["error"], expected)
            self.assertTrue(result[1]["duplicate"])

    def test_duplicates_invalid_resume_and_budget(self):
        with tempfile.TemporaryDirectory() as d:
            run = BenchmarkRun(FakeTask(), d)
            result = run.evaluate([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 2, 3]])
            self.assertEqual(run.budget_used, 1)
            self.assertTrue(result[1]["duplicate"])
            self.assertFalse(result[2]["valid"])
            resumed = BenchmarkRun(FakeTask(), d, resume=True)
            self.assertTrue(resumed.evaluate([[4, 1, 2, 3]])[0]["duplicate"])
            result = resumed.evaluate([[1, 2, 3, 5], [1, 2, 3, 6], [1, 2, 3, 7]])
            self.assertEqual(len(result), 2)
            with self.assertRaises(BudgetExhausted):
                resumed.evaluate([[1, 2, 3, 8]])
            validate_trace(resumed.rows, resumed.task.protocol)

    def test_attempt_budget_counts_invalid_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            run = BenchmarkRun(FakeTask("attempts"), d)
            run.evaluate([[1, 2, 3, 4], [4, 3, 2, 1], [0, 1, 2, 3]])
            self.assertEqual(run.budget_used, 3)

    def test_backend_errors_not_invalid(self):
        task = FakeTask()
        def fail(*args):
            raise RuntimeError("oracle unavailable")
        task.evaluate_one = fail
        with tempfile.TemporaryDirectory() as d:
            run = BenchmarkRun(task, d)
            with self.assertRaises(RuntimeError):
                run.evaluate([[1, 2, 3, 4]])
            self.assertEqual(run.rows, [])
            self.assertEqual(run.budget_used, 0)

    def test_safe_parsing(self):
        task = load_task("dolphin")
        for value in ([1.5, 2, 3, 4], [True, 2, 3, 4], "nodes = __import__('os').getcwd()"):
            with self.assertRaises((ValueError, TypeError)):
                task.canonicalize(value)


class MetricTests(unittest.TestCase):
    def test_coverage_upper_bound_and_effective_nodes(self):
        rows = rows_for_sets(list(combinations(range(1, 6), 4)))
        metrics = network_metrics(rows, 4)
        self.assertAlmostEqual(metrics["normalized_top5_coverage"], 1)
        self.assertAlmostEqual(metrics["n_eff"], 5)
        sets = [(1, 2, 3, 4, 5)] + [(1, 2, 3, 4, n) for n in range(6, 55)]
        self.assertAlmostEqual(network_metrics(rows_for_sets(sets), 5)["normalized_top5_coverage"], 1)

    def test_selection_reuses_existing(self):
        from algorithm.base import Item
        from model.util import nsga2_so_selection
        rows = rows_for_sets(list(combinations(range(1, 9), 4)))
        for i, row in enumerate(rows):
            row["fitness"] = i / 100
            row["min_objectives"] = [i / 100, 1 - i / 100]
        items = []
        for r in rows:
            item = Item(r["canonical"], [])
            item.total, item.scores = r["fitness"], r["min_objectives"]
            items.append(item)
        expected = [x.value for x in nsga2_so_selection(items, 50, 0.5)]
        self.assertEqual([r["canonical"] for r in select(rows, 50, "nsga2_so")], expected)

    def test_molecular_metrics(self):
        rows = [{"canonical": s, "min_objectives": [0.2] * 5} for s in ("c1ccccc1", "Cc1ccccc1", "Oc1ccccc1")]
        result = molecular_metrics(rows)
        self.assertEqual(result["unique_scaffolds"], 1)
        self.assertEqual(result["n_eff"], 1)
        self.assertGreater(result["diversity"], 0)
        self.assertAlmostEqual(result["hv"], 0.9 ** 5)


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.rows = rows_for_sets(list(combinations(range(1, 9), 4)))
        self.protocol = dict(load_task("dolphin").protocol, selection="fitness")

    def test_strong_and_no_gap(self):
        self.assertEqual(diagnose(self.rows, self.protocol)["verdict"], "Strong support")
        for r in self.rows:
            r["fitness"], r["objectives"]["error"] = -0.0041, 0.0041
        self.assertEqual(diagnose(self.rows, self.protocol)["verdict"], "Not established")

    def test_missing_audit_not_treated_as_no_collapse(self):
        p = dict(self.protocol, narrowing_max=1)
        self.assertEqual(diagnose(self.rows, p)["verdict"], "Unassessable")
        self.assertEqual(diagnose(self.rows, p, {"explicit_model_covers_search": False})["verdict"], "Limited support")

    def test_incomplete_trace_rejected(self):
        with self.assertRaises(ValueError):
            diagnose(self.rows[1:], self.protocol)
        self.assertEqual(diagnose(self.rows[:5], self.protocol)["verdict"], "Unassessable")

    def test_permuted_set_cannot_fake_novelty_on_import(self):
        rows = rows_for_sets([(1, 2, 3, 4), (4, 3, 2, 1)])
        with self.assertRaises(ValueError):
            validate_trace(rows, self.protocol)

    def test_text_thresholds_and_audit_window(self):
        audit = {"explicit_model_covers_search": True, "attempt_range": [1, 1000],
                 "dominant_route_share": 0.5, "minor_updates": 5, "major_updates": 0,
                 "new_directions": 0, "reviewer": "test", "evidence": ["principles.jsonl:1-10"]}
        self.assertTrue(text_evidence(audit, [{}] * 1000, self.protocol)["narrowing"])
        audit["minor_updates"] = 6
        self.assertFalse(text_evidence(audit, [{}] * 1000, self.protocol)["narrowing"])
        self.assertIsNone(text_evidence(audit, [{}] * 900, self.protocol)["narrowing"])

    def test_plateau_uses_total_future_improvement(self):
        rows = rows_for_sets([(1, 2, 3, i) for i in range(4, 8)])
        for r, value in zip(rows, [4.0, 4.006, 4.012, 4.018]):
            r["fitness"] = value
        p = load_task("molecules").protocol
        self.assertEqual(plateau(rows, p)["onset_budget"], 3)

    def test_report_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            task = FakeTask(budget=3100)
            run = BenchmarkRun(task, Path(d) / "run")
            run.evaluate([r["candidate"] for r in self.rows])
            run.finish("test")
            result = generate_report(run.path)
            self.assertEqual(result["verdict"], "Strong support")
            summary = summarize([run.path], Path(d) / "summary.csv")
            self.assertEqual(summary[0]["strong_fraction"], 1)
            self.assertIsNone(summary[0]["top1_std"])

    def test_cli_with_figures(self):
        with tempfile.TemporaryDirectory() as d:
            run = BenchmarkRun(FakeTask(budget=3100), d)
            run.evaluate([r["candidate"] for r in self.rows])
            run.finish("test")
            process = subprocess.run([sys.executable, "-m", "benchmark", "report", d, "--plots"],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            for name in ("report.md", "diagnosis.json", "convergence.png", "early_late.png"):
                self.assertGreater((Path(d) / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
