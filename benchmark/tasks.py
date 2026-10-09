"""Lazy adapters: importing the benchmark does not load an LLM or chemistry stack."""

import ast
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NETWORK_DATA_DIR = ROOT / "data/dynamic_system"
GOALS = ["sa", "drd2", "qed", "gsk3b", "jnk3"]


class Task:
    def __init__(self, name, seed=42, data_dir=None, **options):
        self.name, self.seed, self.options = name, seed, options
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_NETWORK_DATA_DIR
        self._evaluator = None
        config = Path(__file__).parent / "configs" / f"{name}.json"
        if not config.exists():
            raise ValueError(f"Unknown task {name!r}")
        self.protocol = json.loads(config.read_text())
        self.protocol.update(options.get("protocol", {}))
        if self.protocol["task"] != name or self.protocol["version"] != "benchmark-v1":
            raise ValueError("Cannot override task identity or protocol version")
        if self.protocol["budget_metric"] not in ("unique_valid", "attempts"):
            raise ValueError("budget_metric must be unique_valid or attempts")
        if type(self.protocol["budget"]) is not int or self.protocol["budget"] <= 0:
            raise ValueError("budget must be a positive integer")
        if self.protocol["selection_size"] != (10 if name == "kernel" else 50):
            raise ValueError("This protocol's thresholds require 50 candidates (10 for kernels)")
        if self.protocol["selection"] not in ("nsga2_so", "fitness"):
            raise ValueError("selection must be nsga2_so or fitness")
        if name == "kernel" and not self.protocol.get("reference"):
            raise ValueError("Kernel requires protocol.reference (expert latency in ms).")

    def canonicalize(self, candidate):
        if self.name == "molecules":
            from rdkit import Chem
            mol = Chem.MolFromSmiles(str(candidate).strip())
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError("Invalid or empty SMILES")
            return Chem.MolToSmiles(mol, canonical=True)
        if self.name == "kernel":
            from problem.kernel.harness import canonical_key, extract_source
            source = extract_source(str(candidate))
            if not source.strip():
                raise ValueError("Empty kernel")
            return canonical_key(source)
        if isinstance(candidate, str):
            text = candidate.strip()
            if text.startswith("nodes ="):
                text = text.split("=", 1)[1].strip()
            candidate = ast.literal_eval(text)
        if isinstance(candidate, dict):
            if candidate.get("weights") is not None:
                raise ValueError("The published network tasks use unweighted node sets")
            candidate = candidate["nodes"]
        if not isinstance(candidate, (list, tuple)) or any(type(n) is not int for n in candidate):
            raise ValueError("Expected a list of integer node ids")
        k, n = self.protocol["subset_size"], self.protocol["num_nodes"]
        if len(candidate) != k or len(set(candidate)) != k or not all(1 <= x <= n for x in candidate):
            raise ValueError(f"Expected {k} distinct nodes in [1, {n}]")
        return json.dumps(sorted(candidate), separators=(",", ":"))

    def evaluate_one(self, candidate, canonical):
        # Infrastructure/oracle exceptions deliberately propagate, not invalid candidates.
        if self.name in ("dolphin", "residence_hall"):
            from problem.dynamic_system.evaluator import SentinelEvaluator
            if self._evaluator is None:
                self._evaluator = SentinelEvaluator(
                    network=self.name, dynamics=self.protocol["dynamics"],
                    data_dir=self.data_dir, allow_weights=False)
            result = self._evaluator.evaluate_subset(json.loads(canonical))
            return {"valid": True, "fitness": result["score"],
                    "objectives": {"error": result["error"]},
                    "min_objectives": [result["error"]]}
        from algorithm.base import Item
        if self._evaluator is None:
            if self.name == "molecules":
                from problem.molecules.evaluator import RewardingSystem
                config = {"goals": GOALS, "optimization_direction": ["min"] + ["max"] * 4}
            else:
                from problem.kernel.evaluator import RewardingSystem
                config = dict(self.options.get("evaluator_config", {}))
                config.setdefault("goals", ["latency"])
            self._evaluator = RewardingSystem(config=config)
        goals = GOALS if self.name == "molecules" else ["latency"]
        item = Item(canonical if self.name == "molecules" else str(candidate), goals)
        try:
            items, log = self._evaluator.evaluate([item], main_history=True)
        except Exception:
            # Legacy sanitizers can mutate their history before an oracle fails.
            self._evaluator = None
            raise
        if not items:
            if log.get("repeated_num"):
                raise RuntimeError("Adapter and evaluator disagree on duplicate identity")
            return {"valid": False, "error": "Evaluator rejected candidate"}
        item = items[0]
        fitness = float(item.total)
        objectives = {k: float(item.property[k]) for k in goals}
        scores = [float(v) for v in item.scores]
        if not math.isfinite(fitness) or not all(math.isfinite(v) for v in scores):
            return {"valid": False, "error": "Non-finite evaluation"}
        return {"valid": True, "fitness": fitness, "objectives": objectives, "min_objectives": scores}

    def initial_population(self):
        if self.name == "molecules":
            data = json.loads((ROOT / "data/data_goal5.json").read_text())
            key = f"random{self.seed - 41}"
            if key not in data:
                raise ValueError("Published molecular initialization supports seeds 42--46")
            return data[key]
        if self.name == "kernel":
            from problem.kernel.evaluator import generate_initial_population
            return generate_initial_population(self.options["evaluator_config"], self.seed)
        import numpy as np
        rng = np.random.default_rng(self.seed)
        return [sorted(map(int, rng.choice(range(1, self.protocol["num_nodes"] + 1),
                         self.protocol["subset_size"], replace=False))) for _ in range(50)]


def load_task(name, seed=42, **options):
    return Task(name, seed, **options)
