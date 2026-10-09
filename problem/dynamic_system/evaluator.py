import ast
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "python_data"
DEFAULT_NETWORK = "dolphin"
DEFAULT_DYNAMICS = "doublewell"


def generate_initial_population(config, seed=42):
    network = config.get("network", DEFAULT_NETWORK) if config else DEFAULT_NETWORK
    dynamics = config.get("dynamics", DEFAULT_DYNAMICS) if config else DEFAULT_DYNAMICS
    data_dir = Path(config.get("data_dir", DEFAULT_DATA_DIR)) if config else DEFAULT_DATA_DIR
    sample_size = int(config.get("initial_samples", 50)) if config else 50

    evaluator = SentinelEvaluator(network=network, dynamics=dynamics, data_dir=data_dir)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(sample_size):
        nodes = sorted(rng.choice(np.arange(1, evaluator.num_nodes + 1), evaluator.default_subset_size, replace=False))
        samples.append(format_candidate(nodes))
    return samples


def format_candidate(nodes, weights=None):
    if weights is None:
        return f"nodes = {list(map(int, nodes))}"
    return f"nodes = {list(map(int, nodes))}\nweights = {list(map(float, weights))}"


def parse_candidate(value):
    text = str(value).strip()
    if not text:
        raise ValueError("empty candidate")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj.get("nodes"), obj.get("weights")
        if isinstance(obj, list):
            return obj, None
    except json.JSONDecodeError:
        pass

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj.get("nodes"), obj.get("weights")
        if isinstance(obj, (list, tuple)):
            return obj, None
    except (SyntaxError, ValueError):
        pass

    if "nodes" in text:
        scope = {}
        exec(text, {"__builtins__": {}}, scope)
        if "nodes" in scope:
            return scope["nodes"], scope.get("weights")

    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError("candidate does not contain node ids")
    return [int(token) for token in tokens], None


class SentinelEvaluator:
    def __init__(
        self,
        network=DEFAULT_NETWORK,
        dynamics=DEFAULT_DYNAMICS,
        data_dir=DEFAULT_DATA_DIR,
        subset_size=None,
        allow_weights=True,
    ):
        self.network = network
        self.dynamics = dynamics
        self.data_dir = Path(data_dir)
        self.allow_weights = allow_weights
        self.fullstate = self._load_fullstate()
        self.num_conditions, self.num_nodes = self.fullstate.shape
        self.default_subset_size = subset_size or int(np.floor(np.log(self.num_nodes)))
        self.full_average = self.fullstate.mean(axis=1)
        self.denominator = float(self.full_average.sum())
        if self.denominator <= 0:
            raise ValueError("full-state average denominator must be positive")

    def _load_fullstate(self):
        path = self.data_dir / "fullstates" / f"fullstate-{self.network}-{self.dynamics}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing converted fullstate: {path}. "
                "Run convert_rds_to_python.R first, for example: "
                "Rscript problem/dynamic_system/convert_rds_to_python.R "
                "--network dolphin,residence_hall --dynamics doublewell,SIS,genereg"
            )
        return np.load(path)

    def validate_nodes(self, nodes):
        arr = np.asarray(nodes, dtype=int)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("nodes must be a non-empty 1D list")
        if arr.size != self.default_subset_size:
            raise ValueError(f"nodes must contain exactly {self.default_subset_size} unique node ids")
        if len(set(arr.tolist())) != arr.size:
            raise ValueError("nodes must be unique")
        if arr.min() < 1 or arr.max() > self.num_nodes:
            raise ValueError(f"node ids must be in [1, {self.num_nodes}]")
        return arr

    def validate_weights(self, weights, expected_size):
        if weights is None:
            return None
        if not self.allow_weights:
            raise ValueError("weighted candidates are disabled")
        arr = np.asarray(weights, dtype=float)
        if arr.ndim != 1 or arr.size != expected_size:
            raise ValueError("weights must match the number of nodes")
        if np.any(arr < 0):
            raise ValueError("weights must be non-negative")
        total = arr.sum()
        if total <= 0:
            raise ValueError("weights must have positive sum")
        return arr / total

    def evaluate_subset(self, nodes, weights=None):
        nodes = self.validate_nodes(nodes)
        weights = self.validate_weights(weights, len(nodes))
        selected = self.fullstate[:, nodes - 1]
        if weights is None:
            subset_average = selected.mean(axis=1)
        else:
            subset_average = selected @ weights
        error = float(np.sum((subset_average - self.full_average) ** 2) / self.denominator)
        score = -error
        return {
            "error": error,
            "score": score,
            "nodes": nodes.tolist(),
            "weights": None if weights is None else weights.tolist(),
        }


class RewardingSystem:
    def __init__(self, config=None):
        self.config = config or {}
        self.evaluator = SentinelEvaluator(
            network=self.config.get("network", DEFAULT_NETWORK),
            dynamics=self.config.get("dynamics", DEFAULT_DYNAMICS),
            data_dir=self.config.get("data_dir", DEFAULT_DATA_DIR),
            subset_size=self.config.get("subset_size"),
            allow_weights=self.config.get("allow_weights", True),
        )
        self.history = set()

    def evaluate(self, items, mol_buffer=None, main_history=False):
        valid_items = []
        invalid_num = 0
        repeated_num = 0

        for item in items:
            try:
                nodes, weights = parse_candidate(item.value)
                result = self.evaluator.evaluate_subset(nodes, weights)
                key = tuple(result["nodes"])
                if key in self.history:
                    repeated_num += 1
                    continue
                self.history.add(key)

                item.value = format_candidate(result["nodes"], result["weights"])
                item.assign_results(
                    {
                        "original_results": {"error": result["error"]},
                        "transformed_results": {"error": result["error"]},
                        "overall_score": result["score"],
                    }
                )
                valid_items.append(item)
            except Exception as exc:
                invalid_num += 1
                print(f"dynamic_system evaluation error: {exc}")

        return valid_items, {"invalid_num": invalid_num, "repeated_num": repeated_num}


def _read_nodeset_rows(data_dir, network, dynamics, weighted=False):
    suffix = "_w" if weighted else ""
    path = data_dir / "nodesets" / f"nodeset-{network}-{dynamics}{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing converted nodeset file: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _parse_number_list(text, cast=float):
    text = (text or "").strip()
    if not text:
        return None
    return [cast(token) for token in text.split()]


def _best_nodeset(data_dir, network, dynamics, category="opt", weighted=False):
    rows = [row for row in _read_nodeset_rows(data_dir, network, dynamics, weighted) if row["category"] == category]
    if not rows:
        raise ValueError(f"No rows for category={category!r}, weighted={weighted}")
    best = min(rows, key=lambda row: float(row["error"]))
    return {
        "stored_error": float(best["error"]),
        "nodes": _parse_number_list(best["nodes"], int),
        "weights": _parse_number_list(best["weights"], float),
        "run": int(best["run"]),
    }


def _demo_case(network, dynamics, weighted=False):
    evaluator = SentinelEvaluator(network=network, dynamics=dynamics, data_dir=DEFAULT_DATA_DIR)
    best = _best_nodeset(DEFAULT_DATA_DIR, network, dynamics, category="opt", weighted=weighted)
    result = evaluator.evaluate_subset(best["nodes"], best["weights"])
    diff = abs(result["error"] - best["stored_error"])
    print(
        f"{network}/{dynamics}/weighted={weighted}: "
        f"stored={best['stored_error']:.12g} computed={result['error']:.12g} "
        f"diff={diff:.3g} nodes={best['nodes']}"
    )
    return diff


if __name__ == "__main__":
    print("Testing converted dynamic-system evaluator against stored author results.")
    diffs = [
        _demo_case("dolphin", "doublewell", weighted=False),
    ]
    try:
        _demo_case("dolphin", "doublewell", weighted=True)
    except FileNotFoundError:
        pass
    if (DEFAULT_DATA_DIR / "fullstates" / "fullstate-dolphin-SIS.npy").exists():
        diffs.append(_demo_case("dolphin", "SIS", weighted=False))
    if (DEFAULT_DATA_DIR / "fullstates" / "fullstate-residence_hall-genereg.npy").exists():
        diffs.append(_demo_case("residence_hall", "genereg", weighted=False))
    max_diff = max(diffs)
    print(f"max_diff={max_diff:.3g}")
    if max_diff > 1e-10:
        raise SystemExit("Evaluator mismatch is larger than tolerance.")
