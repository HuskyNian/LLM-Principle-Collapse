"""End-to-end random-search example using the public chemical_vae ZINC subset."""

import argparse
import csv
import hashlib
from pathlib import Path
import random
import tempfile

from benchmark import BenchmarkRun, load_task
from benchmark.report import generate_report
from benchmark.plotting import plot_run

ZINC_URL = "https://raw.githubusercontent.com/the-matter-lab/chemical_vae/main/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
DEFAULT_CACHE = Path.home() / ".cache/exllm/zinc250k.csv"


def read_smiles(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "smiles" not in reader.fieldnames:
            raise ValueError(f"{path}: expected a CSV with a 'smiles' column, not an HTML/error page")
        smiles = [row["smiles"].strip() for row in reader if row.get("smiles", "").strip()]
    if not smiles:
        raise ValueError(f"{path}: empty SMILES dataset")
    return smiles


def download_zinc(path):
    import requests
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ZINC subset to {path} (about 22 MB)...", flush=True)
    temporary = None
    try:
        with requests.get(ZINC_URL, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False) as stream:
                temporary = Path(stream.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
        read_smiles(temporary)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RandomZincOptimizer:
    """Shuffle dataset rows without replacement; scores do not steer proposals."""

    def __init__(self, smiles, seed):
        self.pool = list(smiles)
        random.Random(seed).shuffle(self.pool)
        self.cursor = 0
        self.best = None

    def propose(self, batch_size):
        batch = self.pool[self.cursor:self.cursor + batch_size]
        self.cursor += len(batch)
        return batch

    def observe(self, results):
        for result in results:
            if result["valid"] and (self.best is None or result["fitness"] > self.best):
                self.best = result["fitness"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zinc-csv", type=Path, help="Local CSV with a smiles column; skips downloading")
    parser.add_argument("--budget", type=int, default=200, help="Unique valid evaluations (200 for smoke test, 5000 for task budget)")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.budget < 50 or args.batch_size < 1:
        parser.error("--budget must be >= 50 and --batch-size must be positive")
    if (args.output / "metadata.json").exists():
        parser.error("Output already contains a run; choose a new --output directory")
    path = args.zinc_csv or DEFAULT_CACHE
    if not path.exists():
        if args.zinc_csv:
            parser.error(f"Local CSV not found: {path}")
        download_zinc(path)
    smiles = read_smiles(path)
    optimizer = RandomZincOptimizer(smiles, args.seed)
    task = load_task("molecules", seed=args.seed, protocol={"budget": args.budget},
                     sampling={"dataset": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "source_url": ZINC_URL if not args.zinc_csv else None,
                               "replacement": False, "initialization_attempts": min(100, args.budget)})
    run = BenchmarkRun(task, args.output, optimizer="Random ZINC")
    print(f"Loaded {len(smiles)} rows; seed={args.seed}, budget={args.budget}. Initializing molecular oracles...", flush=True)
    initial_attempts = min(100, args.budget)
    while run.budget_used < args.budget:
        phase = "initialization" if len(run.rows) < initial_attempts else "search"
        batch_size = min(args.batch_size, initial_attempts - len(run.rows)) if phase == "initialization" else args.batch_size
        candidates = optimizer.propose(batch_size)
        if not candidates:
            break
        results = run.evaluate(candidates, phase=phase)
        optimizer.observe(results)
        print(f"budget={run.budget_used}/{args.budget} attempts={len(run.rows)} best={optimizer.best}", flush=True)
    run.finish("budget" if run.budget_used == args.budget else "dataset_exhausted")
    diagnosis = generate_report(args.output, Path(__file__).with_name("no_principle_model.json"))
    plot_run(args.output, run.rows, diagnosis)
    print(f"{diagnosis['verdict']}; report: {args.output / 'report.md'}", flush=True)
    if args.budget != 5000:
        print("Short integration example, not a full-budget collapse experiment.", flush=True)


if __name__ == "__main__":
    main()
