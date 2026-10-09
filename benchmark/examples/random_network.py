"""Interface smoke example, NOT a competitive baseline or published experiment."""

import argparse
import random

from benchmark import BenchmarkRun, load_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["dolphin", "residence_hall"], default="dolphin")
    parser.add_argument("--data-dir", help="Override the bundled network data directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    task = load_task(args.task, seed=args.seed, data_dir=args.data_dir)
    run = BenchmarkRun(task, args.output, optimizer="random-smoke")
    rng = random.Random(args.seed)
    for _ in range(args.steps):
        if run.budget_used >= task.protocol["budget"]:
            break
        run.evaluate([rng.sample(range(1, task.protocol["num_nodes"] + 1), task.protocol["subset_size"])])
    run.finish("smoke_test_only")


if __name__ == "__main__":
    main()
