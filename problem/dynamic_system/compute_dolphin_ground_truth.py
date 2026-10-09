#!/usr/bin/env python3
import argparse
import csv
import gzip
import itertools
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from evaluator import DEFAULT_DATA_DIR, SentinelEvaluator, format_candidate


_EVALUATOR = None


def _init_worker(network, dynamics, data_dir, subset_size, allow_weights):
    global _EVALUATOR
    _EVALUATOR = SentinelEvaluator(
        network=network,
        dynamics=dynamics,
        data_dir=data_dir,
        subset_size=subset_size,
        allow_weights=allow_weights,
    )


def _evaluate_chunk(combos):
    rows = []
    best = None
    for nodes in combos:
        result = _EVALUATOR.evaluate_subset(nodes)
        error = float(result["error"])
        row = {
            "nodes": " ".join(str(node) for node in result["nodes"]),
            "error": error,
            "score": float(result["score"]),
        }
        rows.append(row)
        if best is None or error < best["error"]:
            best = row
    return rows, best


def _chunked(iterable, chunk_size):
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustively compute ground truth for the dynamic-system dolphin sentinel task."
    )
    parser.add_argument("--network", default="dolphin")
    parser.add_argument("--dynamics", default="doublewell")
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "ground_truth"),
    )
    args = parser.parse_args()

    start = time.time()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = SentinelEvaluator(
        network=args.network,
        dynamics=args.dynamics,
        data_dir=data_dir,
        subset_size=args.subset_size,
        allow_weights=False,
    )
    total = math.comb(evaluator.num_nodes, evaluator.default_subset_size)
    csv_path = output_dir / f"ground_truth-{args.network}-{args.dynamics}-k{evaluator.default_subset_size}.csv.gz"
    json_path = output_dir / f"ground_truth-{args.network}-{args.dynamics}-k{evaluator.default_subset_size}.json"
    top_csv_path = output_dir / f"ground_truth-{args.network}-{args.dynamics}-k{evaluator.default_subset_size}-top{args.top_k}.csv"

    best = None
    top_rows = []
    processed = 0
    log_every = max(args.chunk_size, 50000)

    print(
        f"enumerating {total} combinations for "
        f"{args.network}/{args.dynamics}, n={evaluator.num_nodes}, k={evaluator.default_subset_size}, "
        f"workers={args.workers}"
    )
    print(f"writing full results to {csv_path}")

    combinations = itertools.combinations(range(1, evaluator.num_nodes + 1), evaluator.default_subset_size)
    with gzip.open(csv_path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["nodes", "error", "score"])
        writer.writeheader()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.network, args.dynamics, str(data_dir), args.subset_size, False),
        ) as executor:
            futures = [
                executor.submit(_evaluate_chunk, chunk)
                for chunk in _chunked(combinations, args.chunk_size)
            ]
            for future in as_completed(futures):
                rows, chunk_best = future.result()
                writer.writerows(rows)
                processed += len(rows)
                if chunk_best is not None and (best is None or chunk_best["error"] < best["error"]):
                    best = chunk_best
                if args.top_k > 0:
                    top_rows.extend(rows)
                    top_rows = sorted(top_rows, key=lambda row: row["error"])[: args.top_k]
                if processed == total or processed % log_every < args.chunk_size:
                    print(
                        f"processed {processed}/{total} | "
                        f"current best error={best['error']:.12g} nodes=[{best['nodes']}]"
                    )

    best_nodes = [int(token) for token in best["nodes"].split()]
    summary = {
        "network": args.network,
        "dynamics": args.dynamics,
        "num_nodes": evaluator.num_nodes,
        "subset_size": evaluator.default_subset_size,
        "total_combinations": total,
        "best_error": best["error"],
        "best_score": best["score"],
        "best_nodes": best_nodes,
        "best_candidate": format_candidate(best_nodes),
        "all_results_csv_gz": str(csv_path),
        "top_results_csv": str(top_csv_path),
        "top_results": [
            {
                "rank": rank,
                "nodes": [int(token) for token in row["nodes"].split()],
                "error": row["error"],
                "score": row["score"],
            }
            for rank, row in enumerate(top_rows, start=1)
        ],
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "running_time_seconds": time.time() - start,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.top_k > 0:
        with top_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "nodes", "error", "score"])
            writer.writeheader()
            for rank, row in enumerate(top_rows, start=1):
                writer.writerow({"rank": rank, **row})

    print(f"best: error={summary['best_error']:.12g}, nodes={summary['best_nodes']}")
    if args.top_k > 0:
        print(f"top {args.top_k} saved to {top_csv_path}")
    print(f"summary saved to {json_path}")


if __name__ == "__main__":
    main()
