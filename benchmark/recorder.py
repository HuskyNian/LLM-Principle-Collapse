"""Append-only, inspectable run artifacts. No pickle required."""

import json
import math
from pathlib import Path


class BudgetExhausted(RuntimeError):
    pass


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class BenchmarkRun:
    def __init__(self, task, output_dir, optimizer="custom", resume=False):
        self.task, self.path = task, Path(output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.trace = self.path / "attempts.jsonl"
        self.meta = {"schema_version": 1, "task": task.name, "seed": task.seed,
                     "optimizer": optimizer, "protocol": task.protocol, "options": task.options,
                     "data_dir": str(getattr(task, "data_dir", ""))}
        metadata = self.path / "metadata.json"
        if metadata.exists():
            if not resume:
                raise FileExistsError(f"Run exists: {self.path}; use resume=True")
            if json.loads(metadata.read_text()) != self.meta:
                raise ValueError("Resume metadata differs (task/seed/protocol/optimizer/options)")
        else:
            if resume:
                raise FileNotFoundError(metadata)
            write_json(metadata, self.meta)
        self.rows = read_rows(self.trace) if self.trace.exists() else []
        self.cache = {r["canonical"]: r for r in self.rows if r["valid"] and not r["duplicate"]}
        self.budget_used = self.rows[-1]["budget_used"] if self.rows else 0

    def evaluate(self, candidates, phase="search", principle_ids=None):
        if phase not in ("initialization", "search"):
            raise ValueError("phase must be initialization or search")
        result = []
        for candidate in candidates:
            json.dumps(candidate, allow_nan=False)
            if self.budget_used >= self.task.protocol["budget"]:
                if result:
                    break
                raise BudgetExhausted("Evaluation budget exhausted")
            canonical, duplicate = None, False
            try:
                canonical = self.task.canonicalize(candidate)
            except (ValueError, SyntaxError, TypeError, KeyError) as exc:
                evaluated = {"valid": False, "error": str(exc)}
            else:
                duplicate = canonical in self.cache
                if duplicate:
                    evaluated = {k: self.cache[canonical][k] for k in
                                 ("valid", "fitness", "objectives", "min_objectives")}
                else:
                    evaluated = self.task.evaluate_one(candidate, canonical)
            if evaluated["valid"] and not math.isfinite(evaluated["fitness"]):
                raise ValueError("Non-finite fitness")
            cost = 1 if self.task.protocol["budget_metric"] == "attempts" else int(evaluated["valid"] and not duplicate)
            self.budget_used += cost
            row = {"attempt": len(self.rows) + 1, "candidate": candidate, "canonical": canonical,
                   "duplicate": duplicate, "budget_used": self.budget_used, "phase": phase,
                   "principle_ids": principle_ids or [], **evaluated}
            encoded = json.dumps(row, allow_nan=False)
            with self.trace.open("a") as stream:
                stream.write(encoded + "\n")
            self.rows.append(row)
            if row["valid"] and not duplicate:
                self.cache[canonical] = row
            result.append(row)
        return result

    def record_principle(self, event):
        """Optional raw evidence. Semantic audit remains a separate reviewed artifact."""
        with (self.path / "principles.jsonl").open("a") as stream:
            stream.write(json.dumps({**event, "attempt": len(self.rows)}, allow_nan=False) + "\n")

    def finish(self, reason="budget"):
        write_json(self.path / "termination.json", {"reason": reason,
                   "budget_used": self.budget_used, "attempts": len(self.rows)})


def read_rows(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed trace at line {line_number}; restore before resuming") from exc
        rows.append(row)
    return rows
