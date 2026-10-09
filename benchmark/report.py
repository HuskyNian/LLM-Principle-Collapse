"""Reports consume computed metrics, never the other way around."""

from collections import Counter
import csv
import json
from pathlib import Path
import statistics

from .diagnose import diagnose
from .recorder import read_rows, write_json


def generate_report(directory, audit_path=None, labels_path=None):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    rows = read_rows(directory / "attempts.jsonl")
    audit = json.loads(Path(audit_path).read_text()) if audit_path else None
    labels = json.loads(Path(labels_path).read_text()) if labels_path else None
    result = diagnose(rows, meta["protocol"], audit, labels)
    result.update({k: meta[k] for k in ("optimizer", "task", "seed")})
    result["termination"] = json.loads((directory / "termination.json").read_text()) if (directory / "termination.json").exists() else {"reason": "not recorded; provisional running trace"}
    write_json(directory / "diagnosis.json", result)
    lines = [f"# {meta['optimizer']} / {meta['task']} / seed {meta['seed']}", "",
             f"**{result['verdict']}**", "", "## Evidence", "",
             "```json", json.dumps(result.get("evidence", {"reason": result.get("reason")}), indent=2), "```", "",
             "## Selected Diagnostic Set", "", "| Metric | Early | Late |", "|---|---:|---:|"]
    for key, value in result.get("late", {}).items():
        lines.append(f"| {key} | {result['early'].get(key)} | {value} |")
    lines += ["", "## Attempt Windows", "", "| Window | Attempts | Uniqueness | Validity |", "|---|---:|---:|---:|"]
    for key, values in result["attempt_metrics"].items():
        lines.append(f"| {key} | {values['count']} | {values['uniqueness']} | {values['validity']} |")
    lines += ["", "## Plateau", "", "```json", json.dumps(result.get("plateau"), indent=2), "```", "",
              "Diagnostic sets are top selections",
              "Early is selected after recorded initialization, or the first complete diagnostic set if initialization is absent/insufficient; attempt windows are first/last 1000 attempts (overlap is possible).",
              f"Termination: {result['termination']['reason']}. No synthetic evaluations are added.", ""]
    (directory / "report.md").write_text("\n".join(lines))
    return result


def summarize(directories, output):
    results = [json.loads((Path(d) / "diagnosis.json").read_text()) for d in directories]
    # Do not silently mix tasks, algorithms, or different diagnostic protocols.
    groups = {}
    for r in results:
        key = (r["task"], r["optimizer"], json.dumps(r["protocol"], sort_keys=True))
        groups.setdefault(key, []).append(r)
    summary = []
    for (task, optimizer, protocol), runs in groups.items():
        if len({r["seed"] for r in runs}) != len(runs):
            raise ValueError(f"Duplicate seeds for {task}/{optimizer}")
        counts = Counter(r["verdict"] for r in runs)
        assessable = len(runs) - counts["Unassessable"]
        record = {"task": task, "optimizer": optimizer, "runs": len(runs), "assessable": assessable,
                  **{name: counts[name] for name in ("Strong support", "Limited support", "Not established", "Unassessable")},
                  "support_fraction": (counts["Strong support"] + counts["Limited support"]) / assessable if assessable else None,
                  "strong_fraction": counts["Strong support"] / assessable if assessable else None}
        for metric in ("top1", "top10", "n_eff"):
            values = [r["late"][metric] for r in runs if r.get("late", {}).get(metric) is not None]
            record[f"{metric}_mean"] = statistics.mean(values) if values else None
            record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        summary.append(record)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
    write_json(path.with_suffix(".protocols.json"), [json.loads(key[2]) for key in groups])
    return summary
