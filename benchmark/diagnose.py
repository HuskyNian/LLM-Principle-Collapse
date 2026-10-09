"""Explicit evidence flags and three-way support, with missing evidence separate."""

import math
import json

from .metrics import snapshot, attempt_metrics


def validate_trace(rows, protocol):
    seen, budget = set(), 0
    for index, row in enumerate(rows, 1):
        if row.get("attempt") != index:
            raise ValueError("Trace must include all attempts in consecutive order")
        if type(row.get("valid")) is not bool or type(row.get("duplicate")) is not bool:
            raise ValueError("Each attempt needs boolean valid and duplicate")
        if row["valid"]:
            if not row.get("canonical") or not math.isfinite(row["fitness"]):
                raise ValueError("Valid rows need canonical identity and finite fitness")
            if not row.get("min_objectives") or not all(math.isfinite(x) for x in row["min_objectives"]):
                raise ValueError("Valid rows need finite minimized objective values")
            identity = row["canonical"]
            if protocol["task"] in ("dolphin", "residence_hall"):
                from .tasks import load_task
                identity = load_task(protocol["task"]).canonicalize(json.loads(identity))
            elif protocol["task"] == "molecules":
                from .tasks import load_task
                identity = load_task("molecules").canonicalize(identity)
                if identity != row["canonical"]:
                    raise ValueError("Molecular canonical identity is not canonical SMILES")
            if row["duplicate"] != (identity in seen):
                raise ValueError("Duplicate flags disagree with canonical identities")
            seen.add(identity)
        elif row["duplicate"]:
            raise ValueError("Invalid candidates cannot be marked canonical duplicates")
        budget += 1 if protocol["budget_metric"] == "attempts" else int(row["valid"] and not row["duplicate"])
        if row["budget_used"] != budget or budget > protocol["budget"]:
            raise ValueError("Trace budget disagrees with its protocol")


def plateau(rows, protocol):
    valid = [r for r in rows if r["valid"] and not r["duplicate"]]
    if not valid:
        return None
    def value(row):
        if protocol["task"] == "molecules":
            return row["fitness"]
        return row["objectives"]["latency" if protocol["task"] == "kernel" else "error"]
    maximize = protocol["task"] == "molecules"
    best = None
    trajectory = []
    for row in valid:
        val = value(row)
        best = val if best is None else (max(best, val) if maximize else min(best, val))
        trajectory.append((row["budget_used"], best))
    final = trajectory[-1][1]
    for budget, current in trajectory:
        improvement = final - current if maximize else current - final
        tolerance = protocol["epsilon"]
        if protocol["epsilon_mode"] == "relative_error":
            tolerance *= abs(current)
        elif protocol["epsilon_mode"] == "reference_gap":
            tolerance *= max(0, current - protocol["reference"])
        if improvement <= tolerance + 1e-12:
            return {"onset_budget": budget, "remaining_fraction": 1 - budget / protocol["budget"],
                    "observed_endpoint_budget": rows[-1]["budget_used"], "final_value": final}


def text_evidence(audit, rows, protocol):
    if audit is None:
        return {"eligible": None, "narrowing": None, "reason": "No reviewed text audit / eligibility declaration"}
    if audit.get("explicit_model_covers_search") is False:
        return {"eligible": False, "narrowing": False, "reason": "No covering explicit principle-space model"}
    if audit.get("explicit_model_covers_search") is not True:
        raise ValueError("Audit must declare explicit_model_covers_search as boolean")
    width = protocol["window_attempts"]
    expected = [max(1, len(rows) - width + 1), len(rows)]
    if len(rows) < width or audit.get("attempt_range") != expected:
        return {"eligible": True, "narrowing": None, "reason": "Audit does not cover the last full 1000 attempts"}
    required = ["dominant_route_share", "minor_updates", "major_updates", "new_directions", "reviewer", "evidence"]
    if any(k not in audit for k in required) or not audit["reviewer"] or not audit["evidence"]:
        return {"eligible": True, "narrowing": None, "reason": "Incomplete reviewed audit"}
    if not 0 <= audit["dominant_route_share"] <= 1 or any(type(audit[k]) is not int or audit[k] < 0 for k in required[1:4]):
        raise ValueError("Invalid audit share or event counts")
    t1 = audit["dominant_route_share"] >= protocol["text_dominant_min"]
    t2 = audit["minor_updates"] <= protocol["text_minor_updates_max"] and audit["major_updates"] == 0
    t3 = audit["new_directions"] == 0
    return {"eligible": True, "narrowing": t1 and t2 and t3, "T1": t1, "T2": t2, "T3": t3}


def diagnose(rows, protocol, audit=None, labels=None):
    validate_trace(rows, protocol)
    width = protocol["window_attempts"]
    result = {"protocol": protocol, "attempts": len(rows),
              "attempt_metrics": {"global": attempt_metrics(rows), "early": attempt_metrics(rows[:width]),
                                  "late": attempt_metrics(rows[-width:])}}
    valid = [r for r in rows if r["valid"] and not r["duplicate"]]
    if len(valid) < protocol["selection_size"]:
        return {**result, "verdict": "Unassessable", "reason": "Insufficient distinct valid candidates"}
    initial_end = next(i for i, r in enumerate(rows) if r["attempt"] == valid[protocol["selection_size"] - 1]["attempt"]) + 1
    initial_rows = [i + 1 for i, r in enumerate(rows) if r.get("phase") == "initialization"]
    if initial_rows:
        initial_end = max(initial_end, max(initial_rows))
    early, late = snapshot(rows[:initial_end], protocol, labels), snapshot(rows, protocol, labels)
    p = plateau(rows, protocol)
    gap = p["final_value"] < protocol["gap_fitness_below"] if protocol["task"] == "molecules" else p["final_value"] / protocol["reference"] >= protocol["gap_ratio_min"]
    structural = None if late["n_eff"] is None else late["n_eff"] <= protocol["narrowing_max"]
    text = text_evidence(audit, rows, protocol) if structural is not True else {"eligible": None, "narrowing": None, "reason": "Structural branch suffices"}
    long = p["remaining_fraction"] >= protocol["remaining_fraction"] - 1e-12
    if not gap:
        verdict = "Not established"
    elif structural or text["narrowing"] is True:
        verdict = "Strong support" if long else "Limited support"
    elif structural is None:
        verdict = "Unassessable"
    elif text["narrowing"] is None:
        verdict = "Unassessable"
    else:
        # Covering principle models require stable/no-new directions for limited
        # textual support; non-covering models take the plateau-only branch.
        stable = text["eligible"] is False or (text.get("T2") and text.get("T3"))
        verdict = "Limited support" if stable and long else "Not established"
    reference_comparison = {"reference": protocol["reference"], "observed": p["final_value"],
                            "difference": protocol["reference"] - p["final_value"] if protocol["task"] == "molecules" else p["final_value"] - protocol["reference"]}
    if protocol["task"] != "molecules":
        reference_comparison["ratio"] = p["final_value"] / protocol["reference"]
    return {**result, "early": early, "late": late, "plateau": p, "reference_comparison": reference_comparison,
            "evidence": {"large_reference_gap": gap, "structural_narrowing": structural,
                         "long_plateau": long, "text": text}, "verdict": verdict}
