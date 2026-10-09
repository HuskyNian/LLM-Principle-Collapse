"""Metrics on a declared diagnostic selection, not an inferred live population."""

from collections import Counter
from itertools import combinations
import math


def entropy(counts):
    total = sum(counts)
    return -sum((c / total) * math.log(c / total) for c in counts if c) if total else 0.0


def select(rows, size, policy):
    rows = [r for r in rows if r["valid"] and not r["duplicate"]]
    if policy == "fitness":
        return sorted(rows, key=lambda r: r["fitness"], reverse=True)[:size]
    if policy != "nsga2_so":
        raise ValueError(f"Unsupported diagnostic selection: {policy}")
    if len(rows) <= size:
        return rows
    # Sentinel fitness=-error makes mixed selection identical to fitness ranking.
    if all(len(r["min_objectives"]) == 1 and r["fitness"] == -r["min_objectives"][0] for r in rows):
        return sorted(rows, key=lambda r: r["fitness"], reverse=True)[:size]
    from algorithm.base import Item
    from model.util import nsga2_so_selection
    items, lookup = [], {}
    for row in rows:
        item = Item(row["canonical"], [])
        item.total, item.scores = row["fitness"], row["min_objectives"]
        items.append(item)
        lookup[item.value] = row
    return [lookup[item.value] for item in nsga2_so_selection(items, size, 0.5)]


def molecular_metrics(rows):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mols = [Chem.MolFromSmiles(r["canonical"]) for r in rows]
    if any(m is None for m in mols):
        raise ValueError("Trace marked an invalid SMILES as valid")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(m) for m in mols]
    pairs = list(combinations(fps, 2))
    diversity = sum(1 - DataStructs.TanimotoSimilarity(a, b) for a, b in pairs) / len(pairs) if pairs else 0.0
    labels = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    counts = Counter(labels)
    result = {"diversity": diversity, "n_eff": math.exp(entropy(counts.values())),
              "unique_scaffolds": len(counts), "dominant_share": max(counts.values()) / len(rows)}
    # The empty (acyclic) scaffold is one category, consistent with Murcko counting.
    from pymoo.indicators.hv import HV
    import numpy as np
    result["hv"] = float(HV(ref_point=np.full(5, 1.1))(np.array([r["min_objectives"] for r in rows])))
    return result


def network_metrics(rows, subset_size):
    import json
    sets = [sorted(json.loads(r["canonical"])) for r in rows]
    counts = Counter(n for nodes in sets for n in nodes)
    # For m distinct k-sets, at most min(m*k, m*5, 4*m+1) appearances can
    # lie in five nodes when k=5 (one all-five set); k=4 allows at most five
    # all-inside sets, then three appearances per set.
    m, k = len(sets), subset_size
    if k == 4:
        max_top5 = 4 * min(m, 5) + 3 * max(0, m - 5)
    elif k == 5:
        max_top5 = 5 + 4 * (m - 1)
    else:
        raise ValueError("Normalized top-5 coverage is defined here only for k=4 or k=5")
    return {"n_eff": math.exp(entropy(counts.values())),
            "normalized_top5_coverage": sum(sorted(counts.values(), reverse=True)[:5]) / max_top5,
            "max_pair_reuse": max(Counter(p for s in sets for p in combinations(s, 2)).values()) / m,
            "max_core3_reuse": max(Counter(p for s in sets for p in combinations(s, 3)).values()) / m}


def snapshot(rows, protocol, labels=None):
    selected = select(rows, protocol["selection_size"], protocol["selection"])
    if not selected:
        return None
    valid = [r for r in rows if r["valid"] and not r["duplicate"]]
    best = sorted(valid, key=lambda r: r["fitness"], reverse=True)
    result = {"selected_size": len(selected), "top1": best[0]["fitness"],
              "top10": sum(r["fitness"] for r in best[:10]) / min(10, len(best))}
    if protocol["task"] == "molecules":
        result.update(molecular_metrics(selected))
    elif protocol["task"] == "kernel":
        route_labels = [labels.get(r["canonical"]) for r in selected] if labels else []
        result["n_eff"] = math.exp(entropy(Counter(route_labels).values())) if route_labels and all(route_labels) else None
    else:
        result.update(network_metrics(selected, protocol["subset_size"]))
    return result


def attempt_metrics(rows):
    if not rows:
        return {"uniqueness": None, "validity": None, "count": 0}
    return {"uniqueness": 1 - sum(r["duplicate"] for r in rows) / len(rows),
            "validity": sum(r["valid"] for r in rows) / len(rows), "count": len(rows)}
