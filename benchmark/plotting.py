"""Lightweight per-run figures; no synthetic observations."""

from pathlib import Path


def plot_run(directory, rows, diagnosis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    protocol = diagnosis["protocol"]
    molecule = protocol["task"] == "molecules"
    valid = [r for r in rows if r["valid"] and not r["duplicate"]]
    if not valid:
        return
    values, best = [], None
    for row in valid:
        value = row["fitness"] if molecule else row["objectives"]["latency" if protocol["task"] == "kernel" else "error"]
        best = value if best is None else (max(best, value) if molecule else min(best, value))
        values.append(best)
    fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
    ax.step([r["budget_used"] for r in valid], values, where="post", label="Best observed", color="#0072B2")
    ax.axhline(protocol["reference"], linestyle="--", color="#009E73", label="Reference")
    if diagnosis.get("plateau"):
        onset = diagnosis["plateau"]["onset_budget"]
        index = next(i for i, row in enumerate(valid) if row["budget_used"] >= onset)
        ax.scatter([onset], [values[index]], marker="*", s=110, color="#D55E00", label="Plateau onset", zorder=3)
    ax.set(xlabel="Evaluation budget", ylabel="Top-1 fitness" if molecule else "Best latency (ms)" if protocol["task"] == "kernel" else "Best error",
           xlim=(0, protocol["budget"]))
    ax.legend(loc="best", fontsize=8)
    fig.savefig(directory / "convergence.png", dpi=180)
    plt.close(fig)
    metrics = [k for k in ("n_eff", "diversity", "unique_scaffolds", "dominant_share", "normalized_top5_coverage", "max_pair_reuse", "max_core3_reuse")
               if diagnosis.get("early", {}).get(k) is not None and diagnosis.get("late", {}).get(k) is not None]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.2 * len(metrics), 2.8), layout="constrained", squeeze=False)
    for ax, key in zip(axes[0], metrics):
        ax.bar([0, 1], [diagnosis["early"][key], diagnosis["late"][key]], color=["#56B4E9", "#0072B2"])
        ax.set_xticks([0, 1], ["Early", "Late"])
        ax.set_ylabel(key.replace("_", " "))
    fig.savefig(directory / "early_late.png", dpi=180)
    plt.close(fig)
