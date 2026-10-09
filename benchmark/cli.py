import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

from .report import generate_report, summarize
from .tasks import DEFAULT_NETWORK_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Algorithm-independent principle-collapse benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Check optional dependencies and network data")
    check.add_argument("--data-dir", type=Path, default=DEFAULT_NETWORK_DATA_DIR)
    report = commands.add_parser("report", help="Validate a run and write diagnosis.json/report.md")
    report.add_argument("run_dir")
    report.add_argument("--audit", help="Reviewed textual audit or explicit ineligibility declaration")
    report.add_argument("--labels", help="Kernel canonical-key to audited route-label JSON")
    report.add_argument("--plots", action="store_true", help="Also write per-run PNG figures (requires matplotlib)")
    summary = commands.add_parser("summarize", help="Aggregate diagnosed seeds by task and optimizer")
    summary.add_argument("run_dirs", nargs="+")
    summary.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare-network-data", help="Stage validated, lawfully obtained .npy files")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "report":
        result = generate_report(args.run_dir, args.audit, args.labels)
        if args.plots:
            from .plotting import plot_run
            from .recorder import read_rows
            plot_run(args.run_dir, read_rows(Path(args.run_dir) / "attempts.jsonl"), result)
        print(f"{result['verdict']}: {Path(args.run_dir) / 'report.md'}")
    elif args.command == "summarize":
        print(json.dumps(summarize(args.run_dirs, args.output), indent=2))
    elif args.command == "doctor":
        checks = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "rdkit", "tdc", "pymoo", "pygmo")}
        checks["docker_executable"] = shutil.which("docker") is not None
        checks["note"] = "Import presence is not a compatibility/GPU readiness test. No model API is called."
        if args.data_dir:
            checks["network_data"] = {name: (args.data_dir / "fullstates" / f"fullstate-{name}-{dynamics}.npy").exists()
                                      for name, dynamics in (("dolphin", "doublewell"), ("residence_hall", "genereg"))}
        print(json.dumps(checks, indent=2))
    else:
        import numpy as np
        manifest = {}
        for name, dynamics, nodes in (("dolphin", "doublewell", 62), ("residence_hall", "genereg", 217)):
            filename = f"fullstate-{name}-{dynamics}.npy"
            source = args.source / "fullstates" / filename
            array = np.load(source, allow_pickle=False)
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != nodes or not np.isfinite(array).all() or array.mean(axis=1).sum() <= 0:
                raise ValueError(f"Invalid fullstate array: {source}")
            target = args.destination / "fullstates" / filename
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"Refusing to overwrite different data: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            manifest[filename] = {"sha256": digest, "shape": list(array.shape)}
        from .recorder import write_json
        write_json(args.destination / "manifest.json", manifest)
        print(args.destination / "manifest.json")
