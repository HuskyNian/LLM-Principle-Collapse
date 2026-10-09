# Principle-Level Collapse Benchmark

## Installation and readiness

The core recorder, CLI and decision logic use Python's standard library.
Network scoring additionally needs NumPy. Molecules need RDKit, PyTDC and the
existing oracle assets; molecular metrics need pymoo. 
Starter dependency lists are `benchmark/requirements-network.txt`
and `benchmark/requirements-molecules.txt`. 

```bash
python -m unittest discover -s benchmark/tests -v
```

## Complete example: random ZINC search

From the repository root, in an environment with the molecular dependencies installed:

```bash
python -m benchmark.examples.random_zinc \
  --seed 42 --budget 200 --batch-size 25 \
  --output outputs/random_zinc/42_smoke
```

This runs the complete proposal -> five-objective evaluation -> recording ->
diagnosis -> plotting pipeline, without an LLM or API key. It randomly samples
SMILES without replacement.
The first run downloads about 22 MB to `~/.cache/exllm/zinc250k.csv`; later runs reuse it.
Use `--zinc-csv /path/to/zinc.csv` for a local CSV with a `smiles` column.
The existing `data/zinc.tab` is an HTML error response and is not used.

The first 100 sampled rows are initialization. The command saves `attempts.jsonl`,
`diagnosis.json`, `report.md`, `convergence.png`, and `early_late.png` in the output
directory. The 200-evaluation example is a **smoke test, not a full-budget collapse
experiment**; early/late 1000-attempt windows overlap in such a short run.
For the full molecular budget, use a new output directory:

```bash
python -m benchmark.examples.random_zinc \
  --seed 42 --budget 5000 --batch-size 50 \
  --output outputs/random_zinc/42_full
```

Random ZINC search is only an integration example, not a feedback-driven LLM
optimizer. Replace its `propose`/`observe` methods with your optimizer to reuse
the same evaluation and reporting interface. Existing run directories are never overwritten.

## Selected benchmark results



### Diagnosis decision tree

Combine reference gaps, principle narrowing, and performance plateaus; candidate-level diversity alone is not enough.

![Collapse diagnosis overview](images/benchmark/decision_tree.png)

Full decision tree, including the eligible principle-text audit

![Full collapse diagnosis decision tree](images/benchmark/decision_tree_full.png)

### Convergence across three domains

Performance and effective scaffold/node/route counts reveal different aspects of search.

**Molecular optimization**

![Molecular fitness and effective scaffolds by budget](images/benchmark/molecular_convergence.png)

**Network sentinel-node selection: residence hall**

![Residence hall error and effective nodes by budget](images/benchmark/residence_hall_convergence.png)

**GPU-kernel optimization**

![Kernel performance relative to expert references and effective routes](images/benchmark/kernel_convergence.png)

### Early versus late search

Molecular and network bars use outlined bars for early search and filled bars for late search;
scaffold counts and effective scaffold counts are divided by 50, and effective node
counts by network size (62 or 217). Kernel bars label early/late stages separately.

![Molecular early and late metrics](images/benchmark/molecular_early_late.png)

![Residence hall early and late metrics](images/benchmark/residence_hall_early_late.png)

![Kernel early and late metrics](images/benchmark/kernel_early_late.png)

Dolphin: convergence and early/late comparison

Dolphin provides a simpler task with an exhaustive optimum, helping distinguish harmful collapse from successful convergence.

![Dolphin convergence and effective nodes](images/benchmark/dolphin_convergence.png)

![Dolphin early and late metrics](images/benchmark/dolphin_early_late.png)

## Diversity injection

![Diversity injection on Molecular optimization problem, early and late metrics](images/benchmark/diversity_injection_bars.png)

## Molecular reference-principle interventions

Better directions help some models, but do not guarantee effective use.

![Molecular top-1 convergence with hints from the start or 40 percent of budget](images/benchmark/molecular_hint_convergence.png)

## Network data and attribution

The two network-dynamics arrays used by this benchmark are bundled in
`data/dynamic_system/fullstates/`:


| Task           | Dynamics        | NumPy file                             |
| -------------- | --------------- | -------------------------------------- |
| Dolphin        | Double-well     | `fullstate-dolphin-doublewell.npy`     |
| Residence hall | Gene regulation | `fullstate-residence_hall-genereg.npy` |


The task and dynamics data follow **Observing network dynamics through sentinel
nodes**. The underlying network data source is
[Netzschleuder: the network catalogue](https://networks.skewed.de/).

The benchmark loads these arrays by default, `data/dynamic_system/manifest.json` records their shapes and SHA256 hashes.
Use the following command to test the dolphin problem

```bash
python -m benchmark.examples.random_network --task dolphin --output outputs/dolphin_single_test --steps 100
```

To use another converted data directory, pass `data_dir=...` to `load_task` or
`--data-dir` to the example. The optional
`python -m benchmark prepare-network-data --source SOURCE --destination DESTINATION`
command validates and stages the same two arrays from `SOURCE/fullstates/`.

## Connect your optimizer

**full example in** `benchmark/examples/random_network.py & random_zinc.py`

```python
from benchmark import BenchmarkRun, load_task

task = load_task("molecules", seed=42)
run = BenchmarkRun(task, "outputs/my_method/42", optimizer="my-method")
initial = run.evaluate(task.initial_population(), phase="initialization")
# my_optimizer.observe(initial)
while run.budget_used < task.protocol["budget"]:
    candidates = my_optimizer.propose()
    results = run.evaluate(candidates)
    my_optimizer.observe(results)
run.finish("budget")
```

Use your own stopping guard for low uniqueness, because your algorithm may have very low uniqueness at a certain stage which may cause you huge amount of money and time to consume all the budget but results are not better. `evaluate` returns only the
processed prefix if a batch reaches the budget; it raises `BudgetExhausted` if
already exhausted. Invalid and duplicate proposals are recorded, not discarded.
Duplicates return cached scores. Backend failures propagate, rather than being
misreported as invalid candidates. Runs are single-writer, sequential evaluators.

Initialization counts toward evaluation budget. `resume=True` restores recorder
history, duplicate cache and counters, NOT your optimizer/RNG/LLM state. Restore
those yourself. Do not mix old and changed metadata in one run directory.

Network tasks default to `data/dynamic_system/`; `data_dir=...` overrides it.
Nodes are distinct integers, unordered,
one-indexed; weighted variants are outside this protocol.

## Diagnose and aggregate

```bash
python -m benchmark report outputs/my_method/42 --audit benchmark/examples/no_principle_model.json --plots
python -m benchmark summarize outputs/my_method/42 outputs/my_method/43 --output outputs/summary.csv
```

Artifacts: `metadata.json`, `attempts.jsonl`, optional `principles.jsonl`,
`termination.json`, then `diagnosis.json` and `report.md`. Summary CSV reports
sample SD only when at least two seeds are present. Different protocols are
grouped separately. Unassessable runs are excluded from support-rate denominators.
`--plots` additionally writes per-run convergence and early/late PNGs (matplotlib).
These are individual-run plots, not five-seed uncertainty plots. Legends stay
inside axes. Curves stop at the observed endpoint; the x-axis shows planned budget.

Any optimizer can instead export this schema directly. `report` validates
attempt ordering, identity/duplicate flags, and budget accounting. Valid records
need `fitness` (higher is better), `objectives` (raw), and `min_objectives` (the
existing minimized objectives). All attempts, including invalids, must be logged.
Use the recorder example to see the exact schema. Kernel rows require raw `objectives.latency` in milliseconds;
network rows require `objectives.error`.

## Metrics and diagnosis

Protocol values are frozen in `benchmark/configs/*.json` as **benchmark-v1**, an explicit
public-interface protocol. Always archive the resolved protocol alongside a run.

- Uniqueness = 1 - duplicate attempts / all attempts (invalids remain in the
denominator, matching ExLLM); validity = valid attempts / all attempts.
- Early/late attempt windows are first/last 1000 generated attempts; shorter
windows report their actual count and may overlap.
- Diagnostic metrics are computed on each optimizer’s own top-50 candidate set after initialization (early) and at the end of the run (late).
- Plateau onset is the earliest best-so-far point whose total subsequent observed
improvement is within epsilon, not just one flat step. Molecules: 0.01 fitness;
networks: 1% of the error at that point; kernels: 10% of its expert latency gap.
- Long plateau means at least 30% of planned budget remains at Plateau onset. 
- Large gap: molecular top1 < 4.3; network/kernel error or latency >= 2x reference.
References are best-known attainable results, except exhaustive Dolphin.
- Structural narrowing: molecular N_eff <= 3; network N_eff <= 4 times the
50-distinct-set lower bound (7.837115314 / 7.981300322); kernel route N_eff <= 2.
Changing selection size requires new calibrated thresholds, not these defaults.



## Text audits and kernel annotations

The no-principle-model example explicitly declares ineligibility; ordinary
experience memory alone does not qualify as a covering principle-space model.
Eligible algorithms must supply a reviewed `text_audit.json`. Merge wording
variants into semantic routes and weight actual usage over the last 1000 attempts.
T1: dominant route >= 50%; T2: <=5 minor updates, zero major updates; T3: zero
newly proposed directions (including unadopted directions). Reviewer and source
evidence are mandatory. For eligible models, T1+T2+T3 gives narrowing; otherwise only !T1+T2+T3 with a
long plateau gives Limited support. Other fully audited cases are Not established.

Kernel reports accept `--labels routes.json`, a mapping from canonical source
keys to reviewed decomposition labels. Missing labels yield unknown structural
evidence. Live scoring wraps the existing official harness:

```python
task = load_task("kernel", protocol={"reference": expert_latency_ms},
                 evaluator_config={"problem_name": problem_name})
```

Set `SOL_EXECBENCH_ROOT` before importing the kernel adapter. You still need the
external SOL-ExecBench problem assets, compatible NVIDIA GPU, Docker and the
existing `sol-execbench:fixed` image. Expert references must match hardware and
workloads. 

---


