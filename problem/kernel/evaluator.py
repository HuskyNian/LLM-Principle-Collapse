"""ExLLM problem definition for GPU kernel optimization on SOL-ExecBench.

Interface contract (see algorithm/MOO.py, algorithm/base.py):
  * ``generate_initial_population(config, seed) -> list[str]``
  * ``RewardingSystem(config=...)`` with
    ``evaluate(items, mol_buffer=None, main_history=False) -> (items, log_dict)``

A candidate's ``value`` is the full source of a single Python file defining
``run(...)``. That covers both Triton and plain-PyTorch solutions, which is the
same shape the published FlashInfer-Bench reference solutions use.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional

from problem.kernel import harness
from problem.kernel.harness import (
    EvalResult,
    ProblemSpec,
    build_problem_spec,
    canonical_key,
    evaluate_candidate,
    extract_source,
    stage_problem,
)

BENCHMARK_ROOT = os.path.join(harness.SOL_ROOT, "data", "benchmark")
DEFAULT_RUN_ROOT = os.path.join(harness.SOL_ROOT, "out", "exllm_runs")


# --------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------


def _cfg(config, key, default=None):
    """Read from a ConfigLoader or a plain dict, tolerating both."""
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if getter is None:
        return default
    try:
        value = getter(key, default=default)
    except TypeError:                     # plain dict.get
        value = getter(key, default)
    return default if value is None else value


def _resolve_problem_dir(problem_name: str) -> str:
    """Accept either 'FlashInfer-Bench/018_...' or a bare problem name."""
    direct = os.path.join(BENCHMARK_ROOT, problem_name)
    if os.path.isdir(direct):
        return direct
    for category in sorted(os.listdir(BENCHMARK_ROOT)):
        candidate = os.path.join(BENCHMARK_ROOT, category, problem_name)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"problem {problem_name!r} not found under {BENCHMARK_ROOT}")


# --------------------------------------------------------------------------
# initial population
# --------------------------------------------------------------------------


def generate_initial_population(config, seed) -> List[str]:
    """提供初始种群。框架契约: 返回字符串列表，每个字符串 = 一个初始候选的
    "基因"（对 kernel 任务就是一份完整 Python 源码），框架会把每个字符串
    包成 Item 对象。

    ``init_mode`` 三种模式:
      * ``reference``（默认）—— 返回一个元素: 题目自带 reference.py 的
        **源码文本**（下面的 reference_src 就是把那个文件 read() 进来的字符串，
        即朴素 PyTorch 实现的代码本身）。这样所有 run 从同一个中性、未优化的
        起点出发，之后轨迹里出现的任何设计路线都是优化器自己走出来的，
        不是我们喂的。只返回 1 份是因为框架按字符串去重，返回 50 份一样的
        也会塌成 1 个个体；不足 pop_size 的部分由框架的 llm_init 用
        backbone 现场生成补齐。
      * ``llm`` —— 返回空列表 []。框架看到空列表会转而让 backbone 从零
        生成整个初始种群（= 直接采样模型先验作为起点，留作对照实验用）。
      * ``files`` —— 从 ``init_seed_paths`` 指定的文件读种子（自定义起点）。
    """
    mode = _cfg(config, "init_mode", "reference")
    pop_size = int(_cfg(config, "optimization.pop_size", 20))

    if mode == "llm":
        return []

    if mode == "files":
        paths = _cfg(config, "init_seed_paths", []) or []
        sources = []
        for path in paths:
            with open(path) as fh:
                sources.append(fh.read())
        if not sources:
            raise ValueError("init_mode='files' but init_seed_paths is empty")
        return [sources[i % len(sources)] for i in range(pop_size)]

    problem_name = _cfg(config, "problem_name")
    if not problem_name:
        raise ValueError("config must set problem_name")
    with open(os.path.join(_resolve_problem_dir(problem_name), "reference.py")) as fh:
        reference_src = fh.read()

    # One genuine seed. MOO dedups on value, so returning pop_size identical
    # copies would collapse to a single individual anyway; llm_init tops the
    # population back up to pop_size from there.
    return [reference_src]


# --------------------------------------------------------------------------
# fitness
# --------------------------------------------------------------------------


class RewardingSystem:
    """kernel 任务的评测打分器（ExLLM 的"问题接口"）。

    它是什么: ExLLM 框架对每个问题只认两样东西——generate_initial_population
    函数和一个叫 RewardingSystem 的类。框架启动时按 config 里的
    `evalutor_path` 动态 import 本模块并实例化本类（model/MOLLM.py:83-86），
    存为 MOO 里的 self.reward_system；此后每一代进化，框架把整批新候选
    交给 reward_system.evaluate() 打分。分子任务的同名类算 5 个分子性质；
    我们这个类做的是: 编译 + 上真卡计时 + 数值校验 + 打分。

    打分是刻意的临时方案: 原始耗时全部记录在 original_results 和
    checkpoint 里，阶段 4 无论换什么锚点都可以离线重算，不用重跑实验。
    transformed 分数 = 对 T_ref 的 log 加速比压到 [0,1] 并翻成"越小越好"
    ——因为框架的 NSGA-II 支配比较和 hypervolume 参考点 1.1
    (model/util.py:54-64, 148-153) 把"[0,1] 最小化"这个约定写死了。
    取 log 是因为这些题的耗时横跨三个数量级，线性归一会把所有正常候选
    挤进一条缝里，轨迹失去分辨率。
    """

    def __init__(self, config=None):
        self.config = config
        self.objs = list(_cfg(config, "goals", ["latency"]))

        problem_name = _cfg(config, "problem_name")
        if not problem_name:
            raise ValueError("config must set problem_name")
        self.problem_name = problem_name
        self.language = _cfg(config, "kernel_language", "triton")

        run_tag = _cfg(config, "run_tag") or f"{problem_name}_{int(time.time())}"
        self.workdir = os.path.join(_cfg(config, "run_root", DEFAULT_RUN_ROOT), str(run_tag))
        os.makedirs(self.workdir, exist_ok=True)
        harness._open_for_container(self.workdir)   # 容器 uid 与宿主不同的机器上评测才写得进来

        workload_uuids = _cfg(config, "workload_uuids", None)
        self.problem: ProblemSpec = stage_problem(
            _resolve_problem_dir(problem_name),
            os.path.join(self.workdir, "problem"),
            workload_uuids=workload_uuids,
        )
        self._render_prompt_info(config)

        self.t_ref_ms = float(_cfg(config, "t_ref_ms", 0) or 0)
        if self.t_ref_ms <= 0:
            self.t_ref_ms = self._t_ref_from_cache(problem_name)
        # Width of the scoring window, in decades of speedup over T_ref.
        self.log_decades = float(_cfg(config, "score_log_decades", 3.0))

        # The run scheduler assigns GPUs via the environment, which takes
        # precedence over the config so one config can be reused everywhere.
        # SOL_KERNEL_EVAL_GPUS (comma list) enables in-batch parallelism:
        # candidates within one generation are independent, so they can be
        # benchmarked on different GPUs at the same time. Only generations are
        # inherently serial (the next prompt needs this generation's results).
        # Cross-run safety comes from the per-GPU file locks in harness.py.
        self.gpu = str(os.environ.get("SOL_KERNEL_EVAL_GPU") or _cfg(config, "eval_gpu", "0"))
        pool = os.environ.get("SOL_KERNEL_EVAL_GPUS") or _cfg(config, "eval_gpus", "")
        self.gpu_pool = [g.strip() for g in str(pool).split(",") if g.strip()] or [self.gpu]
        # "persistent"（默认）= 长驻容器 + docker exec 钉卡（快，跨 run 多卡并行）；
        # "container" = 每次评测一个新容器（旧模式，回退用）。见 harness.py 注释
        self.eval_mode = _cfg(config, "eval_mode", "persistent")
        self.image = _cfg(config, "eval_image", harness.DEFAULT_IMAGE)
        self.warmup_runs = int(_cfg(config, "eval_warmup_runs", 10))
        self.iterations = int(_cfg(config, "eval_iterations", 50))
        self.timeout_s = int(_cfg(config, "eval_timeout_s", 300))
        self.compile_timeout_s = int(_cfg(config, "eval_compile_timeout_s", 300))

        # 去重集合: 存 canonical_key 指纹（见 harness.canonical_key 的注释）。
        # 没有它 repeated_num 恒为 0、uniqueness 恒为 1，候选层多样性指标失真。
        self.history: set = set()

        # 最近失败记录（错误回传用，见 get_error_feedback）。deque(maxlen=N)
        # 是个定长队列: 塞满后再 append 会自动挤掉最旧的一条。
        self.error_feedback = bool(_cfg(config, "error_feedback", False))
        self.recent_failures: deque = deque(
            maxlen=int(_cfg(config, "error_feedback_window", 3)))
        # 单条错误信息注入 prompt 的长度上限（完整注入是原则，上限只是
        # 防极端长日志撑爆 prompt 的保险，超限时会标注 [truncated]）
        self.error_feedback_max_chars = int(_cfg(config, "error_feedback_max_chars", 2000))

        self.eval_log_path = os.path.join(self.workdir, "eval_log.jsonl")
        self.n_evaluated = 0

    def _render_prompt_info(self, config) -> None:
        """Inject the concrete problem definition into the prompt material.

        Unlike SMILES, a kernel candidate has a schema: the exact run()
        signature, output convention and shape ranges of THIS problem. The
        static prompt_info.yaml is a template with a {PROBLEM_SPEC} slot; we
        render it against the staged problem and repoint the in-memory config
        at the rendered file. This works because MOLLM constructs the
        RewardingSystem (model/MOLLM.py:86) before MOO builds the
        PromptTemplate, which is what reads prompt_info_path.
        """
        template_path = _cfg(config, "prompt_info_path")
        if not template_path:
            return  # 无 prompt 模板（如纯评测用途的实例化），跳过渲染
        with open(template_path) as fh:
            text = fh.read()
        if "{PROBLEM_SPEC}" not in text:
            return  # already rendered, or a hand-written per-problem file
        spec_text = build_problem_spec(self.problem.problem_dir)
        # Indent to match the YAML block scalar the placeholder sits in.
        indented = spec_text.replace("\n", "\n  ")
        rendered_path = os.path.join(self.workdir, "prompt_info_rendered.yaml")
        with open(rendered_path, "w") as fh:
            fh.write(text.replace("{PROBLEM_SPEC}", indented))
        config.config["prompt_info_path"] = rendered_path

    @staticmethod
    def _t_ref_from_cache(problem_name: str) -> float:
        """Look the anchor up in the measured cache.

        The anchor is fixed per problem and measured once on this machine
        (problem/kernel/t_ref_cache.json, produced by
        SOL-ExecBench/out/pc_smoke/run_tref_batch.sh). Re-measuring it per run
        would make trajectories from different runs incomparable.
        """
        cache_path = os.path.join(os.path.dirname(__file__), "t_ref_cache.json")
        try:
            with open(cache_path) as fh:
                cache = json.load(fh)
        except OSError as exc:
            raise ValueError(f"no t_ref_ms in config and cache unreadable: {exc}") from exc
        if problem_name not in cache:
            raise ValueError(
                f"no t_ref_ms in config and {problem_name!r} absent from {cache_path}. "
                "Run out/pc_smoke/run_tref_batch.sh for this problem first."
            )
        return float(cache[problem_name]["t_ref_ms"])

    # -- scoring -----------------------------------------------------------

    def _transform(self, latency_ms: float) -> float:
        speedup = self.t_ref_ms / latency_ms
        # speedup <= 1x  -> 1.0 (worst);  speedup >= 10^decades -> 0.0 (best)
        normalized = 1.0 - math.log10(max(speedup, 1e-12)) / self.log_decades
        return float(min(1.0, max(0.0, normalized)))

    def _results_for(self, ev: EvalResult) -> Dict[str, Any]:
        transformed = self._transform(ev.latency_ms)
        # Every key here is echoed back to the model in the parent listing, so
        # keep it to quantities a kernel author would actually act on.
        original = {obj: ev.latency_ms for obj in self.objs}
        original["speedup_vs_ref"] = self.t_ref_ms / ev.latency_ms
        return {
            "original_results": original,
            "transformed_results": {obj: transformed for obj in self.objs},
            "overall_score": float(len(self.objs) * (1.0 - transformed)),
        }

    # -- main entry point --------------------------------------------------

    def evaluate(self, items, mol_buffer=None, main_history=False):
        """给一批候选打分。框架每代调用一次，传入整批新生成的候选。

        入参 items: Item 对象列表。Item 是框架的"个体"容器（algorithm/base.py），
        它的 .value 字段存该候选的内容字符串——分子任务里是 SMILES，
        kernel 任务里是整个 Python 源文件的文本（字段名 value 是框架起的
        通用名，我们只是沿用）。此处 .value 的初始内容 = LLM 回复中
        <candidate>...</candidate> 之间的原始文本（可能还包着 markdown 围栏）。

        返回 (存活候选列表, {'invalid_num': 无效数, 'repeated_num': 重复数})。
        两条铁律:
          * 框架不做任何过滤——无效的、重复的必须在这里剔除，只返回活下来的；
          * 这里绝不能抛异常——框架没有兜底，一抛整个 run 就死，
            所以循环体整个包在 try/except 里。
        """
        survivors = []
        invalid_num = 0
        repeated_num = 0

        # ---- 阶段 1（串行）: 解析 + 去重。去重必须按顺序做（后到的重复
        #      让位给先到的），且共享 self.history，串行最简单可靠。 ----
        todo = []   # (item, source, key)
        for item in items:
            source = extract_source(getattr(item, "value", None))
            if source is None:
                invalid_num += 1
                self._log({"outcome": "PARSE_FAIL"})
                continue
            key = canonical_key(source)
            if key in self.history:
                repeated_num += 1
                continue
            self.history.add(key)
            item.value = source          # 规范化回写（见类注释）
            todo.append((item, source, key))

        # ---- 阶段 2（并行）: 编译 + 上真卡计时。 ----
        #
        # 为什么能并行：todo 里是同一代的候选，彼此之间没有任何依赖
        # （下一代才依赖这一代的分数），所以可以同时测。
        #
        # 怎么并行（三件套，逐行对应下面的代码）：
        #   1. gpu_q —— 一个"钥匙筐"：把可用卡号（来自调度器传的
        #      SOL_KERNEL_EVAL_GPUS，通常是全部 8 张）放进线程安全的队列。
        #   2. _run_one —— 每个候选的评测流程：先从筐里取一把卡钥匙
        #      （gpu_q.get()，筐空则阻塞等别人归还），用这张卡评测，
        #      结束后无论成败都把钥匙放回（finally 里的 put）。
        #   3. ThreadPoolExecutor —— 开"钥匙数"个线程，把 todo 里的候选
        #      分发给它们。10 个候选、8 张卡 → 前 8 个立刻开测，
        #      后 2 个等谁先测完谁的卡。
        #
        # 跨 run 的争抢不归这里管：可能别的 run 进程也在用 3 号卡，
        # 所以 evaluate_candidate 内部还要过 harness._gpu_lock（每卡一把
        # 文件锁，跨进程生效）——本进程的钥匙筐只保证"我自己不同时
        # 往一张卡塞两个评测"，文件锁保证"全机不同时往一张卡塞两个评测"。
        import queue as _queue
        from concurrent.futures import ThreadPoolExecutor

        gpu_q: "_queue.Queue" = _queue.Queue()
        for g in self.gpu_pool:
            gpu_q.put(g)                          # 装钥匙

        def _run_one(pair):
            item, source, key = pair
            gpu = gpu_q.get()                     # 取钥匙（可能等）
            try:
                return evaluate_candidate(
                    source=source, problem=self.problem, workdir=self.workdir,
                    language=self.language, image=self.image, gpu=gpu,
                    timeout_s=self.timeout_s, compile_timeout_s=self.compile_timeout_s,
                    warmup_runs=self.warmup_runs, iterations=self.iterations,
                    eval_mode=self.eval_mode)
            except Exception:                     # noqa: BLE001 - 绝不外抛
                return traceback.format_exc()[-1500:]
            finally:
                gpu_q.put(gpu)                    # 还钥匙

        with ThreadPoolExecutor(max_workers=len(self.gpu_pool)) as ex:
            results = list(ex.map(_run_one, todo))   # 分发并等全部完成

        # ---- 阶段 3（串行，按原顺序）: 判定、打分、记录。保持输入顺序，
        #      使 mol_buffer 的序号（轨迹分析的 x 轴）确定。 ----
        for (item, source, key), ev in zip(todo, results):
            if isinstance(ev, str):               # 阶段 2 捕获的异常文本
                invalid_num += 1
                self._log({"outcome": "EVALUATOR_EXCEPTION", "traceback": ev})
                continue
            self.n_evaluated += 1
            if not ev.ok:
                invalid_num += 1
                self._remember_failure(source, ev)
                self._log({"outcome": ev.status, "n_passed": ev.n_passed,
                           "n_total": ev.n_total, "key": key, "log": ev.log[:800]})
                continue
            item.assign_results(self._results_for(ev))
            # 逐用例明细挂在 Item 上，随 checkpoint 存盘
            item.kernel_eval = {"key": key, "latency_ms": ev.latency_ms,
                                "per_workload": ev.per_workload}
            survivors.append(item)
            self._log({"outcome": "PASSED", "key": key, "latency_ms": ev.latency_ms,
                       "speedup_vs_ref": self.t_ref_ms / ev.latency_ms})

        return survivors, {"invalid_num": invalid_num, "repeated_num": repeated_num}

    # -- error feedback ------------------------------------------------------

    _CLOCK_PREFIX = re.compile(r"^Clocks locked: \w+\s*")

    def _remember_failure(self, source: str, ev: EvalResult) -> None:
        """把一次失败整理成可回传的错误信息，存进定长队列。

        原则: 错误信息**完整**进入下一轮 prompt——模型要靠它定位错在哪，
        只剔除对模型没有信息量的部分:
          * "Clocks locked" 状态行（harness 自身的运行日志）
          * 指向 harness 内部文件 eval_driver.py 的堆栈帧及其代码行
            （那是评测器的调用栈，不是候选代码的问题位置）
        保留真正的异常信息和指向候选源码的行（如 Triton 编译错误里的
        "at 6:20: ..." 源码定位）。超过 error_feedback_max_chars 才截断，
        且明确标注 [truncated]。
        """
        # 候选第一行如果是模型自述的设计意图注释（mutation 指令要求的），
        # 一并记下来，让反馈里能对上"哪个想法失败了"
        first_line = source.strip().splitlines()[0]
        intent = first_line if first_line.startswith("#") else ""

        raw = self._CLOCK_PREFIX.sub("", (ev.log or "").strip())
        kept: List[str] = []
        skip_next = False
        for ln in raw.splitlines():
            if skip_next:                      # 跳过 harness 堆栈帧下面那行代码
                skip_next = False
                continue
            if "eval_driver.py" in ln:
                skip_next = True
                continue
            if ln.strip() == "Traceback (most recent call last):":
                continue
            kept.append(ln)
        err = "\n".join(kept).strip()

        if ev.status == "INCORRECT_NUMERICAL":
            suffix = (f"output values did not match the reference "
                      f"({ev.n_passed}/{ev.n_total} workloads passed)")
            err = (err + "\n" + suffix).strip() if err else suffix

        if len(err) > self.error_feedback_max_chars:
            err = err[: self.error_feedback_max_chars] + "\n...[truncated]"
        self.recent_failures.append({"status": ev.status, "intent": intent, "err": err})

    def get_error_feedback(self) -> str:
        """供框架在生成 prompt 后追加的"最近失败摘要"。

        背景: ExLLM 原本把无效候选静默丢弃，模型永远看不到自己错在哪，
        pilot 里近半预算耗在盲试错上。kernel 领域的搜索系统（K-Search 等）
        普遍把错误文本回传。此机制由 config `error_feedback` 开关控制，
        分子任务默认关闭、行为不变；开启后由 algorithm/MOO.py 的
        mutation/crossover 在 prompt 末尾追加本函数的返回值。
        """
        if not self.error_feedback or not self.recent_failures:
            return ""
        lines = ["", "Recent failed attempts on this problem (avoid repeating these mistakes):"]
        for i, f in enumerate(self.recent_failures, 1):
            intent = f" (intent: {f['intent']})" if f["intent"] else ""
            lines.append(f"--- failure {i}: [{f['status']}]{intent}")
            lines.append(f["err"])
        lines.append("")
        return "\n".join(lines)

    # -- logging -----------------------------------------------------------

    def _log(self, record: Dict[str, Any]) -> None:
        record["ts"] = time.time()
        record["problem"] = self.problem_name
        try:
            with open(self.eval_log_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            pass
