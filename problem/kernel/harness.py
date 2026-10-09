"""Bridge between ExLLM and the official SOL-ExecBench evaluation harness.

Design notes
------------
* All benchmarking goes through the official Docker image so that clock locking
  and the anti-reward-hacking checks stay in force. We never time kernels
  ourselves.
* The image must be ``sol-execbench:fixed`` (the stock image ships a broken
  Triton install; see ***_workspace/ROADMAP.md).
* A candidate is a single Python source file implementing ``run(...)``. Both
  ``triton`` and ``pytorch`` solutions fit that shape. Multi-file CUDA C++ is a

"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SOL_ROOT = os.path.abspath(os.environ.get("SOL_EXECBENCH_ROOT", "/localhome/local-***j/SOL-ExecBench"))
CONTAINER_ROOT = "/sol-execbench"
DEFAULT_IMAGE = "sol-execbench:fixed"

# Languages we currently emit. Both map to a single ``main.py``.
SINGLE_FILE_LANGUAGES = {"triton", "pytorch"}


class HarnessError(RuntimeError):
    """Raised when the harness itself misbehaves (not when a kernel fails)."""


# ---------------------------------------------------------------------------
# 全机器范围的"评测互斥锁"
#
# 为什么需要:
#   计时要求 GPU 频率恒定（空闲时降到 120 MHz、忙时自动升到 1965 MHz，
#   频率浮动则同一 kernel 每次测出来都不一样）。官方评测容器的启动脚本
#   会调 `nvidia-smi -lgc/-lmc` 把频率锁到 1500/3996 MHz，退出时解锁——
#   而且这条命令不带 -i 参数，作用于**全部 8 张卡**（这是官方脚本的行为，
#   不是我们的选择）。
#   于是出现竞态: 容器 A 在 0 号卡上计时中，容器 B 在 1 号卡上先跑完退出，
#   B 的清理动作把 8 张卡全部解锁 → A 后面的测量全在浮动频率下进行，
#   而 A 自己毫不知情（它只在启动时检查过一次锁频状态）→ 数据悄悄作废。
#
# 解决: 用一个文件锁把"起评测容器"这个动作全机器串行化。
#   fcntl.flock 是 Linux 的文件锁: 多个进程对同一个文件请求排他锁
#   (LOCK_EX) 时，同一时刻只有一个能拿到，其余阻塞等待。锁的载体必须是
#   一个所有进程都能看到的固定路径文件——放 /tmp 是惯例: 全局可见、
#   重启自动清空、内容本身无意义（我们只用它的"锁"属性，不写数据）。
#   五个 backbone 的实验是五个独立 Python 进程，进程内的锁（threading.Lock）
#   管不到它们，所以必须用文件锁这种跨进程机制。
#
# 适用范围: 这把全局锁只服务于旧的 eval_mode="container"（每评测起一个
#   容器）。默认的 persistent 模式没有锁频竞态，用的是下面的每卡锁 _gpu_lock。
# ---------------------------------------------------------------------------
_BENCH_LOCK_PATH = "/tmp/sol_execbench_bench.lock"


@contextmanager
def _flock(path: str):
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)   # 拿不到锁就在这里阻塞，直到持有者释放
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def _bench_lock():
    return _flock(_BENCH_LOCK_PATH)


def _gpu_lock(gpu: str):
    """长驻容器模式下的每卡互斥锁。

    长驻容器只在启动/退出时各锁/解锁一次频率，锁频竞态已不存在，因此
    评测不再需要全机串行；但同一张卡上仍不允许两个评测同时跑（互相抢
    SM 会直接污染计时），所以按卡各设一把文件锁：不同卡的评测并行，
    同一张卡的评测排队。
    """
    return _flock(f"/tmp/sol_execbench_gpu{gpu}.lock")


# ---------------------------------------------------------------------------
# 长驻评测容器（eval_mode: "persistent"）
#
# 每评测一次起一个容器（默认模式）有两个代价: ①每次 ~15 秒冷启动+锁频;
# ②锁频/解锁是全局动作，迫使全机评测串行。长驻方案: 起一个常驻容器
# （入口脚本在启动时锁一次频，容器存活期间频率一直锁着），之后每次评测
# 用 `docker exec` 在容器里开进程，并用 CUDA_VISIBLE_DEVICES 把它钉到
# 指定的卡上。exec 出来的进程看不到入口脚本导出的环境变量，所以要显式
# 传 SOL_EXECBENCH_CLOCKS_LOCKED=1（锁频状态由容器存活这一事实保证）。
# 容器被 docker stop/rm 时，入口脚本的清理钩子会统一解锁频率。
# 2026-08-27 校验实验（单卡独测 vs 8 卡并发, 轻/重负载各一）已通过:
# 错峰启动下逐卡 geomean 偏差 +0.1%~+1.5%，详见 ROADMAP "多卡并发校验结论"。
# 容器生命周期: ensure_persistent_container 按需启动（幂等）; 之后所有评测
# 复用同一容器（编译缓存也随之复用）; 需要结束时手动
# `docker rm -f sol_execbench_persistent`，其入口脚本的清理钩子会统一解锁频率。
# ---------------------------------------------------------------------------
PERSISTENT_NAME = "sol_execbench_persistent"


def ensure_persistent_container(image: str = DEFAULT_IMAGE, name: str = PERSISTENT_NAME) -> None:
    """长驻容器不在运行则启动它（幂等，可并发调用——放在锁里做检查）。"""
    with _flock(f"/tmp/sol_execbench_persist_start.lock"):
        probe = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name=^{name}$"],
            capture_output=True, text=True)
        if probe.stdout.strip():
            return
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        run = subprocess.run(
            ["docker", "run", "-d", "--name", name, "--gpus", "all",
             "--ipc=host", "--privileged",
             "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
             "-v", f"{SOL_ROOT}:{CONTAINER_ROOT}",
             "-e", f"FLASHINFER_TRACE_DIR={CONTAINER_ROOT}/data/flashinfer-trace",
             image, "sleep", "infinity"],
            capture_output=True, text=True)
        if run.returncode != 0:
            raise HarnessError(f"failed to start persistent container: {run.stderr[-500:]}")
        # 给入口脚本几秒完成锁频
        import time as _time
        _time.sleep(8)


# --------------------------------------------------------------------------
# candidate canonicalisation
# --------------------------------------------------------------------------

# (?m) = MULTILINE 模式，让 $ 匹配每一行的行尾；整体含义：每一行里从 # 到行尾的内容（Python 注释）
_COMMENT_RE = re.compile(r"(?m)#.*$")
# ("""|''') 捕获开头的三引号（两种都行）；(?:.|\n)*? 非贪婪地匹配任意内容（含换行）；
# \1 反向引用，要求结尾三引号与开头同款。整体含义：一段三引号字符串（docstring）
_DOCSTRING_RE = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')


def canonical_key(source: str) -> str:
    """给 kernel 源码算一个"去重指纹"——kernel 版的 canonical SMILES。

    问题背景：ExLLM 用"是否见过一模一样的字符串"判断候选重复。分子任务里
    RDKit 会把同一个分子的各种写法规范化成唯一的 canonical SMILES，所以重复
    判定有意义；kernel 源码没有现成的规范形——同一份代码改个注释、调个缩进
    就是"新字符串"。若不处理，repeated_num 永远是 0、uniqueness 永远是 1.0，
    "候选层多样性"这个指标就成了假满分，而"候选层多样 vs principle 层塌缩"
    的对照是整篇论文的核心，指标必须是真的。

    做法（刻意选"句法规范化"，不做语义等价判定）：
      1. 删掉所有三引号字符串（docstring）
      2. 删掉所有 # 注释
      3. 删掉所有空白字符（空格/换行/缩进）
      4. 对剩下的"纯代码字符流"取 SHA-256 哈希作为指纹
    效果：只改注释/格式 → 同一指纹（算重复）；改了任何实际代码 → 不同指纹
    （算新候选）。我们度量的是"提案"的多样性，不是数值输出的多样性，所以
    两个算法不同但结果相同的程序应该算两个候选——句法规范化正好给出这个语义。

    SHA-256 是密码学哈希：任意长度文本 → 固定 64 个十六进制字符的指纹，
    相同输入必得相同指纹，不同输入碰撞概率可忽略。用短指纹进 set 查重，
    比拿几 KB 的源码字符串互相比较又快又省内存。
    """
    stripped = _DOCSTRING_RE.sub("", source)
    stripped = _COMMENT_RE.sub("", stripped)
    stripped = re.sub(r"\s+", "", stripped)
    return hashlib.sha256(stripped.encode("utf-8", "replace")).hexdigest()


# --------------------------------------------------------------------------
# candidate extraction
# --------------------------------------------------------------------------

# 匹配 markdown 代码围栏：``` 开头，(?:python|py)? 是可选的语言标记
# （(?:...) 表示"分组但不捕获"，? 表示可有可无），\s*\n 是可选空格加换行，
# (.*?) 非贪婪捕获围栏内的全部内容，直到遇到收尾的 ```。
# re.DOTALL 让 . 也能匹配换行符，否则 (.*?) 跨不了行、抓不到多行代码。
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_source(raw: str) -> Optional[str]:
    """从候选载荷里抠出 kernel 源码。

    入参 raw = LLM 回复中 <candidate> 与 </candidate> 之间的原始文本
    （框架已经用它自己的正则把这段抠出来了）。虽然 prompt 明确要求"只放
    Python 源码、不要 markdown 围栏"，模型还是经常忍不住包一层 ```python
    ... ```，所以这里两种形式都容忍：
      * 有围栏 → 取最长的那段围栏内容（防止回复里还夹着小段示例代码）
      * 没围栏 → 原文直接当源码
    最后用一个必要条件校验：文件里必须定义 run() 函数。
    该正则 r"(?m)^\\s*def\\s+run\\s*\\(" 的含义：(?m) 开多行模式使 ^ 匹配
    每一行的行首；^\\s* 允许行首有缩进；然后是字面量 def、至少一个空白、
    run、可选空白、左括号。匹配不到 → 这段文本不是合法候选，返回 None
    （调用方会把它计入 invalid_num）。
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None

    fences = _FENCE_RE.findall(text)
    if fences:
        text = max(fences, key=len).strip()

    if not re.search(r"(?m)^\s*def\s+run\s*\(", text):
        return None
    return text


# --------------------------------------------------------------------------
# problem / workload handling
# --------------------------------------------------------------------------


@dataclass
class ProblemSpec:
    """One SOL-ExecBench problem, staged for evaluation."""

    name: str
    problem_dir: str          # host path to the staged problem directory
    definition_name: str      # the `name` field inside definition.json
    n_workloads: int
    t_ref_ms: Optional[float] = None    # geomean reference latency, cached
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def container_problem_dir(self) -> str:
        return _to_container_path(self.problem_dir)


def _to_container_path(host_path: str) -> str:
    host_path = os.path.abspath(host_path)
    if not host_path.startswith(SOL_ROOT):
        raise HarnessError(
            f"path {host_path!r} is outside {SOL_ROOT}; the container only "
            "mounts the SOL-ExecBench tree"
        )
    return CONTAINER_ROOT + host_path[len(SOL_ROOT):]


def stage_problem(
    source_problem_dir: str,
    dest_dir: str,
    workload_uuids: Optional[List[str]] = None,
) -> ProblemSpec:
    """把一道官方题目复制成一份"可评测的工作副本"，顺手修两个坑。

    入参:
      source_problem_dir  官方题目目录，例如
                          SOL-ExecBench/data/benchmark/FlashInfer-Bench/025_rmsnorm_h4096/
                          里面固定有三个文件: definition.json / reference.py / workload.jsonl
                          （definition.json 是数据集里本来就有的，不是从别处拼的）
      dest_dir            工作副本放哪（我们的运行目录下），原始数据永远不动
      workload_uuids      不为 None 时，只保留这份 uuid 清单里的测试用例；
                          None（默认）= 用全部测试用例

    修的两个坑:
      1. FlashInfer-Bench 类别的 26 道题 hf_id 全是空字符串 ""（这一类题来自
         FlashInfer 的评测集、不是从某个 HuggingFace 模型抓的，官方就没填），
         但官方 CLI 自己的数据校验却要求 hf_id 非空——官方数据过不了官方代码。
         补什么都行，因为 hf_id 只是"出处标签"，没有任何下游逻辑读它的值，
         校验只看非空。我们填 "flashinfer-bench" 只是让它可读。
      2. 官方批量脚本 run_dataset.py 裁剪测试用例的方式是 --max-workloads N
         = 取列表"前 N 行"。这里改为按 uuid 显式指定，因为 018/020 等题的
         用例是按尺寸从小到大排的，取前缀会恰好只留下"专家和 LLM 差距最小"
         的小形状，把我们要研究的提升空间从数据里裁掉。
    """
    os.makedirs(dest_dir, exist_ok=True)

    # --- definition.json: 读入 -> 补 hf_id -> 写到工作副本 ---
    # `with open(...) as fh` 是 Python 的文件用法: 打开文件，句柄存到变量 fh，
    # with 块结束后自动关闭。json.load(fh) 把整个 JSON 文件解析成 Python dict。
    with open(os.path.join(source_problem_dir, "definition.json")) as fh:
        definition = json.load(fh)
    if not definition.get("hf_id"):          # 空字符串 "" 在布尔判断里是 False
        definition["hf_id"] = "flashinfer-bench"
    with open(os.path.join(dest_dir, "definition.json"), "w") as fh:
        json.dump(definition, fh)

    # --- reference.py: 原样复制（正确性检查的"标准答案"） ---
    with open(os.path.join(source_problem_dir, "reference.py")) as fh:
        reference_src = fh.read()
    with open(os.path.join(dest_dir, "reference.py"), "w") as fh:
        fh.write(reference_src)

    # --- workload.jsonl: 读入全部用例，按需筛选，写回 ---
    # jsonl = "JSON Lines" 格式: 每一行是一个独立的 JSON 对象（一个测试用例）。
    with open(os.path.join(source_problem_dir, "workload.jsonl")) as fh:
        workloads = [json.loads(line) for line in fh if line.strip()]
    if workload_uuids is not None:
        wanted = set(workload_uuids)
        selected = [w for w in workloads if w.get("uuid") in wanted]
        # 校验: 清单里的 uuid 必须都真实存在，防止配置里抄错 uuid 悄悄少测
        missing = wanted - {w.get("uuid") for w in selected}
        if missing:
            raise HarnessError(f"workload uuids not found in {source_problem_dir}: {sorted(missing)}")
        workloads = selected
    with open(os.path.join(dest_dir, "workload.jsonl"), "w") as fh:
        for w in workloads:
            # json.dumps(w) 把 dict 序列化回 JSON 字符串，+"\n" 让它独占一行
            # ——合起来就是按 jsonl 格式逐行写回
            fh.write(json.dumps(w) + "\n")

    return ProblemSpec(
        name=os.path.basename(source_problem_dir),
        problem_dir=dest_dir,
        definition_name=definition["name"],
        n_workloads=len(workloads),
    )


def build_problem_spec(problem_dir: str) -> str:
    """Render the problem definition into prompt text.

    Kernel candidates are not schema-free the way SMILES strings are: the
    model must know the exact ``run()`` parameter list, the output convention
    and the shape ranges, or every proposal dies on a signature mismatch. This
    text is injected into the task description of every prompt.
    """
    with open(os.path.join(problem_dir, "definition.json")) as fh:
        definition = json.load(fh)
    with open(os.path.join(problem_dir, "workload.jsonl")) as fh:
        workloads = [json.loads(line) for line in fh if line.strip()]

    lines: List[str] = []
    lines.append(f"## Problem: {definition['name']}")
    lines.append(definition.get("description", "").strip())
    lines.append("")

    consts, exprs, variables = [], [], []
    for axis, spec in (definition.get("axes") or {}).items():
        kind = spec.get("type")
        if kind == "const":
            consts.append(f"{axis} = {spec.get('value')}")
        elif kind == "expr":
            exprs.append(f"{axis} = {spec.get('expression')}")
        else:
            values = sorted(w["axes"][axis] for w in workloads if axis in w.get("axes", {}))
            span = f"{values[0]} .. {values[-1]}" if values else "varies"
            variables.append(f"{axis} in [{span}]")
    if consts:
        lines.append("Compile-time constants: " + ", ".join(consts))
    if exprs:
        lines.append("Derived: " + ", ".join(exprs))
    if variables:
        lines.append(f"Variable axes across the {len(workloads)} benchmark workloads: "
                     + ", ".join(variables))
    lines.append("")

    input_names = list((definition.get("inputs") or {}).keys())
    lines.append("Your run() MUST have exactly this signature (same names, same order):")
    lines.append(f"    def run({', '.join(input_names)}):")
    lines.append("")
    lines.append("Inputs:")
    for name, spec in (definition.get("inputs") or {}).items():
        shape = spec.get("shape")
        shape_s = "scalar" if shape is None else "[" + ", ".join(map(str, shape)) + "]"
        desc = spec.get("description", "")
        lines.append(f"  - {name}: {shape_s}, {spec.get('dtype')}" + (f" — {desc}" if desc else ""))
    lines.append("Outputs (return them exactly the way the reference implementation does —")
    lines.append("same structure, names, shapes and dtypes):")
    for name, spec in (definition.get("outputs") or {}).items():
        shape = spec.get("shape")
        shape_s = "scalar" if shape is None else "[" + ", ".join(map(str, shape)) + "]"
        desc = spec.get("description", "")
        lines.append(f"  - {name}: {shape_s}, {spec.get('dtype')}" + (f" — {desc}" if desc else ""))
    lines.append("")

    lines.append("Reference implementation (defines the exact semantics; it is deliberately")
    lines.append("naive and slow — your job is to produce the same results much faster):")
    lines.append("```python")
    lines.append(definition["reference"].strip())
    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Outcome of benchmarking one candidate on one problem."""

    ok: bool
    status: str                     # PASSED, COMPILE_ERROR, ... or HARNESS_ERROR
    latency_ms: Optional[float] = None       # geomean over passed workloads
    per_workload: List[Dict[str, Any]] = field(default_factory=list)
    n_passed: int = 0
    n_total: int = 0
    log: str = ""

    @property
    def all_passed(self) -> bool:
        return self.n_total > 0 and self.n_passed == self.n_total


def _geomean(values: List[float]) -> float:
    import math
    return math.exp(sum(math.log(v) for v in values) / len(values))


def build_solution_json(
    source: str,
    definition_name: str,
    language: str = "triton",
    name: str = "exllm_candidate",
) -> Dict[str, Any]:
    if language not in SINGLE_FILE_LANGUAGES:
        raise HarnessError(f"unsupported language {language!r}")
    return {
        "name": name,
        "definition": definition_name,
        "author": "exllm",
        "description": "candidate generated by ExLLM",
        "spec": {
            "languages": [language],
            "target_hardware": ["B200"],
            "entry_point": "main.py::run",
            "dependencies": [],
            "destination_passing_style": False,
        },
        "sources": [{"path": "main.py", "content": source}],
    }


def _open_for_container(path: str) -> None:
    """容器内进程以镜像固定的 uid 1029 运行；宿主用户 uid 不同的机器上（如 b200-203，
    uid 2850）它写不进宿主创建的 775 目录，评测会以 HARNESS_ERROR 失败。把评测工作目录
    放开为 777 即可，对 uid 相同的机器无副作用。"""
    try:
        os.chmod(path, 0o777)
    except OSError:
        pass


def evaluate_candidate(
    source: str,
    problem: ProblemSpec,
    workdir: str,
    language: str = "triton",
    image: str = DEFAULT_IMAGE,
    gpu: str = "0",
    timeout_s: int = 300,
    compile_timeout_s: int = 300,
    warmup_runs: int = 10,
    iterations: int = 50,
    benchmark_reference: bool = False,
    wall_timeout_s: int = 1800,
    eval_mode: str = "persistent",
) -> EvalResult:
    """把一个候选 kernel 交给官方 harness 编译+计时，返回结构化结果。

    eval_mode:
      * "persistent"（默认）——在长驻容器里 docker exec（见
        ensure_persistent_container 上方的说明），按 CUDA_VISIBLE_DEVICES 钉卡，
        按卡上锁（_gpu_lock）: 不同卡上的评测并行，同一张卡排队。
        快（免 ~15s 冷启动，025 实测 23s→7s）且无锁频竞态。
      * "container"——每次评测起一个新容器（旧模式，回退用）。锁频/解锁是
        全局动作，必须用全局文件锁（_bench_lock）把全机评测串行化。

    五个步骤:
      1. 把源码包装成官方 solution.json 格式，连同评测参数配置一起写盘
      2. 拼 docker 命令，在官方容器里运行 `sol-execbench <题目> --solution ...`
         （锁频、CUPTI 计时、正确性检查、反作弊全部发生在容器内，
          我们自己一毫秒都不测）
      3. 在文件锁保护下执行（persistent 模式用每卡锁，container 模式用全局锁）
      4. 解析容器写出的 traces jsonl（每个测试用例一条记录）
      5. 全部用例通过 → 汇总耗时；任一用例失败 → 整个候选判无效

    约定: 候选写得再烂也不抛异常——失败以 ok=False 的 EvalResult 返回；
    只有 harness 本身坏了（没写出 traces 等）才算 HARNESS_ERROR。
    """
    os.makedirs(workdir, exist_ok=True)
    _open_for_container(workdir)
    # 给本次评测的三个落盘文件起个唯一后缀，避免并发/重跑时互相覆盖
    tag = uuid.uuid4().hex[:12]

    solution_path = os.path.join(workdir, f"sol_{tag}.json")
    config_path = os.path.join(workdir, f"cfg_{tag}.json")
    traces_path = os.path.join(workdir, f"traces_{tag}.jsonl")

    with open(solution_path, "w") as fh:
        json.dump(build_solution_json(source, problem.definition_name, language), fh)
    with open(config_path, "w") as fh:
        json.dump(
            {
                "warmup_runs": warmup_runs,
                "iterations": iterations,
                "lock_clocks": True,
                "benchmark_reference": benchmark_reference,
                "seed": 200,
            },
            fh,
        )

    bench_args = [
        "sol-execbench", problem.container_problem_dir,
        "--solution", _to_container_path(solution_path),
        "--config", _to_container_path(config_path),
        "--lock-clocks",
        "--timeout", str(timeout_s),
        "--compile-timeout", str(compile_timeout_s),
        "-o", _to_container_path(traces_path),
    ]
    if eval_mode == "persistent":
        ensure_persistent_container(image)
        cmd = [
            "docker", "exec", "-w", CONTAINER_ROOT,
            "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
            # exec 进程继承不到入口脚本导出的变量，锁频状态由容器存活保证
            "-e", "SOL_EXECBENCH_CLOCKS_LOCKED=1",
            PERSISTENT_NAME, *bench_args,
        ]
        lock = _gpu_lock(gpu)
    else:
        cmd = [
            "docker", "run", "--rm",
            "--gpus", f"device={gpu}",
            "--ipc=host", "--privileged",
            "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
            "-v", f"{SOL_ROOT}:{CONTAINER_ROOT}",
            "-e", f"FLASHINFER_TRACE_DIR={CONTAINER_ROOT}/data/flashinfer-trace",
            image, *bench_args,
        ]
        lock = _bench_lock()

    try:
        with lock:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=wall_timeout_s
            )
        stdout_tail = (proc.stdout or "")[-4000:]
        stderr_tail = (proc.stderr or "")[-4000:]
    except subprocess.TimeoutExpired:
        return EvalResult(ok=False, status="WALL_TIMEOUT",
                          log=f"docker run exceeded {wall_timeout_s}s")

    if not os.path.exists(traces_path):
        return EvalResult(
            ok=False, status="HARNESS_ERROR",
            log=f"no traces written.\nstdout:{stdout_tail}\nstderr:{stderr_tail}",
        )

    records = []
    with open(traces_path) as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    # ---- 解析逐用例结果。每条记录的 evaluation.status 是九种状态之一
    # (PASSED / COMPILE_ERROR / INCORRECT_NUMERICAL / RUNTIME_ERROR / ...)，
    # PASSED 的记录才带 performance.latency_ms。 ----
    per_workload = []
    latencies = []
    statuses = []
    for rec in records:
        ev = rec.get("evaluation") or {}
        status = ev.get("status", "UNKNOWN")
        statuses.append(status)
        entry = {
            "uuid": (rec.get("workload") or {}).get("uuid"),
            "axes": (rec.get("workload") or {}).get("axes"),
            "status": status,
        }
        perf = ev.get("performance") or {}
        if status == "PASSED" and perf.get("latency_ms"):
            entry["latency_ms"] = perf["latency_ms"]
            entry["reference_latency_ms"] = perf.get("reference_latency_ms")
            latencies.append(perf["latency_ms"])
        per_workload.append(entry)

    n_total = len(records)
    n_passed = sum(1 for s in statuses if s == "PASSED")

    # ---- "全过才有效"门槛 ----
    # n_total = 这道题的测试用例数, n_passed = 其中 PASSED 的条数。
    # 只要有任何一个用例失败，整个候选判无效、不给任何分数。原因: 允许
    # "部分正确"就等于奖励只在自己碰巧能处理的形状上跑得快的 kernel
    # ——例如只支持小 batch 的实现会拿到漂亮的小 batch 均值——这实质上
    # 是一种 reward hacking。官方 leaderboard 同样要求全部用例通过。
    if n_total == 0:
        return EvalResult(ok=False, status="HARNESS_ERROR", n_total=0,
                          log=f"empty trace file.\nstdout:{stdout_tail}")
    if n_passed < n_total:
        # 取第一个失败用例的状态和错误日志作为这次失败的"代表原因"，
        # 供 eval_log 记录与错误回传（get_error_feedback）使用
        failing = next((s for s in statuses if s != "PASSED"), "UNKNOWN")
        first_log = next(
            ((r.get("evaluation") or {}).get("log", "")
             for r in records
             if (r.get("evaluation") or {}).get("status") != "PASSED"),
            "",
        )
        return EvalResult(
            ok=False, status=failing, per_workload=per_workload,
            n_passed=n_passed, n_total=n_total, log=(first_log or "")[:2000],
        )

    return EvalResult(
        ok=True, status="PASSED", latency_ms=_geomean(latencies),
        per_workload=per_workload, n_passed=n_passed, n_total=n_total,
    )
