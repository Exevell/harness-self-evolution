"""
PyTRIO 大模型顾问（LLM Advisor）
================================

让 Harness 在「策略选择」环节真正调用 PyTRIO 大模型：模型读取基线代码与历史实验
记录，在受约束的决策空间内选策略并给出超参补丁；Harness 校验后落到训练器。

接口为 OpenAI 兼容（POST {base_url}/chat/completions），与 PyTRIO 官方文档一致：
    base_url = https://pytrio.com/api/openai/v1
    api_key  = PyTrIO API Key
    model    = "Qwen/Qwen3.5-4B" 等基础模型，或 "trio://<model>/<version>" 采样权重

环境变量：
    PYTRIO_API_KEY / TRIO_API_KEY / OPENAI_API_KEY   任一即可（按顺序优先）
    PYTRIO_BASE_URL                                  默认 https://pytrio.com/api/openai/v1
    PYTRIO_MODEL                                     默认 Qwen/Qwen3.5-4B
    PYTRIO_TIMEOUT                                   默认 60 秒
    HARNESS_LLM=0                                    显式关闭大模型（回退到启发式）

设计原则：
    * 无 Key / 无网络 / 返回不合法时，一律回退到启发式选择，绝不中断自进化；
    * 模型只能从策略池中选，超参只能在 PARAM_BOUNDS 内改，避免"模型乱改"；
    * 每次决策连同耗时、模型名、原始回复一起记录，便于审计与复现。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "https://pytrio.com/api/openai/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
API_KEY_ENV_NAMES = ("PYTRIO_API_KEY", "TRIO_API_KEY", "OPENAI_API_KEY")

# 允许大模型调整的超参及其取值范围（越界会被夹到边界，类型不对直接丢弃）
PARAM_BOUNDS: Dict[str, Any] = {
    "epochs": ("int", 20, 600),
    "learning_rate": ("float", 1e-4, 0.05),
    "batch_size": ("int", 16, 512),
    "gauss_aug": ("float", 0.0, 1.0),
    "shrink_a": ("float", 0.0, 1.0),
}

SYSTEM_PROMPT = """你是工业模型自进化系统的策略顾问（Harness 的一个环节）。
你的任务：阅读给定基线模型代码、漂移诊断结果与历史实验记录，从候选策略中选出一个最可能提升验证指标的策略，并可选地给出超参补丁。

硬性约束：
1. strategy 必须严格等于候选策略清单中的某个 name，不得自造；
2. param_overrides 只允许出现这些字段：epochs、learning_rate、batch_size、gauss_aug、shrink_a；
3. 你没有真实数据标签的访问权，不得引用验证集自身统计量（例如以验证集标签中位数做中心）；
4. 只输出一个 JSON 对象，不要输出 Markdown、解释性前后缀或代码块围栏。

输出 JSON 结构：
{"strategy": "<候选 name>",
 "param_overrides": {"epochs": 120, "learning_rate": 0.01},
 "reason": "一句话说明为什么选它（引用诊断证据）",
 "code_notes": "一句话说明你从基线代码中读到了什么、据此做了什么判断"}
"""


def _api_key() -> Optional[str]:
    """按「环境变量 → 密钥文件」顺序取 Key；平台注入方式不同也能兼容。"""
    for name in API_KEY_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    key_file = os.environ.get("PYTRIO_API_KEY_FILE", "")
    candidates = [key_file] if key_file else []
    candidates += [
        "/data/pytrio_api_key",
        "/data/secrets/pytrio_api_key",
        "/run/secrets/pytrio_api_key",
    ]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                value = fh.read().strip()
            if value:
                return value
        except OSError:
            continue
    return None


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def _coerce_params(raw: Any) -> Dict[str, Any]:
    """校验并夹取超参补丁；非法项静默丢弃。"""
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        spec = PARAM_BOUNDS.get(key)
        if spec is None or isinstance(value, bool):
            continue
        kind, low, high = spec
        try:
            num = int(value) if kind == "int" else float(value)
        except (TypeError, ValueError):
            continue
        num = max(low, min(high, num))
        out[key] = num
    return out


class LLMAdvisor:
    """封装 PyTRIO 聊天补全调用与决策校验。"""

    def __init__(self, task: str, baselines_dir: Optional[Path] = None, log_fn=None):
        self.task = task
        self.baselines_dir = Path(baselines_dir) if baselines_dir else None
        self.base_url = os.environ.get("PYTRIO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.model = os.environ.get("PYTRIO_MODEL", DEFAULT_MODEL)
        self.timeout = float(os.environ.get("PYTRIO_TIMEOUT", "60"))
        self.api_key = _api_key()
        self.disabled_reason: Optional[str] = None
        if os.environ.get("HARNESS_LLM", "1") == "0":
            self.disabled_reason = "HARNESS_LLM=0"
        elif not self.api_key:
            self.disabled_reason = "未设置 PyTRIO API Key"
        self._log = log_fn or (lambda msg: print(msg))

    # ---------- 可用性 ----------
    def available(self) -> bool:
        return self.disabled_reason is None

    def describe(self) -> str:
        if not self.available():
            return f"未启用（{self.disabled_reason}）"
        return f"{self.model} @ {self.base_url}"

    # ---------- 上下文构建 ----------
    def _code_excerpt(self, max_chars: int = 2400) -> Dict[str, str]:
        """读取基线 model.py 与训练配置片段，作为"读代码"的证据。"""
        result: Dict[str, str] = {}
        if not self.baselines_dir:
            return result
        model_py = self.baselines_dir / "model.py"
        if model_py.exists():
            try:
                text = model_py.read_text(encoding="utf-8", errors="replace")
                result["model_py"] = text[:max_chars]
            except OSError:
                pass
        cfg_dir = self.baselines_dir / "configs"
        if cfg_dir.exists():
            for cfg in sorted(cfg_dir.glob("*.json"))[:1]:
                try:
                    result["config_" + cfg.name] = cfg.read_text(
                        encoding="utf-8", errors="replace"
                    )[:1200]
                except OSError:
                    pass
        return result

    def build_context(
        self,
        diagnosis: Dict[str, Any],
        monitor_result: Dict[str, Any],
        pool: List[Any],
        history: List[Any],
        heuristic_name: str,
        round_num: int,
        metric_type: str,
    ) -> Dict[str, Any]:
        hist = []
        for rec in history[-5:]:
            hist.append(
                {
                    "stage": getattr(rec, "stage_id", ""),
                    "strategy": (getattr(rec, "strategy", {}) or {}).get("name", ""),
                    "metric": getattr(rec, "validation_metric", None),
                    "passed": bool(getattr(rec, "passed", False)),
                    "notes": getattr(rec, "notes", ""),
                }
            )
        pool_desc = []
        for s in pool:
            pool_desc.append(
                {
                    "name": s.name,
                    "epochs": s.epochs,
                    "learning_rate": s.learning_rate,
                    "batch_size": s.batch_size,
                    "sample_weight": s.sample_weight,
                    "gauss_aug": s.gauss_aug,
                    "shrink_a": s.shrink_a,
                    "cosine_anneal": s.cosine_anneal,
                }
            )
        monitor_brief = {
            k: v
            for k, v in (monitor_result or {}).items()
            if not isinstance(v, (list, dict)) or k in ("summary", "dominant", "gap")
        }
        return {
            "task": self.task,
            "round": round_num,
            "metric_type": metric_type,
            "diagnosis": {
                "root_cause": diagnosis.get("root_cause"),
                "severity": diagnosis.get("severity"),
                "action": diagnosis.get("action"),
                "evidence": diagnosis.get("evidence"),
            },
            "monitor": monitor_brief,
            "history": hist,
            "heuristic_choice": heuristic_name,
            "candidate_strategies": pool_desc,
            "code": self._code_excerpt(),
        }

    # ---------- 调用 ----------
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def decide(
        self,
        diagnosis: Dict[str, Any],
        monitor_result: Dict[str, Any],
        pool: List[Any],
        history: List[Any],
        heuristic_name: str,
        round_num: int,
        metric_type: str,
    ) -> Dict[str, Any]:
        """返回决策字典；任何异常都以 source=fallback 返回，不抛出。"""
        if not self.available():
            return {"source": "fallback", "reason": self.disabled_reason}

        context = self.build_context(
            diagnosis, monitor_result, pool, history, heuristic_name, round_num, metric_type
        )
        allowed = {s.name for s in pool}
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "以下是本轮决策上下文（JSON）：\n"
                    + json.dumps(context, ensure_ascii=False, indent=2),
                },
            ],
        }

        started = time.time()
        last_error = ""
        for attempt in (1, 2):
            try:
                data = self._post(payload)
                content = ""
                choices = data.get("choices") or []
                if choices:
                    content = ((choices[0] or {}).get("message") or {}).get("content") or ""
                parsed = json.loads(_strip_fence(content))
                name = str(parsed.get("strategy", "")).strip()
                if name not in allowed:
                    last_error = f"策略不在候选清单: {name!r}"
                    continue
                overrides = _coerce_params(parsed.get("param_overrides"))
                usage = data.get("usage") or {}
                return {
                    "source": "llm",
                    "strategy": name,
                    "param_overrides": overrides,
                    "reason": str(parsed.get("reason", ""))[:400],
                    "code_notes": str(parsed.get("code_notes", ""))[:400],
                    "model": self.model,
                    "endpoint": self.base_url,
                    "latency_s": round(time.time() - started, 2),
                    "attempt": attempt,
                    "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")},
                }
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}"
            except urllib.error.URLError as exc:
                last_error = f"网络错误: {exc.reason}"
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = f"返回内容不是合法 JSON: {exc}"
            except (KeyError, TypeError, IndexError) as exc:
                last_error = f"响应结构异常: {exc}"

        return {
            "source": "fallback",
            "reason": last_error or "未知错误",
            "model": self.model,
            "endpoint": self.base_url,
            "latency_s": round(time.time() - started, 2),
        }
