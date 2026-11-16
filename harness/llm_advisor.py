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
0. 回答的第一个字符必须是 "{"，先给 JSON 再考虑别的；不要写思考过程、不要复述任务、
   不要列分析提纲。给出 JSON 后立即结束回答。（这是一次自动化的机器调用，没有人类读者）
1. strategy 必须严格等于候选策略清单中的某个 name，不得自造；
2. param_overrides 只允许出现这些字段：epochs、learning_rate、batch_size、gauss_aug、shrink_a；
3. 你没有真实数据标签的访问权，不得引用验证集自身统计量（例如以验证集标签中位数做中心）；
4. 只输出一个 JSON 对象，不要输出 Markdown、解释性前后缀或代码块围栏。

输出 JSON 结构（示例里的值只是示意，"<候选 name>" 必须替换成候选清单里的真实名字；
禁止原样输出省略号、"..."、"候选之一" 这类占位符）：
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


def _extract_json(text: str) -> str:
    """从模型输出中取出 JSON 对象：先去代码围栏，再退回首个 { 到最后一个 }。"""
    text = _strip_fence(text or "")
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _loose_parse(text: str) -> Dict[str, Any]:
    """尽力从模型输出里取回 strategy / param_overrides / reason / code_notes。

    模型偶发输出残缺 JSON（缺冒号、带中文标点、尾随逗号），严格解析会整轮回退；
    这里按字段做正则兜底，保证"模型真给了答案"时不至于白跑一次调用。
    """
    import re

    text = _extract_json(text or "")
    try:
        value = json.loads(text, strict=False)
        if isinstance(value, dict):
            return value
    except (json.JSONDecodeError, ValueError):
        pass

    out: Dict[str, Any] = {}
    match = re.search(r'"strategy"\s*:\s*"([^"]+)"', text)
    if match:
        out["strategy"] = match.group(1)
    params: Dict[str, Any] = {}
    block = re.search(r'"param_overrides"\s*:\s*{([^{}]*)}', text)
    if block:
        for key, raw in re.findall(r'"([a-z_]+)"\s*:\s*(-?\d+(?:\.\d+)?)', block.group(1)):
            params[key] = float(raw) if "." in raw else int(raw)
    out["param_overrides"] = params
    for key in ("reason", "code_notes"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        if match:
            out[key] = match.group(1)
    return out


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
        self.timeout = float(os.environ.get("PYTRIO_TIMEOUT", "180"))
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
    def _code_excerpt(self, max_chars: int = 900) -> Dict[str, str]:
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
                    )[:600]
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
        for rec in history[-3:]:
            hist.append(
                {
                    "stage": getattr(rec, "stage_id", ""),
                    "strategy": (getattr(rec, "strategy", {}) or {}).get("name", ""),
                    "metric": getattr(rec, "validation_metric", None),
                    "passed": bool(getattr(rec, "passed", False)),
                    "notes": str(getattr(rec, "notes", ""))[:120],
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
        started = time.time()
        last_error = ""
        for attempt in (1, 2):
            # 第 2 次尝试换成极简上下文：推理模型的思考长度与上下文强相关，
            # 完整上下文偶尔会让它一路思考到 token 预算耗尽、content 为空。
            ctx = context if attempt == 1 else {
                "task": self.task,
                "metric_type": metric_type,
                "heuristic_choice": heuristic_name,
                "candidates": [s["name"] for s in context["candidate_strategies"]],
                "last_metric": (context["history"][-1]["metric"] if context["history"] else None),
            }
            payload = {
                "model": self.model,
                "temperature": 0.0,
                # 关掉模型的思考过程：Qwen3 系列会把 token 预算全用在 reasoning，
                # 导致 content 为空（实测 4096/8192 都能被思考吃光）。
                "enable_thinking": False,
                "max_tokens": 2048,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "本轮决策上下文（JSON）："
                        + json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))
                        + "。只输出一行 JSON，strategy 从候选清单里选真实名字，"
                        + f'例如 {{"strategy":"{heuristic_name}","param_overrides":{{}},"reason":"","code_notes":""}}'
                    },
                ],
            }
            try:
                data = self._post(payload)
                content = ""
                reasoning = ""
                finish_reason = ""
                choices = data.get("choices") or []
                if choices:
                    choice = choices[0] or {}
                    message = choice.get("message") or {}
                    finish_reason = str(choice.get("finish_reason") or "")
                    content = message.get("content") or ""
                    reasoning = message.get("reasoning") or ""
                if not str(content).strip() and reasoning:
                    # 推理模型把 JSON 写在 reasoning 里（content 为空）
                    content = reasoning
                # strict=False：模型偶尔在字符串里直接换行；再退正则兜底
                parsed = _loose_parse(content)
                if not isinstance(parsed, dict) or "strategy" not in parsed:
                    raise ValueError("模型输出中没有可识别的 strategy 字段")
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
                    "finish_reason": finish_reason,
                    "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")},
                }
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:200].decode("utf-8", "replace")
                last_error = f"HTTP {exc.code}: {detail}"
                # 4xx（鉴权/余额/参数错误）重试无意义，直接回退
                if exc.code not in (429, 500, 502, 503, 504):
                    break
            except urllib.error.URLError as exc:
                last_error = f"网络错误: {exc.reason}"
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = f"返回内容不是合法 JSON: {exc}"
            except (KeyError, TypeError, IndexError) as exc:
                last_error = f"响应结构异常: {exc}"
            except Exception as exc:  # 超时 / socket 异常：绝不让 harness 崩掉
                last_error = f"调用异常: {type(exc).__name__}: {exc}"
                break

        return {
            "source": "fallback",
            "reason": last_error or "未知错误",
            "model": self.model,
            "endpoint": self.base_url,
            "latency_s": round(time.time() - started, 2),
        }
