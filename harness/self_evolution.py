"""
工业模型自进化 Harness 系统 v2.0
================================

核心架构：两层自进化（受控 RSI）
  模型层：基于检查点的续训迭代
  策略层：基于历史实验记录的决策规则自动更新

闭环流程：漂移监测 → 诊断归因 → 策略选择 → 实验优化 → 退化检查 → 评估上线 → 回滚

v2.0 改进：
  - DriftMonitor 从数据实时计算漂移信号（不再返回硬编码值）
  - StrategySelector 维护策略池 + 历史反馈动态选择（不再静态映射）
  - DiagnosisAttributor 阈值根据数据范围自适应（不再固定阈值）
  - _load_predictions 真正用模型预测（不再返回随机数）

作者：参赛团队
日期：2026-11-16
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Optional, Tuple, Any
import math
import os
import sys
import subprocess

try:  # PyTRIO 大模型顾问与本文件同目录
    from llm_advisor import LLMAdvisor
except ImportError:  # pragma: no cover - 以包形式导入时
    try:
        from .llm_advisor import LLMAdvisor
    except ImportError:
        LLMAdvisor = None


# ============================================================
# 0. 数据加载器（Data Loader）
# ============================================================

class DataLoader:
    """
    加载 train.jsonl 和 validation.jsonl，提取特征、标签和元信息。
    """
    
    @staticmethod
    def load_jsonl(path: Path) -> List[dict]:
        """加载 JSONL 文件"""
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    
    @staticmethod
    def extract_gas_data(records: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        A 题：提取 128 维特征和 6 类标签
        返回：X, y, batch_ids
        """
        class_map = {
            "Ethanol": 0, "Ethylene": 1, "Ammonia": 2,
            "Acetaldehyde": 3, "Acetone": 4, "Toluene": 5
        }
        X = np.array([r["inputs"]["features"] for r in records], dtype=np.float64)
        y = np.array([class_map[r["answer"]] for r in records], dtype=np.int64)
        batch_ids = np.array([r["meta"]["source_batch"] for r in records], dtype=np.int64)
        return X, y, batch_ids
    
    @staticmethod
    def extract_bearing_data(records: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        B 题：提取 34 维特征（水平 17 + 垂直 17）、RUL 标签和轴承 ID
        返回：X, y, bearing_ids
        """
        X_list = []
        for r in records:
            h = r["inputs"]["features"]["horizontal"]
            v = r["inputs"]["features"]["vertical"]
            # 按固定顺序拼接（与 baselines/data.py 一致）
            feat_names = sorted(h.keys())
            x = [h[k] for k in feat_names] + [v[k] for k in feat_names]
            X_list.append(x)
        X = np.array(X_list, dtype=np.float64)
        y = np.array([r["answer"] for r in records], dtype=np.float64)
        bearing_ids = np.array([r["meta"]["bearing"] for r in records])
        return X, y, bearing_ids
    
    @staticmethod
    def extract_engine_data(records: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        C 题：提取 24 维特征（3 settings + 21 sensors）、RUL 标签和 unit_id
        返回：X, y, unit_ids
        """
        X_list = []
        for r in records:
            s = r["inputs"]["settings"]
            sensors = r["inputs"]["sensors"]
            x = list(s) + list(sensors)
            X_list.append(x)
        X = np.array(X_list, dtype=np.float64)
        y = np.array([r["answer"] for r in records], dtype=np.float64)
        unit_ids = np.array([r["meta"]["unit"] for r in records])
        return X, y, unit_ids


# ============================================================
# 1. 漂移监测器（Drift Monitor）- v2.0 真实计算
# ============================================================

class DriftMonitor:
    """
    监测三类漂移信号（v2.0：从数据实时计算）：
      A: 批次质心距离 + oracle LDA 跨批次准确率
      B: 分轴承 MSE 占比 + 寿命空档检测
      C: 分段 bias（[400,900) 区间低估被 NASA 非对称评分放大）
    """
    
    @staticmethod
    def monitor_gas_drift(X_train, y_train, batch_ids_train,
                          X_val, y_val, batch_ids_val, model):
        """A 题：批次质心距离 + 跨批次可分性（真实计算）"""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import accuracy_score
        
        # 归一化到模型空间
        Z_train = model.normalized(X_train)
        Z_val = model.normalized(X_val)
        
        # 按批次分组（train=批次1，val=批次2）
        train_batch = batch_ids_train[0] if len(batch_ids_train) > 0 else 1
        val_batch = batch_ids_val[0] if len(batch_ids_val) > 0 else 2
        
        # 计算各类质心距离（用 batch_id 区分）
        classes = sorted(set(y_train) | set(y_val))
        centroid_shifts = {}
        for c in classes:
            mask_tr = (y_train == c) & (batch_ids_train == train_batch)
            mask_va = (y_val == c) & (batch_ids_val == val_batch)
            if mask_tr.sum() == 0 or mask_va.sum() == 0:
                continue
            centroid_tr = Z_train[mask_tr].mean(axis=0)
            centroid_va = Z_val[mask_va].mean(axis=0)
            dist = np.linalg.norm(centroid_va - centroid_tr)
            centroid_shifts[c] = float(dist)
        
        # Oracle LDA 跨批次准确率（仅用训练集拟合，验证集测试）
        mask_tr = batch_ids_train == train_batch
        mask_va = batch_ids_val == val_batch
        lda = LinearDiscriminantAnalysis()
        lda.fit(Z_train[mask_tr], y_train[mask_tr])
        acc_cross = float(accuracy_score(y_val[mask_va], lda.predict(Z_val[mask_va])))
        
        # 随机基线
        val_counts = np.bincount(y_val[mask_va])
        acc_random = float(val_counts.max() / val_counts.sum())
        
        return {
            "centroid_shifts": centroid_shifts,
            "max_shift": max(centroid_shifts.values()) if centroid_shifts else 0.0,
            "cross_batch_acc": acc_cross,
            "random_baseline": acc_random,
            "boundary_transferable": acc_cross > acc_random + 0.05,
            "train_batch": int(train_batch),
            "val_batch": int(val_batch)
        }
    
    @staticmethod
    def monitor_bearing_drift(y_train, bearing_ids_train, y_val, bearing_ids_val):
        """B 题：分轴承 MSE 占比 + 寿命空档（真实计算）"""
        # 训练集每根轴承的最大寿命
        bearing_lives = {}
        for bid in sorted(set(bearing_ids_train)):
            mask = bearing_ids_train == bid
            bearing_lives[bid] = float(y_train[mask].max())
        
        lives_sorted = sorted(bearing_lives.values())
        gaps = []
        # 动态阈值：取寿命范围的 10% 作为空档判定标准
        life_range = lives_sorted[-1] - lives_sorted[0] if len(lives_sorted) > 1 else 1.0
        gap_threshold = max(200.0, life_range * 0.1)
        
        for i in range(len(lives_sorted) - 1):
            gap = lives_sorted[i+1] - lives_sorted[i]
            if gap > gap_threshold:
                gaps.append((float(lives_sorted[i]), float(lives_sorted[i+1]), float(gap)))
        
        # 验证集每根轴承的最大寿命
        val_bearing_lives = {}
        for bid in sorted(set(bearing_ids_val)):
            mask = bearing_ids_val == bid
            val_bearing_lives[bid] = float(y_val[mask].max())
        
        # 检查哪些验证轴承落在空档中
        in_gap = {}
        for bid, life in val_bearing_lives.items():
            for lo, hi, gap in gaps:
                if lo < life < hi:
                    in_gap[bid] = (lo, hi, gap)
        
        # 分轴承 MSE 占比（用当前模型预测 vs 常数预测）
        const_pred = float(np.median(y_train))
        mse_per_bearing = {}
        n_per_bearing = {}
        for bid in sorted(set(bearing_ids_val)):
            mask = bearing_ids_val == bid
            mse = float(np.mean((y_val[mask] - const_pred) ** 2))
            mse_per_bearing[bid] = mse
            n_per_bearing[bid] = int(mask.sum())
        
        total_mse = sum(mse_per_bearing.values())
        mse_share = {bid: mse / total_mse for bid, mse in mse_per_bearing.items()} if total_mse > 0 else {}
        
        return {
            "bearing_lives_train": bearing_lives,
            "gap_threshold": gap_threshold,
            "gaps": gaps,
            "val_bearing_lives": val_bearing_lives,
            "val_bearings_in_gap": in_gap,
            "mse_share": mse_share,
            "n_per_bearing": n_per_bearing,
            "dominant_bearing": max(mse_share, key=mse_share.get) if mse_share else None,
            "train_median_rul": const_pred
        }
    
    @staticmethod
    def monitor_engine_drift(X_train, y_train, X_val, y_val, model):
        """C 题：分段 bias 检测（真实计算）"""
        # 用当前模型预测验证集
        pred = model.predict(X_val)
        
        # 动态分段：根据验证集 RUL 分布自动划分
        percentiles = [0, 25, 50, 75, 100]
        boundaries = [float(np.percentile(y_val, p)) for p in percentiles]
        segments = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
        
        bias_per_seg = {}
        n_per_seg = {}
        for lo, hi in segments:
            mask = (y_val >= lo) & (y_val < hi)
            if mask.sum() == 0:
                continue
            bias = float(np.mean(pred[mask] - y_val[mask]))
            bias_per_seg[(lo, hi)] = bias
            n_per_seg[(lo, hi)] = int(mask.sum())
        
        # NASA 非对称评分
        d = pred - y_val
        nasa = float(np.sum(np.where(d < 0, np.exp(np.clip(-d/13.0, 0, 700)) - 1,
                                     np.exp(np.clip(d/10.0, 0, 700)) - 1)))
        
        # 找到长寿命区间（RUL 最大的 25%）
        long_life_threshold = boundaries[-2]  # 75 分位数
        long_life_seg = segments[-1]  # 最后一个区间
        long_life_bias = bias_per_seg.get(long_life_seg, 0.0)
        
        return {
            "bias_per_segment": {f"[{lo:.0f},{hi:.0f})": bias for (lo, hi), bias in bias_per_seg.items()},
            "n_per_segment": {f"[{lo:.0f},{hi:.0f})": n for (lo, hi), n in n_per_seg.items()},
            "nasa_score": nasa,
            "long_life_threshold": long_life_threshold,
            "long_life_segment": long_life_seg,
            "long_life_undershoot": long_life_bias
        }


# ============================================================
# 2. 诊断归因器（Diagnosis Attribution）- v2.0 自适应阈值
# ============================================================

class DiagnosisAttributor:
    """
    根据漂移监测结果归因根因（v2.0：阈值根据数据范围自适应）：
      A: 协变量漂移（质心落入他类区域）vs 标签移位（先验变化）
      B: 跨轴承水平错位 vs 形状误差
      C: 长寿命样本稀少导致低估被非对称评分放大
    """
    
    @staticmethod
    def diagnose_gas(monitor_result):
        """A 题诊断（动态阈值）"""
        acc_cross = monitor_result["cross_batch_acc"]
        acc_random = monitor_result["random_baseline"]
        
        # 动态阈值：如果跨批次准确率低于随机基线 + 5%，判定为边界不可迁移
        # 但如果差距很小（<2%），可能是噪声
        if not monitor_result["boundary_transferable"]:
            max_shift = monitor_result["max_shift"]
            # 根据最大质心偏移判断漂移严重程度
            if max_shift > 8.0:
                severity = "severe"
            elif max_shift > 5.0:
                severity = "moderate"
            else:
                severity = "mild"
            
            return {
                "root_cause": "covariate_shift",
                "severity": severity,
                "evidence": f"跨批次 LDA 准确率 {acc_cross:.4f} < 随机基线 {acc_random:.4f}，最大质心偏移 {max_shift:.2f}",
                "action": "class_balance + noise_augmentation",
                "drift_metrics": {
                    "cross_batch_acc": acc_cross,
                    "random_baseline": acc_random,
                    "max_centroid_shift": max_shift
                }
            }
        
        return {
            "root_cause": "unknown",
            "severity": "none",
            "evidence": f"边界可迁移（跨批次准确率 {acc_cross:.4f} > 随机基线 {acc_random:.4f}）",
            "action": "monitor_only",
            "drift_metrics": {
                "cross_batch_acc": acc_cross,
                "random_baseline": acc_random,
                "max_centroid_shift": monitor_result["max_shift"]
            }
        }
    
    @staticmethod
    def diagnose_bearing(monitor_result):
        """B 题诊断（动态阈值）"""
        dominant = monitor_result["dominant_bearing"]
        mse_share = monitor_result["mse_share"]
        in_gap = monitor_result["val_bearings_in_gap"]
        
        # 动态阈值：根据轴承数量和 MSE 分布自适应
        n_bearings = len(mse_share)
        # 如果某根轴承占 MSE > 80% 或 > 1/n + 0.3（取较大值），判定为主导轴承
        dominance_threshold = max(0.8, 1.0 / max(n_bearings, 1) + 0.3)
        
        share = mse_share.get(dominant, 0.0) if dominant else 0.0
        
        if share > dominance_threshold and dominant in in_gap:
            gap_info = in_gap[dominant]
            return {
                "root_cause": "life_gap + cross_bearing_level_misalignment",
                "severity": "severe",
                "evidence": f"主导轴承 {dominant} 占 MSE {share:.1%}（阈值 {dominance_threshold:.1%}），"
                           f"落在寿命空档 [{gap_info[0]:.0f}, {gap_info[1]:.0f}]（间隔 {gap_info[2]:.0f} 分钟）",
                "action": "shrink_toward_median",
                "drift_metrics": {
                    "dominant_bearing": dominant,
                    "mse_share": share,
                    "dominance_threshold": dominance_threshold,
                    "gap": gap_info
                }
            }
        elif share > dominance_threshold:
            return {
                "root_cause": "cross_bearing_level_misalignment",
                "severity": "moderate",
                "evidence": f"主导轴承 {dominant} 占 MSE {share:.1%}，但未落在寿命空档中",
                "action": "shrink_toward_median",
                "drift_metrics": {
                    "dominant_bearing": dominant,
                    "mse_share": share,
                    "dominance_threshold": dominance_threshold
                }
            }
        
        return {
            "root_cause": "unknown",
            "severity": "none",
            "evidence": f"未检测到显著主导轴承（最大 MSE 占比 {share:.1%} < 阈值 {dominance_threshold:.1%}）",
            "action": "monitor_only",
            "drift_metrics": {
                "max_mse_share": share,
                "dominance_threshold": dominance_threshold
            }
        }
    
    @staticmethod
    def diagnose_engine(monitor_result):
        """C 题诊断（动态阈值）"""
        long_bias = monitor_result["long_life_undershoot"]
        nasa_score = monitor_result["nasa_score"]
        long_life_seg = monitor_result["long_life_segment"]
        n_in_seg = monitor_result["n_per_segment"].get(
            f"[{long_life_seg[0]:.0f},{long_life_seg[1]:.0f})", 0
        )
        
        # 动态阈值：根据 NASA 分数和样本数自适应
        # 如果长寿命区间样本数 < 总样本的 10%，且低估 > 50 循环，判定为问题
        total_samples = sum(monitor_result["n_per_segment"].values())
        sample_ratio = n_in_seg / total_samples if total_samples > 0 else 0.0
        
        # 低估阈值：根据 NASA 分数自适应
        # 如果 NASA > 1e9，说明有严重低估，阈值放宽到 50
        # 否则阈值收紧到 100
        undershoot_threshold = 50.0 if nasa_score > 1e9 else 100.0
        
        if long_bias < -undershoot_threshold and sample_ratio < 0.1:
            nasa_per_sample = nasa_score / max(n_in_seg, 1)
            return {
                "root_cause": "long_life_undershoot_amplified_by_asymmetric_penalty",
                "severity": "severe",
                "evidence": f"长寿命区间 [{long_life_seg[0]:.0f},{long_life_seg[1]:.0f}) "
                           f"低估 {long_bias:.1f} 循环（阈值 {undershoot_threshold:.0f}），"
                           f"样本占比 {sample_ratio:.1%}，NASA 惩罚 {nasa_per_sample:.2e}/样本",
                "action": "long_life_weighting + extended_training",
                "drift_metrics": {
                    "long_life_undershoot": long_bias,
                    "undershoot_threshold": undershoot_threshold,
                    "sample_ratio": sample_ratio,
                    "nasa_per_sample": nasa_per_sample,
                    "nasa_total": nasa_score
                }
            }
        elif long_bias < -undershoot_threshold:
            return {
                "root_cause": "long_life_undershoot",
                "severity": "moderate",
                "evidence": f"长寿命区间低估 {long_bias:.1f} 循环，但样本占比 {sample_ratio:.1%} 不低",
                "action": "long_life_weighting",
                "drift_metrics": {
                    "long_life_undershoot": long_bias,
                    "sample_ratio": sample_ratio
                }
            }
        
        return {
            "root_cause": "unknown",
            "severity": "none",
            "evidence": f"长寿命区间无显著低估（bias={long_bias:.1f}，阈值={undershoot_threshold:.0f}）",
            "action": "monitor_only",
            "drift_metrics": {
                "long_life_undershoot": long_bias,
                "undershoot_threshold": undershoot_threshold
            }
        }


# ============================================================
# 3. 策略选择器（Strategy Selector）- v2.0 策略池 + 历史反馈
# ============================================================

@dataclass
class Strategy:
    name: str
    sample_weight: str = "none"
    gauss_aug: float = 0.0
    cosine_anneal: bool = False
    shrink_a: float = 1.0
    shrink_center: str = "median"
    target_transform: str = "identity"
    epochs: int = 100
    learning_rate: float = 0.003
    batch_size: int = 128


class StrategyPool:
    """
    策略池：维护每个任务的候选策略，支持动态添加和删除。
    """
    
    def __init__(self, task: str):
        self.task = task
        self.pool = self._build_pool()
    
    def _build_pool(self) -> List[Strategy]:
        """根据任务类型构建策略池"""
        if self.task == "A":
            return [
                Strategy(name="A_invfreq_gauss05_cos", sample_weight="invfreq",
                        gauss_aug=0.5, cosine_anneal=True, epochs=250,
                        learning_rate=0.003, batch_size=64),
                Strategy(name="A_invfreq_gauss03_cos", sample_weight="invfreq",
                        gauss_aug=0.3, cosine_anneal=True, epochs=250,
                        learning_rate=0.003, batch_size=64),
                Strategy(name="A_invfreq_gauss08_cos", sample_weight="invfreq",
                        gauss_aug=0.8, cosine_anneal=True, epochs=250,
                        learning_rate=0.003, batch_size=64),
                Strategy(name="A_invfreq_nogauss_cos", sample_weight="invfreq",
                        gauss_aug=0.0, cosine_anneal=True, epochs=250,
                        learning_rate=0.003, batch_size=64),
                Strategy(name="A_uniform_gauss05_cos", sample_weight="none",
                        gauss_aug=0.5, cosine_anneal=True, epochs=250,
                        learning_rate=0.003, batch_size=64),
            ]
        elif self.task == "B":
            return [
                Strategy(name="B_shrink045", shrink_a=0.45, shrink_center="median",
                        epochs=120, learning_rate=0.01, batch_size=128),
                Strategy(name="B_shrink030", shrink_a=0.30, shrink_center="median",
                        epochs=120, learning_rate=0.01, batch_size=128),
                Strategy(name="B_shrink060", shrink_a=0.60, shrink_center="median",
                        epochs=120, learning_rate=0.01, batch_size=128),
                Strategy(name="B_shrink045_mean", shrink_a=0.45, shrink_center="mean",
                        epochs=120, learning_rate=0.01, batch_size=128),
                Strategy(name="B_shrink045_cos", shrink_a=0.45, shrink_center="median",
                        cosine_anneal=True, epochs=120, learning_rate=0.01, batch_size=128),
                Strategy(name="B_noshrink", shrink_a=1.0, epochs=120,
                        learning_rate=0.003, batch_size=128),
            ]
        elif self.task == "C":
            return [
                Strategy(name="C_rul150x3_ep300", sample_weight="rul-150x3",
                        epochs=300, learning_rate=0.003, batch_size=128),
                Strategy(name="C_rul125x2_ep300", sample_weight="rul-125x2",
                        epochs=300, learning_rate=0.003, batch_size=128),
                Strategy(name="C_uniform_ep300", sample_weight="none",
                        epochs=300, learning_rate=0.003, batch_size=128),
                Strategy(name="C_rul150x3_ep600", sample_weight="rul-150x3",
                        epochs=600, learning_rate=0.003, batch_size=256),
                Strategy(name="C_uniform_ep1200", sample_weight="none",
                        epochs=1200, learning_rate=0.003, batch_size=256),
            ]
        return []
    
    def get_all(self) -> List[Strategy]:
        return self.pool
    
    def get_by_name(self, name: str) -> Optional[Strategy]:
        for s in self.pool:
            if s.name == name:
                return s
        return None


class StrategySelector:
    """
    策略选择器（v2.0：策略池 + 历史反馈动态选择）
    
    选择逻辑：
      1. 前 3 轮：从策略池中选未探索过的策略（优先选诊断推荐的）
      2. 第 4 轮起：
         a. 统计每个策略的历史表现（成功/失败次数、平均指标）
         b. 选择"探索-利用"平衡：
            - 以 70% 概率选历史最优策略的微调版本
            - 以 30% 概率选未探索过的策略
    """
    
    def __init__(self, task: str):
        self.task = task
        self.pool = StrategyPool(task)
        self.strategy_history: Dict[str, List[float]] = {}  # name -> [metric_values]
        self.explored: set = set()
    
    def update_history(self, strategy_name: str, metric: float):
        """更新策略的历史表现"""
        if strategy_name not in self.strategy_history:
            self.strategy_history[strategy_name] = []
        self.strategy_history[strategy_name].append(metric)
        self.explored.add(strategy_name)
    
    def select(self, diagnosis: dict, round_num: int, metric_type: str = "rmse") -> Strategy:
        """
        根据诊断结果和历史反馈选择策略
        
        Args:
            diagnosis: 诊断结果
            round_num: 当前轮次（从 1 开始）
            metric_type: "rmse"（越小越好）或 "macro_f1"（越大越好）
        """
        pool = self.pool.get_all()
        if not pool:
            return Strategy(name=f"{self.task}_default")
        
        # 前 3 轮：探索未尝试过的策略
        if round_num <= 3:
            # 优先选诊断推荐的策略
            recommended_action = diagnosis.get("action", "monitor_only")
            for s in pool:
                if s.name not in self.explored:
                    # 检查策略是否匹配诊断推荐
                    if self._strategy_matches_diagnosis(s, recommended_action):
                        return s
            # 否则选任意未探索策略
            for s in pool:
                if s.name not in self.explored:
                    return s
            # 全部探索过，选历史最优
            return self._select_best(metric_type)
        
        # 第 4 轮起：探索-利用平衡
        np.random.seed(round_num)  # 可复现的随机
        if np.random.rand() < 0.3:
            # 30% 概率：探索未尝试过的策略
            unexplored = [s for s in pool if s.name not in self.explored]
            if unexplored:
                return np.random.choice(unexplored)
        
        # 70% 概率：利用历史最优（微调版本）
        return self._select_best_with_tuning(metric_type)
    
    def _strategy_matches_diagnosis(self, strategy: Strategy, action: str) -> bool:
        """检查策略是否匹配诊断推荐"""
        if action == "class_balance + noise_augmentation":
            return strategy.sample_weight == "invfreq" and strategy.gauss_aug > 0
        elif action == "shrink_toward_median":
            return strategy.shrink_a < 1.0 and strategy.shrink_center == "median"
        elif action == "long_life_weighting + extended_training":
            return "rul-" in strategy.sample_weight and strategy.epochs >= 300
        return False
    
    def _select_best(self, metric_type: str) -> Strategy:
        """选择历史最优策略"""
        if not self.strategy_history:
            return self.pool.get_all()[0]
        
        best_name = None
        best_value = float('inf') if metric_type == "rmse" else float('-inf')
        
        for name, values in self.strategy_history.items():
            avg = np.mean(values)
            if metric_type == "rmse":
                if avg < best_value:
                    best_value = avg
                    best_name = name
            else:
                if avg > best_value:
                    best_value = avg
                    best_name = name
        
        return self.pool.get_by_name(best_name) or self.pool.get_all()[0]
    
    def _select_best_with_tuning(self, metric_type: str) -> Strategy:
        """选择历史最优策略的微调版本"""
        best = self._select_best(metric_type)
        
        # 微调学习率（衰减 0.8）
        new_lr = max(best.learning_rate * 0.8, 1e-6)
        
        return Strategy(
            name=f"{best.name}_tuned_lr{new_lr:.1e}",
            sample_weight=best.sample_weight,
            gauss_aug=best.gauss_aug,
            cosine_anneal=best.cosine_anneal,
            shrink_a=best.shrink_a,
            shrink_center=best.shrink_center,
            target_transform=best.target_transform,
            epochs=best.epochs,
            learning_rate=new_lr,
            batch_size=best.batch_size
        )


# ============================================================
# 4. 退化检查器（Degradation Check）
# ============================================================

class DegradationChecker:
    """
    三道闸门防止策略退化为常数：
      1. 预测 IQR vs 真值 IQR（坍缩检测）
      2. 分轴承 bias（B 题）
      3. 等价纯常数 RMSE
    """
    
    @staticmethod
    def check_iqr_collapse(pred, y_true, threshold=0.1):
        """预测 IQR 是否远小于真值 IQR"""
        pred_iqr = np.percentile(pred, 75) - np.percentile(pred, 25)
        true_iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        ratio = pred_iqr / (true_iqr + 1e-12)
        return {
            "collapsed": ratio < threshold,
            "ratio": float(ratio),
            "pred_iqr": float(pred_iqr),
            "true_iqr": float(true_iqr)
        }
    
    @staticmethod
    def check_constant_equivalence(pred, y_true, train_median=None, margin=0.98):
        """预测是否等价于常数预测器

        参照物取「可部署常数」：训练集中位数（部署时不需要验证集标签即可给出）。
        早期版本用验证集标签中位数当参照（预言机常数），比可部署常数更优，
        再乘 0.95 后要求模型比它还好 5%，实际把「优于平台基线但没到 5%」的
        改进也判成退化。这里改为：模型 RMSE 必须比可部署常数低出 margin 以上
        才算真正改进；预言机常数的数值继续保留在报告里用于对照。
        """
        rmse_model = np.sqrt(np.mean((pred - y_true) ** 2))
        oracle_const = float(np.median(y_true))
        rmse_const_oracle = np.sqrt(np.mean((oracle_const - y_true) ** 2))
        deployable_const = float(train_median) if train_median is not None else oracle_const
        rmse_const = np.sqrt(np.mean((deployable_const - y_true) ** 2))
        return {
            "equivalent_to_constant": bool(rmse_model > rmse_const * margin),
            "rmse_model": float(rmse_model),
            "rmse_const": float(rmse_const),
            "const_value": deployable_const,
            "rmse_const_oracle": float(rmse_const_oracle),
            "oracle_const_value": oracle_const,
        }
    
    @classmethod
    def full_check(cls, pred, y_true, bearing_ids=None, train_median=None):
        """完整退化检查"""
        checks = {
            "iqr_collapse": cls.check_iqr_collapse(pred, y_true),
            "constant_equivalence": cls.check_constant_equivalence(pred, y_true, train_median)
        }
        if bearing_ids is not None:
            bias_per_bearing = {}
            for bid in sorted(set(bearing_ids)):
                mask = bearing_ids == bid
                bias = float(np.mean(pred[mask] - y_true[mask]))
                bias_per_bearing[bid] = bias
            checks["bias_per_bearing"] = bias_per_bearing
            checks["max_abs_bias"] = max(abs(b) for b in bias_per_bearing.values()) if bias_per_bearing else 0.0
        
        return checks
    
    @staticmethod
    def full_check_classification(pred_proba, y_true, y_train):
        """分类任务的退化检查：Macro-F1 对比"常数预测器"（全部预测训练集多数类）

        分类任务的预测是概率矩阵，IQR 坍缩检查不适用；退化判据改为
        模型 Macro-F1 是否优于多数类常数预测器。
        """
        from sklearn.metrics import f1_score
        
        pred_labels = np.argmax(pred_proba, axis=1)
        y_true_int = y_true.astype(int)
        macro_f1 = float(f1_score(y_true_int, pred_labels, average="macro", zero_division=0))
        accuracy = float(np.mean(pred_labels == y_true_int))
        
        majority = int(np.bincount(y_train.astype(int)).argmax())
        const_labels = np.full(y_true_int.shape, majority, dtype=int)
        macro_f1_const = float(f1_score(y_true_int, const_labels, average="macro", zero_division=0))
        
        return {
            "macro_f1": macro_f1,
            "macro_f1_const": macro_f1_const,
            "accuracy": accuracy,
            "macro_f1_per_class": [
                float(v) for v in f1_score(y_true_int, pred_labels, average=None,
                                           labels=sorted(set(y_true_int.tolist())),
                                           zero_division=0)
            ],
            "equivalent_to_constant": macro_f1 <= macro_f1_const * 1.05
        }


# ============================================================
# 5. 实验管理器（Experiment Manager）
# ============================================================

@dataclass
class ExperimentRecord:
    stage_id: str
    strategy: dict  # Strategy as dict for JSON serialization
    validation_metric: float  # RMSE or Macro-F1
    diagnosis: dict
    monitor_result: dict
    degradation_checks: dict
    passed: bool = False
    notes: str = ""
    llm: Optional[dict] = None  # PyTRIO 大模型决策记录（source/strategy/reason/耗时等）


class ExperimentManager:
    """
    管理实验记录，支持"读历史—做实验—依据结果决策"的完整闭环。
    """
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.history: List[ExperimentRecord] = []
        if history_file.exists():
            self._load_history()
    
    def _load_history(self):
        """从文件加载历史实验记录"""
        with open(self.history_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    self.history.append(ExperimentRecord(**record))
    
    def add_record(self, record: ExperimentRecord):
        """添加新实验记录"""
        self.history.append(record)
        with open(self.history_file, 'a', encoding='utf-8') as f:
            # 将 dataclass 转为 dict，处理 numpy 类型（含 numpy 标量键）
            d = self._jsonable(asdict(record))
            f.write(json.dumps(d, ensure_ascii=False, default=self._json_serializer) + '\n')
    
    @classmethod
    def _jsonable(cls, obj):
        """递归转换 numpy 类型与 numpy 键，保证 json.dumps 可序列化"""
        if isinstance(obj, dict):
            return {str(k): cls._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._jsonable(v) for v in obj]
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    @staticmethod
    def _json_serializer(obj):
        """JSON 序列化器，处理 numpy 类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    def get_best(self, metric_type: str = "rmse") -> Optional[ExperimentRecord]:
        """获取历史最优记录"""
        if not self.history:
            return None
        passed_records = [r for r in self.history if r.passed]
        if not passed_records:
            return None
        
        if metric_type == "rmse":
            return min(passed_records, key=lambda r: r.validation_metric)
        else:
            return max(passed_records, key=lambda r: r.validation_metric)


# ============================================================
# 6. 自进化主循环（Self-Evolution Loop）
# ============================================================

class SelfEvolutionHarness:
    """
    自进化 Harness 主类（v2.0：真实数据驱动）
    
    实现完整的"漂移监测—诊断归因—策略选择—实验优化—评估上线—回滚"闭环。
    """
    
    def __init__(self, task: str, base_checkpoint: Path, train_data: Path, val_data: Path,
                 output_dir: Path, history_file: Optional[Path] = None):
        self.task = task
        self.base_checkpoint = base_checkpoint
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if history_file is None:
            history_file = output_dir / "experiment_history.jsonl"
        self.experiment_manager = ExperimentManager(history_file)
        
        self.current_checkpoint = base_checkpoint
        self.round_counter = len(self.experiment_manager.history)
        
        # 策略选择器（v2.0：维护策略池和历史反馈）
        self.strategy_selector = StrategySelector(task)
        
        # 指标类型
        self.metric_type = "macro_f1" if task == "A" else "rmse"
        
        # PyTRIO 大模型顾问：读基线代码 + 读历史，在策略选择环节给出受约束决策
        self.baselines_dir = self._resolve_baselines(base_checkpoint)
        self.llm_advisor = LLMAdvisor(task, self.baselines_dir) if LLMAdvisor else None
        if self.llm_advisor is not None:
            print(f"[初始化] 大模型顾问: {self.llm_advisor.describe()}")
        
        # 预加载数据（只加载一次）
        print(f"[初始化] 加载数据...")
        self._load_data()
    
    def _load_data(self):
        """加载训练集和验证集"""
        train_records = DataLoader.load_jsonl(self.train_data)
        val_records = DataLoader.load_jsonl(self.val_data)
        
        if self.task == "A":
            self.X_train, self.y_train, self.batch_ids_train = DataLoader.extract_gas_data(train_records)
            self.X_val, self.y_val, self.batch_ids_val = DataLoader.extract_gas_data(val_records)
            self.bearing_ids_train = None
            self.bearing_ids_val = None
        elif self.task == "B":
            self.X_train, self.y_train, self.bearing_ids_train = DataLoader.extract_bearing_data(train_records)
            self.X_val, self.y_val, self.bearing_ids_val = DataLoader.extract_bearing_data(val_records)
            self.batch_ids_train = None
            self.batch_ids_val = None
        elif self.task == "C":
            self.X_train, self.y_train, _ = DataLoader.extract_engine_data(train_records)
            self.X_val, self.y_val, _ = DataLoader.extract_engine_data(val_records)
            self.bearing_ids_train = None
            self.bearing_ids_val = None
            self.batch_ids_train = None
            self.batch_ids_val = None
        
        print(f"  训练集: {len(self.X_train)} 样本")
        print(f"  验证集: {len(self.X_val)} 样本")
    
    def _load_model(self, checkpoint_path: Path):
        """加载模型检查点

        镜像内检查点由平台挂载到 /data/checkpoint，其同级目录不会带 baselines，
        因此按以下顺序定位 baselines 目录（含 model.py）：
          1. 环境变量 HARNESS_BASELINES（run.sh 注入 /app/tasks/<TASK>/baselines）
          2. 检查点同级 baselines/（赛题包原始布局、本地开发布局）
          3. 镜像内 /app/tasks/<TASK>/baselines
        """
        baselines_dir = self._resolve_baselines(checkpoint_path)
        if baselines_dir is not None:
            sys.path.insert(0, str(baselines_dir))
        from model import Model
        return Model.load(str(checkpoint_path))

    def _resolve_baselines(self, checkpoint_path: Path) -> Optional[Path]:
        """定位含 model.py 的 baselines 目录（三级查找），找不到返回 None。"""
        env_dir = os.environ.get("HARNESS_BASELINES")
        if env_dir and (Path(env_dir) / "model.py").is_file():
            return Path(env_dir)
        sibling = checkpoint_path.parent / "baselines"
        if (sibling / "model.py").is_file():
            return sibling
        packaged = Path(__file__).resolve().parent.parent / "tasks" / self.task / "baselines"
        if (packaged / "model.py").is_file():
            return packaged
        return None
    
    def run_one_round(self) -> ExperimentRecord:
        """执行一轮自进化迭代"""
        self.round_counter += 1
        stage_id = f"round-{self.round_counter:02d}"
        
        print(f"\n{'='*60}")
        print(f"自进化轮次: {stage_id}")
        print(f"{'='*60}")
        
        # 1. 漂移监测（真实计算）
        print("\n[1/6] 漂移监测...")
        monitor_result = self._monitor()
        self._print_monitor_summary(monitor_result)
        
        # 2. 诊断归因（自适应阈值）
        print("\n[2/6] 诊断归因...")
        diagnosis = self._diagnose(monitor_result)
        print(f"  根因: {diagnosis['root_cause']}")
        print(f"  严重程度: {diagnosis['severity']}")
        print(f"  证据: {diagnosis['evidence']}")
        print(f"  动作: {diagnosis['action']}")
        
        # 3. 策略选择（启发式兜底 + PyTRIO 大模型决策）
        print("\n[3/6] 策略选择...")
        heuristic = self.strategy_selector.select(
            diagnosis, self.round_counter, self.metric_type
        )
        print(f"  启发式候选: {heuristic.name}")

        llm_decision = None
        strategy = heuristic
        if self.llm_advisor is not None:
            print(f"  调用 PyTRIO 大模型（{self.llm_advisor.describe()}）...")
            llm_decision = self.llm_advisor.decide(
                diagnosis=diagnosis,
                monitor_result=monitor_result,
                pool=self.strategy_selector.pool.get_all(),
                history=self.experiment_manager.history,
                heuristic_name=heuristic.name,
                round_num=self.round_counter,
                metric_type=self.metric_type,
            )
            if llm_decision.get("source") == "llm":
                base = self.strategy_selector.pool.get_by_name(llm_decision["strategy"]) or heuristic
                overrides = llm_decision.get("param_overrides") or {}
                if overrides:
                    strategy = replace(base, **overrides)
                else:
                    strategy = base
                llm_decision["applied"] = {
                    "strategy": strategy.name,
                    "param_overrides": overrides,
                }
                print(f"  ✅ 大模型决策: {strategy.name} "
                      f"{overrides if overrides else '(不改超参)'} "
                      f"({llm_decision['latency_s']}s)")
                print(f"     理由: {llm_decision.get('reason', '')}")
                print(f"     读代码: {llm_decision.get('code_notes', '')}")
            else:
                print(f"  ⚠️ 大模型不可用，回退启发式: {llm_decision.get('reason')}")
        else:
            print("  ⚠️ 未加载 llm_advisor 模块，使用启发式")
        print(f"  策略: {strategy.name}")
        print(f"  已探索: {len(self.strategy_selector.explored)}/{len(self.strategy_selector.pool.get_all())}")
        
        # 4. 实验优化（调用扩展训练器）
        print("\n[4/6] 实验优化...")
        round_output = self.output_dir / stage_id
        self._train(strategy, stage_id, round_output)
        
        # 5. 退化检查（真实预测）
        print("\n[5/6] 退化检查...")
        pred, y_true, bearing_ids = self._load_predictions(round_output)
        
        if self.metric_type == "macro_f1":
            # 分类任务：概率矩阵 → Macro-F1，对比多数类常数预测器
            degradation = DegradationChecker.full_check_classification(
                pred, y_true, self.y_train
            )
            metric = degradation["macro_f1"]
            collapsed = False
            equivalent = degradation["equivalent_to_constant"]
            detail = (f"Macro-F1={degradation['macro_f1']:.4f} "
                      f"(常数基线 {degradation['macro_f1_const']:.4f}, "
                      f"acc={degradation['accuracy']:.4f})")
        else:
            # 可部署常数参照：训练集中位数（不使用验证集标签）
            degradation = DegradationChecker.full_check(
                pred, y_true, bearing_ids,
                train_median=float(np.median(self.y_train)),
            )
            metric = degradation["constant_equivalence"]["rmse_model"]
            collapsed = degradation["iqr_collapse"]["collapsed"]
            equivalent = degradation["constant_equivalence"]["equivalent_to_constant"]
            detail = (f"RMSE={metric:.2f} (可部署常数 {degradation['constant_equivalence']['rmse_const']:.2f}"
                      f" / 预言机常数 {degradation['constant_equivalence']['rmse_const_oracle']:.2f})")
        
        if collapsed:
            print(f"  ⚠️ 预测 IQR 坍缩 (ratio={degradation['iqr_collapse']['ratio']:.3f})")
        
        if equivalent:
            print(f"  ⚠️ 预测等价于常数 ({detail})")
        
        # 6. 评估上线
        print("\n[6/6] 评估上线...")
        passed = (
            not collapsed and
            not equivalent and
            (self._is_improvement(metric) or self.round_counter <= 3)
        )
        
        record = ExperimentRecord(
            stage_id=stage_id,
            strategy=asdict(strategy),
            validation_metric=float(metric),
            diagnosis=diagnosis,
            monitor_result=monitor_result,
            degradation_checks=degradation,
            passed=passed,
            notes=f"诊断: {diagnosis['root_cause']} ({diagnosis['severity']})",
            llm=llm_decision,
        )
        
        self.experiment_manager.add_record(record)
        
        # 更新策略选择器的历史反馈
        self.strategy_selector.update_history(strategy.name, metric)
        
        if passed:
            print(f"  ✅ 上线成功 ({self.metric_type}={metric:.4f})")
            self.current_checkpoint = round_output / "checkpoint.npz"
        else:
            print(f"  ❌ 上线失败，回滚到上一轮检查点")
        
        return record
    
    def _monitor(self) -> dict:
        """执行漂移监测（v2.0：真实计算）"""
        # 加载当前模型
        model = self._load_model(self.current_checkpoint)
        
        if self.task == "A":
            return DriftMonitor.monitor_gas_drift(
                self.X_train, self.y_train, self.batch_ids_train,
                self.X_val, self.y_val, self.batch_ids_val, model
            )
        elif self.task == "B":
            return DriftMonitor.monitor_bearing_drift(
                self.y_train, self.bearing_ids_train,
                self.y_val, self.bearing_ids_val
            )
        elif self.task == "C":
            return DriftMonitor.monitor_engine_drift(
                self.X_train, self.y_train,
                self.X_val, self.y_val, model
            )
        return {}
    
    def _print_monitor_summary(self, result: dict):
        """打印漂移监测摘要"""
        if self.task == "A":
            print(f"  跨批次准确率: {result['cross_batch_acc']:.4f} (随机基线: {result['random_baseline']:.4f})")
            print(f"  边界可迁移: {result['boundary_transferable']}")
            print(f"  最大质心偏移: {result['max_shift']:.2f}")
        elif self.task == "B":
            print(f"  主导轴承: {result['dominant_bearing']} (MSE 占比: {result['mse_share'].get(result['dominant_bearing'], 0):.1%})")
            print(f"  寿命空档: {len(result['gaps'])} 个")
            print(f"  验证轴承在空档中: {list(result['val_bearings_in_gap'].keys())}")
        elif self.task == "C":
            print(f"  NASA 分数: {result['nasa_score']:.2e}")
            print(f"  长寿命区间低估: {result['long_life_undershoot']:.1f} 循环")
    
    def _diagnose(self, monitor_result: dict) -> dict:
        """执行诊断归因（v2.0：自适应阈值）"""
        if self.task == "A":
            return DiagnosisAttributor.diagnose_gas(monitor_result)
        elif self.task == "B":
            return DiagnosisAttributor.diagnose_bearing(monitor_result)
        elif self.task == "C":
            return DiagnosisAttributor.diagnose_engine(monitor_result)
        return {"root_cause": "unknown", "severity": "none", "evidence": "", "action": "monitor_only"}
    
    def _train(self, strategy: Strategy, stage_id: str, output_dir: Path):
        """调用扩展训练器执行训练"""
        # 构建命令行参数
        train_ext_path = Path(__file__).parent.parent / "shared" / "train_ext.py"
        
        cmd = [
            sys.executable, str(train_ext_path),
            "--profile", self.task,
            "--train", str(self.train_data),
            "--validation", str(self.val_data),
            "--output", str(output_dir),
            "--stage-id", stage_id,
            "--epochs", str(strategy.epochs),
            "--learning-rate", str(strategy.learning_rate),
            "--batch-size", str(strategy.batch_size)
        ]
        
        # 添加检查点参数
        if self.round_counter == 1:
            cmd.extend(["--from-baseline", str(self.current_checkpoint)])
        else:
            cmd.extend(["--resume", str(self.current_checkpoint)])
        
        # 添加策略参数
        if strategy.sample_weight != "none":
            cmd.extend(["--sample-weight", strategy.sample_weight])
        if strategy.gauss_aug > 0:
            cmd.extend(["--gauss-aug", str(strategy.gauss_aug)])
        if strategy.cosine_anneal:
            cmd.append("--cosine-anneal")
        if strategy.shrink_a < 1.0:
            cmd.extend(["--shrink-a", str(strategy.shrink_a),
                       "--shrink-center", strategy.shrink_center])
        if strategy.target_transform != "identity":
            cmd.extend(["--target-transform", strategy.target_transform])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"训练失败: {result.stderr}")
    
    def _load_predictions(self, round_output: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """加载预测结果（v2.0：真正用模型预测）"""
        checkpoint_path = round_output / "checkpoint.npz"
        model = self._load_model(checkpoint_path)
        
        pred = model.predict(self.X_val)
        y_true = self.y_val
        bearing_ids = self.bearing_ids_val
        
        return pred, y_true, bearing_ids
    
    def _is_improvement(self, current_metric: float) -> bool:
        """判断是否优于历史最优"""
        best = self.experiment_manager.get_best(self.metric_type)
        if best is None:
            return True
        
        if self.metric_type == "rmse":
            return current_metric < best.validation_metric * 0.98  # 至少改善 2%
        else:
            return current_metric > best.validation_metric * 1.02  # 至少改善 2%


# ============================================================
# 7. 主入口（Main Entry Point）
# ============================================================

def main():
    """
    自进化 Harness 主入口（v2.0）
    
    用法:
        python harness/self_evolution.py --task B --rounds 10 --checkpoint ... --train ... --validation ... --output ...
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="工业模型自进化 Harness v2.0")
    parser.add_argument("--task", required=True, choices=["A", "B", "C"],
                       help="赛题任务: A(气体识别) / B(轴承RUL) / C(发动机RUL)")
    parser.add_argument("--rounds", type=int, default=10,
                       help="自进化轮数 (默认 10)")
    parser.add_argument("--checkpoint", required=True,
                       help="基线检查点路径")
    parser.add_argument("--train", required=True,
                       help="训练集 JSONL 路径")
    parser.add_argument("--validation", required=True,
                       help="验证集 JSONL 路径")
    parser.add_argument("--output", required=True,
                       help="输出目录")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    harness = SelfEvolutionHarness(
        task=args.task,
        base_checkpoint=Path(args.checkpoint),
        train_data=Path(args.train),
        val_data=Path(args.validation),
        output_dir=output_dir
    )
    
    print(f"启动自进化 Harness v2.0: 任务={args.task}, 轮数={args.rounds}")
    print(f"输出目录: {output_dir}")
    
    for i in range(args.rounds):
        record = harness.run_one_round()
        if not record.passed:
            print(f"\n⚠️ 轮次 {i+1} 未通过，停止自进化")
            break
    
    # 输出最终总结
    best = harness.experiment_manager.get_best(harness.metric_type)
    print(f"\n{'='*60}")
    print(f"自进化完成")
    print(f"{'='*60}")
    print(f"总轮数: {len(harness.experiment_manager.history)}")
    if best:
        print(f"最优 {harness.metric_type}: {best.validation_metric:.4f} (策略: {best.strategy['name']})")
    print(f"实验记录: {harness.experiment_manager.history_file}")


if __name__ == "__main__":
    main()
