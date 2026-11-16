"""
工业模型自进化 Harness 系统
===========================

自动化实现"漂移监测—诊断归因—策略选择—实验优化—评估上线—回滚"完整闭环。

核心模块:
  - DriftMonitor: 漂移监测器（批次质心、寿命空档、分段 bias）
  - DiagnosisAttributor: 诊断归因器（协变量漂移、跨轴承错位、长寿命低估）
  - StrategySelector: 策略选择器（类平衡、收缩、长寿命加权）
  - DegradationChecker: 退化检查器（IQR 坍缩、常数等价、分轴承 bias）
  - ExperimentManager: 实验管理器（历史记录、读历史决策）
  - SelfEvolutionHarness: 自进化主循环

用法:
  python -m harness.self_evolution --task B --rounds 10 --checkpoint ... --train ... --validation ... --output ...
"""

from .self_evolution import (
    DriftMonitor,
    DiagnosisAttributor,
    StrategySelector,
    DegradationChecker,
    ExperimentManager,
    SelfEvolutionHarness,
    Strategy,
    ExperimentRecord
)

__version__ = "1.0.0"
__all__ = [
    "DriftMonitor",
    "DiagnosisAttributor", 
    "StrategySelector",
    "DegradationChecker",
    "ExperimentManager",
    "SelfEvolutionHarness",
    "Strategy",
    "ExperimentRecord"
]
