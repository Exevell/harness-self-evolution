#!/usr/bin/env python
"""
将实验历史上传到 SwanLab

用法:
  python upload_to_swanlab.py --history submission_runs/B/experiment_history.jsonl
"""

import argparse
import json
from pathlib import Path

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


def upload_to_swanlab(history_file: Path, project_name: str = "industrial-model-evolution"):
    """上传实验历史到 SwanLab"""
    
    if not SWANLAB_AVAILABLE:
        print("❌ 错误: SwanLab 未安装")
        print("   请运行: pip install swanlab")
        print("   文档: https://docs.swanlab.cn")
        return
    
    # 读取历史记录
    if not history_file.exists():
        print(f"❌ 错误: 历史记录文件不存在: {history_file}")
        return
    
    records = []
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        print("⚠️ 警告: 历史记录为空")
        return
    
    print(f"📂 读取到 {len(records)} 条实验记录")
    
    # 提取任务类型
    task = None
    for record in records:
        if 'stage_id' in record:
            if 'round' in record['stage_id']:
                # 从输出目录推断任务
                parent = history_file.parent.name
                if parent in ['A', 'B', 'C']:
                    task = parent
                    break
    
    if task is None:
        task = "unknown"
    
    print(f"🎯 任务: {task}")
    print(f"📊 项目: {project_name}")
    print()
    
    # 初始化 SwanLab
    swanlab.init(
        project=project_name,
        experiment_name=f"{task}_self_evolution",
        config={
            "task": task,
            "total_rounds": len(records),
            "harness_version": "1.0.0"
        }
    )
    
    # 上传每条记录
    for record in records:
        stage_id = record.get('stage_id', 'unknown')
        strategy = record.get('strategy', {})
        
        # 基础指标
        metrics = {
            f"{stage_id}/validation_rmse": record.get('validation_rmse', 0.0),
            f"{stage_id}/passed": record.get('passed', False),
            f"{stage_id}/strategy_name": strategy.get('name', 'unknown'),
            f"{stage_id}/learning_rate": strategy.get('learning_rate', 0.0),
            f"{stage_id}/batch_size": strategy.get('batch_size', 0),
            f"{stage_id}/epochs": strategy.get('epochs', 0),
        }
        
        # Macro-F1（如果有）
        if record.get('validation_macro_f1') is not None:
            metrics[f"{stage_id}/validation_macro_f1"] = record['validation_macro_f1']
        
        # NASA 分数（如果有）
        if record.get('nasa_score') is not None:
            metrics[f"{stage_id}/nasa_score"] = record['nasa_score']
        
        # 退化检查
        degradation = record.get('degradation_checks', {})
        if degradation:
            # IQR 坍缩
            iqr_collapse = degradation.get('iqr_collapse', {})
            if iqr_collapse:
                metrics[f"{stage_id}/iqr_ratio"] = iqr_collapse.get('ratio', 0.0)
                metrics[f"{stage_id}/iqr_collapsed"] = iqr_collapse.get('collapsed', False)
            
            # 常数等价
            const_eq = degradation.get('constant_equivalence', {})
            if const_eq:
                metrics[f"{stage_id}/rmse_model"] = const_eq.get('rmse_model', 0.0)
                metrics[f"{stage_id}/rmse_const"] = const_eq.get('rmse_const', 0.0)
                metrics[f"{stage_id}/equivalent_to_constant"] = const_eq.get('equivalent_to_constant', False)
            
            # 分轴承 bias
            bias_per_bearing = degradation.get('bias_per_bearing', {})
            if bias_per_bearing:
                metrics[f"{stage_id}/max_abs_bias"] = degradation.get('max_abs_bias', 0.0)
                for bearing, bias in bias_per_bearing.items():
                    metrics[f"{stage_id}/bias_{bearing}"] = bias
        
        # 策略参数
        metrics[f"{stage_id}/sample_weight"] = strategy.get('sample_weight', 'none')
        metrics[f"{stage_id}/gauss_aug"] = strategy.get('gauss_aug', 0.0)
        metrics[f"{stage_id}/cosine_anneal"] = strategy.get('cosine_anneal', False)
        metrics[f"{stage_id}/shrink_a"] = strategy.get('shrink_a', 1.0)
        
        # 上传
        swanlab.log(metrics)
        print(f"  ✅ 上传 {stage_id}")
    
    # 上传最优策略
    print("\n📈 计算最优策略...")
    passed_records = [r for r in records if r.get('passed', False)]
    if passed_records:
        # 根据任务类型选择指标
        if task == "A":
            best = max(passed_records, key=lambda r: r.get('validation_macro_f1', 0.0))
            swanlab.log({"best/macro_f1": best['validation_macro_f1']})
            print(f"  🏆 最优 Macro-F1: {best['validation_macro_f1']:.4f} ({best['stage_id']})")
        else:
            best = min(passed_records, key=lambda r: r.get('validation_rmse', float('inf')))
            swanlab.log({"best/rmse": best['validation_rmse']})
            print(f"  🏆 最优 RMSE: {best['validation_rmse']:.2f} ({best['stage_id']})")
        
        swanlab.log({
            "best/strategy_name": best['strategy']['name'],
            "best/learning_rate": best['strategy']['learning_rate'],
            "best/batch_size": best['strategy']['batch_size']
        })
    else:
        print("  ⚠️ 没有通过的实验记录")
    
    print("\n✅ 上传完成!")
    print(f"📊 查看结果: https://swanlab.cn/@your-username/{project_name}")
    
    swanlab.finish()


def main():
    parser = argparse.ArgumentParser(
        description="将实验历史上传到 SwanLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python upload_to_swanlab.py --history submission_runs/B/experiment_history.jsonl
  python upload_to_swanlab.py --history submission_runs/A/experiment_history.jsonl --project my-project
        """
    )
    
    parser.add_argument(
        "--history",
        type=Path,
        required=True,
        help="实验历史记录文件路径 (experiment_history.jsonl)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="industrial-model-evolution",
        help="SwanLab 项目名称 (默认 industrial-model-evolution)"
    )
    
    args = parser.parse_args()
    
    upload_to_swanlab(args.history, args.project)


if __name__ == "__main__":
    main()
