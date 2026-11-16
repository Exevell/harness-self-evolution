#!/usr/bin/env python
"""
一键运行自进化 Harness 的入口脚本

用法:
  python run_harness.py --task B --rounds 10
  python run_harness.py --task A --rounds 10
  python run_harness.py --task C --rounds 10
"""

import argparse
import sys
from pathlib import Path
from harness.self_evolution import SelfEvolutionHarness


# 赛题包路径配置
TASK_CONFIGS = {
    "A": {
        "name": "气体识别",
        "package": "cpu-a-guide-20261110-v1",
        "metric": "macro_f1"
    },
    "B": {
        "name": "轴承 RUL",
        "package": "cpu-b-rul-20261110-v1", 
        "metric": "rmse"
    },
    "C": {
        "name": "发动机 RUL",
        "package": "cpu-c-guide-20261110-v1",
        "metric": "rmse"
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="一键运行自进化 Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_harness.py --task B --rounds 10
  python run_harness.py --task A --rounds 10 --base-dir /path/to/packages
        """
    )
    
    parser.add_argument(
        "--task", 
        required=True, 
        choices=["A", "B", "C"],
        help="赛题任务: A(气体识别) / B(轴承RUL) / C(发动机RUL)"
    )
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=10,
        help="自进化轮数 (默认 10)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"C:\Users\1\Downloads"),
        help="赛题包所在基础目录 (默认 C:\\Users\\1\\Downloads)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录 (默认 submission_runs/{task})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有实验记录恢复"
    )
    
    args = parser.parse_args()
    
    # 加载任务配置
    config = TASK_CONFIGS[args.task]
    package_dir = args.base_dir / config["package"]
    
    # 检查赛题包是否存在
    if not package_dir.exists():
        print(f"❌ 错误: 赛题包不存在: {package_dir}")
        print(f"   请确认赛题包已下载到 {args.base_dir}")
        sys.exit(1)
    
    # 设置路径
    checkpoint = package_dir / "checkpoint.npz"
    train_data = package_dir / "public" / "train.jsonl"
    val_data = package_dir / "public" / "validation.jsonl"
    
    # 检查文件是否存在
    for path, desc in [(checkpoint, "检查点"), (train_data, "训练集"), (val_data, "验证集")]:
        if not path.exists():
            print(f"❌ 错误: {desc}不存在: {path}")
            sys.exit(1)
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = Path("submission_runs") / args.task
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打印配置
    print("="*70)
    print(f"自进化 Harness - 一键运行脚本")
    print("="*70)
    print(f"任务: {args.task} ({config['name']})")
    print(f"轮数: {args.rounds}")
    print(f"赛题包: {package_dir}")
    print(f"检查点: {checkpoint}")
    print(f"训练集: {train_data}")
    print(f"验证集: {val_data}")
    print(f"输出目录: {args.output_dir}")
    print(f"评估指标: {config['metric']}")
    print("="*70)
    print()
    
    # 创建 Harness
    harness = SelfEvolutionHarness(
        task=args.task,
        base_checkpoint=checkpoint,
        train_data=train_data,
        val_data=val_data,
        output_dir=args.output_dir
    )
    
    # 如果是恢复模式，跳过已完成的轮次
    start_round = 0
    if args.resume:
        start_round = len(harness.experiment_manager.history)
        print(f"📂 恢复模式: 从第 {start_round + 1} 轮继续")
        print()
    
    # 执行自进化
    try:
        for i in range(start_round, args.rounds):
            record = harness.run_one_round()
            
            # 打印当前轮次结果
            print(f"\n📊 轮次 {i+1}/{args.rounds} 结果:")
            if config["metric"] == "rmse":
                print(f"   RMSE: {record.validation_metric:.2f}")
            else:
                print(f"   Macro-F1: {record.validation_metric:.4f}")
            print(f"   通过: {'✅' if record.passed else '❌'}")
            print(f"   策略: {record.strategy['name']}")
            print()
            
            if not record.passed:
                # 单轮未通过只是回滚，不能终止自进化：继续用下一策略探索，
                # 这样才能形成 ≥3 次「读历史—做实验—依据结果决策」的闭环。
                print(f"⚠️  轮次 {i+1} 未通过退化检查，已回滚，继续下一轮探索")
                continue
        
        # 输出最终总结
        print("="*70)
        print("自进化完成")
        print("="*70)
        
        best = harness.experiment_manager.get_best(metric_type=config["metric"])
        print(f"总轮数: {len(harness.experiment_manager.history)}")
        print(f"通过轮数: {sum(1 for r in harness.experiment_manager.history if r.passed)}")
        
        if best:
            if config["metric"] == "rmse":
                print(f"最优 RMSE: {best.validation_metric:.2f}")
            else:
                print(f"最优 Macro-F1: {best.validation_metric:.4f}")
            print(f"最优策略: {best.strategy['name']}")
            print(f"最优轮次: {best.stage_id}")
        
        print(f"\n实验记录: {harness.experiment_manager.history_file}")
        print(f"输出目录: {args.output_dir}")
        
        # 生成 SwanLab 上传命令
        print("\n" + "="*70)
        print("下一步: 上传到 SwanLab")
        print("="*70)
        print(f"python upload_to_swanlab.py --history {harness.experiment_manager.history_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，已保存当前进度")
        print(f"可使用 --resume 参数恢复: python run_harness.py --task {args.task} --resume")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        print(f"可使用 --resume 参数恢复: python run_harness.py --task {args.task} --resume")
        sys.exit(1)


if __name__ == "__main__":
    main()
