# analyze_results.py

import os
import json
import pandas as pd
from typing import List, Dict, Any

# --- 配置 ---
OUTPUTS_DIR = './outputs'  # 指向你的实验输出目录
TARGET_DATASETS = ['small', 'small_3x', 'small_30x']


def find_and_parse_results(outputs_dir: str) -> List[Dict[str, Any]]:
    """
    遍历输出目录，查找所有 results.json 文件并解析它们。
    """
    all_results = []
    print(f"正在扫描目录: {outputs_dir}")

    if not os.path.isdir(outputs_dir):
        print(f"错误: 目录 '{outputs_dir}' 不存在。请确保路径正确。")
        return []

    for root, _, files in os.walk(outputs_dir):
        if 'results.json' in files:
            results_path = os.path.join(root, 'results.json')
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)

                    params = data.get('args', {})
                    metrics = data.get('test_metrics', {})

                    if not params or not metrics or not params.get('model'):
                        continue

                    flat_record = {
                        'model': params.get('model'),
                        'dataset': params.get('dataset_name'),
                        'voltage_ve': metrics.get('voltage_ve'),
                        'spike_f1': metrics.get('spike_f1'),
                        **params
                    }
                    all_results.append(flat_record)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"警告: 无法解析或处理文件: {results_path}。错误: {e}")

    return all_results


def generate_report(df: pd.DataFrame, primary_metric: str):
    """
    根据给定的主要指标生成并打印分析报告。
    """
    secondary_metric = 'spike_f1' if primary_metric == 'voltage_ve' else 'voltage_ve'

    print("=" * 80)
    print(f"实验结果分析报告 (最佳模型按 '{primary_metric}' 排序)")
    print("=" * 80)

    for dataset in TARGET_DATASETS:
        print(f"\n--- 数据集: {dataset} ---\n")

        df_tcn = df[(df['model'] == 'tcn') & (df['dataset'] == dataset)]
        df_s4d = df[(df['model'] == 's4d') & (df['dataset'] == dataset)]

        best_tcn = df_tcn.loc[df_tcn[primary_metric].idxmax()] if not df_tcn.empty and df_tcn[
            primary_metric].notna().any() else None
        best_s4d = df_s4d.loc[df_s4d[primary_metric].idxmax()] if not df_s4d.empty and df_s4d[
            primary_metric].notna().any() else None

        if best_tcn is not None:
            print("  [TCN] 最佳性能:")
            print(f"    -  {primary_metric}: {best_tcn[primary_metric]:.4f}")
            print(f"    -  {secondary_metric} (参考): {best_tcn[secondary_metric]:.4f}")
            print("    - 最佳参数配置:")
            tcn_params = SEARCH_SPACE[0]['params'].keys()
            for p in tcn_params:
                print(f"        --{p:<18} {best_tcn.get(p, 'N/A')}")
        else:
            print("  [TCN] 未找到该数据集的有效结果。")

        print("-" * 50)

        if best_s4d is not None:
            print("  [S4D] 最佳性能:")
            print(f"    -  {primary_metric}: {best_s4d[primary_metric]:.4f}")
            print(f"    -  {secondary_metric} (参考): {best_s4d[secondary_metric]:.4f}")
            print("    - 最佳参数配置:")
            s4d_params = SEARCH_SPACE[1]['params'].keys()
            for p in s4d_params:
                print(f"        --{p:<18} {best_s4d.get(p, 'N/A')}")
        else:
            print("  [S4D] 未找到该数据集的有效结果。")


def main():
    """
    主函数，执行分析并为多个指标打印报告。
    """
    results_data = find_and_parse_results(OUTPUTS_DIR)

    if not results_data:
        print("未找到任何有效的实验结果。")
        return

    df = pd.DataFrame(results_data)
    df_filtered = df[df['dataset'].isin(TARGET_DATASETS)].copy()

    if df_filtered.empty:
        print(f"在日志中未找到与目标数据集 {TARGET_DATASETS} 相关的结果。")
        return

    print(f"\n找到了 {len(df_filtered)} 条与目标数据集相关的实验记录。")

    # --- 自动为两个主要指标生成报告 ---
    generate_report(df_filtered, 'voltage_ve')
    generate_report(df_filtered, 'spike_f1')

    print("\n" + "=" * 80)
    print("分析完成。")
    print("=" * 80)


if __name__ == '__main__':
    SEARCH_SPACE: List[Dict[str, Any]] = [
        {
            'model': 'tcn',
            'params': {
                'lr': [1e-3, 5e-4], 'hidden_channels': [32, 48, 64], 'num_levels': [4, 8],
                'dropout': [0.2], 'input_kernel_size': [7, 15, 31, 53], 'weight_decay': [1e-3],
                'fusion_mode': ['add', 'ablate']
            }
        },
        {
            'model': 's4d',
            'params': {
                'lr': [2e-3, 1e-3], 'd_model': [128], 'n_layers': [2, 4, 6],
                'd_state': [64], 'dropout': [0.1], 'weight_decay': [1e-3],
                'fusion_mode': ['add', 'ablate']
            }
        }
    ]
    main()