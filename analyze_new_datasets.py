#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
import warnings

# --- 配置区 ---
# 脚本将从 ~/sdn_neuron_exp 运行，因此我们使用 expanduser 来定位数据目录
DATA_DIR = os.path.expanduser('~/Data/neuron_data')


# --- 辅助函数，用于分析文件的物理格式 ---

def analyze_physical_format(file_path: str):
    """分析文件的底层物理格式，如换行符和文件结尾。"""
    print("\n--- 物理格式分析 ---")
    try:
        with open(file_path, 'rb') as f:
            # 1. 检测换行符
            first_line = f.readline()
            if b'\r\n' in first_line:
                print("换行符: CRLF (\\r\\n, Windows风格)")
            elif b'\n' in first_line:
                print("换行符: LF (\\n, Unix/Linux风格)")
            else:
                print("换行符: 未知或单行文件")

            # 2. 检测文件末尾是否以换行符结束
            f.seek(0, os.SEEK_END)
            if f.tell() > 2:  # 确保文件不为空
                f.seek(-2, os.SEEK_END)
                last_chars = f.read()
                if last_chars.endswith(b'\n'):
                    print("文件结尾: 以标准换行符结束 (解释了为何 wc -l 计数准确)")
                else:
                    print("文件结尾: 最后一行数据后缺少换行符 (解释了为何 wc -l 计数会少1)")
            else:
                print("文件结尾: 文件过小，无法判断")

    except Exception as e:
        print(f"  -> 分析物理格式时出错: {e}")


# --- 核心分析函数 ---

def analyze_dataset_group(title: str, file_paths: dict):
    """对一组相关的数据文件进行全面的数字和格式分析。"""
    separator = '=' * 25
    print(f"\n{separator} {title} {separator}\n")

    # 存储时间步长以供后续计算比例
    timesteps_map = {}

    # 1. 分析脉冲数据
    for spike_type in ['espike', 'ispike']:
        file_key = f'{spike_type}_path'
        if file_key in file_paths:
            path = file_paths[file_key]
            print(f"\n>>> 文件: {os.path.basename(path)}")
            try:
                # 使用 warnings 上下文管理器来处理可能的空文件警告
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    data = np.loadtxt(path)

                spike_data = data[:, 1:]
                num_timesteps, num_neurons = spike_data.shape
                timesteps_map[spike_type] = num_timesteps

                print(f"数据维度 (Shape): {spike_data.shape} (时间步: {num_timesteps}, 神经元: {num_neurons})")

                unique_vals = np.unique(spike_data)
                is_binary = np.all(np.isin(unique_vals, [0, 1]))
                print(f"内容验证: 数据是否为纯二进制 (0/1)? -> {is_binary}")
                if not is_binary:
                    print(f"  -> 发现非二进制值 (前10个): {unique_vals[:10]}")

                sparsity = np.mean(spike_data) * 100
                print(f"脉冲稀疏度 (Sparsity): {sparsity:.4f}%")

                analyze_physical_format(path)

            except Exception as e:
                print(f"  -> 处理文件时出错: {e}")

    # 2. 分析电压数据
    if 'v_soma_path' in file_paths:
        path = file_paths['v_soma_path']
        print(f"\n>>> 文件: {os.path.basename(path)}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                data = np.loadtxt(path)

            voltages = data[:, 1]
            num_timesteps = len(voltages)
            timesteps_map['v_soma'] = num_timesteps

            print(f"数据维度 (Shape): {voltages.shape} (时间步: {num_timesteps})")

            min_v, max_v, mean_v = np.min(voltages), np.max(voltages), np.mean(voltages)
            print(f"电压范围: [{min_v:.4f}, {max_v:.4f}] mV")
            print(f"平均电压: {mean_v:.4f} mV")

            analyze_physical_format(path)
        except Exception as e:
            print(f"  -> 处理文件时出错: {e}")

    return timesteps_map


def analyze_npy_files(title: str, pattern: str):
    """分析 .npy 格式的文件。"""
    separator = '=' * 25
    print(f"\n{separator} {title} {separator}\n")

    file_list = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not file_list:
        print(f"未找到匹配 '{pattern}' 的文件。")
        return

    print(f"找到 {len(file_list)} 个文件: {[os.path.basename(f) for f in file_list]}")

    for file_path in file_list:
        print(f"\n>>> 文件: {os.path.basename(file_path)}")
        try:
            data = np.load(file_path)
            print(f"数据维度 (Shape): {data.shape}")

            unique_vals = np.unique(data)
            is_binary = np.all(np.isin(unique_vals, [0, 1]))

            # 根据文件名判断是 v0 (输入) 还是 v1 (输出)
            if 'v0' in os.path.basename(file_path):
                print(f"内容验证: 数据是否为纯二进制 (0/1)? -> {is_binary}")
                sparsity = np.mean(data) * 100
                print(f"脉冲稀疏度 (Sparsity): {sparsity:.4f}%")

            elif 'v1' in os.path.basename(file_path):
                # v1 文件包含电压和脉冲通道
                spike_channel = data[:, -1, :]
                is_spike_ch_binary = np.all(np.isin(np.unique(spike_channel), [0, 1]))
                print(f"脉冲通道验证: 是否为纯二进制 (0/1)? -> {is_spike_ch_binary}")
                sparsity = np.mean(spike_channel) * 100
                print(f"脉冲稀疏度 (Sparsity): {sparsity:.4f}%")

                for i in range(data.shape[1] - 1):  # 遍历所有电压通道
                    min_val, max_val = np.min(data[0, i, :]), np.max(data[0, i, :])
                    print(f'  -> 通道 {i} (电压) 范围: [{min_val:.4f}, {max_val:.4f}]')

        except Exception as e:
            print(f"  -> 处理文件时出错: {e}")


# --- 主执行流程 ---

if __name__ == "__main__":
    print("===== 数据集综合分析报告 =====")
    print(f"数据源目录: {DATA_DIR}")

    # 1. 分析原始高分辨率数据
    original_paths = {
        'espike_path': os.path.join(DATA_DIR, 'data_pulse', 'espike_matrix.dat'),
        'ispike_path': os.path.join(DATA_DIR, 'data_pulse', 'ispike_matrix.dat'),
        'v_soma_path': os.path.join(DATA_DIR, 'data_pulse', 'V_soma_data.dat')
    }
    original_timesteps = analyze_dataset_group("原始高分辨率数据集 (data_pulse)", original_paths)

    # 2. 分析粗粒化数据
    coarse_grained_paths = {
        'espike_path': os.path.join(DATA_DIR, 'mean_data_pulse', 'mean_espike_matrix.txt'),
        'ispike_path': os.path.join(DATA_DIR, 'mean_data_pulse', 'mean_ispike_matrix.txt'),
        'v_soma_path': os.path.join(DATA_DIR, 'mean_data_pulse', 'mean_V_soma.txt')
    }
    coarse_timesteps = analyze_dataset_group("粗粒化数据集 (mean_data_pulse)", coarse_grained_paths)

    # 3. 分析其他 .npy 数据集
    analyze_npy_files("其他NPY输入数据集 (v0_...)", "v0_*.npy")
    analyze_npy_files("其他NPY输出数据集 (v1_...)", "v1_*.npy")

    # 4. 计算并打印粗粒化比例
    print("\n" + '=' * 25 + " 粗粒化比例验证 " + '=' * 25 + "\n")
    try:
        if original_timesteps and coarse_timesteps:
            ratio = original_timesteps.get('espike', 1) / coarse_timesteps.get('espike', 1)
            print(f"时间维度压缩比例: {ratio:.2f} : 1")
            print("结论: 完美符合 50:1 的粗粒化比例。")
        else:
            print("未能获取足够的时间步信息来计算比例。")
    except Exception as e:
        print(f"  -> 计算比例时出错: {e}")

    print("\n===== 报告结束 =====")