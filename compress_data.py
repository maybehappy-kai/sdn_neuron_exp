import numpy as np
import os
import sys


def process_and_save_data(base_path, v0_steps, compression_factor):
    """
    截取、压缩并保存神经元数据。

    Args:
        base_path (str): 数据文件所在的基础路径。
        v0_steps (int): 从 v0.npy 中截取的时间步数。
        compression_factor (int): 压缩倍数。
    """
    V0_FILE = os.path.join(base_path, 'v0.npy')
    V1_FILE = os.path.join(base_path, 'v1.npy')

    if not all(os.path.exists(f) for f in [V0_FILE, V1_FILE]):
        print(f"错误：无法在 '{base_path}' 中找到 v0.npy 或 v1.npy。")
        return

    print(f"\n{'=' * 20}")
    print(f"开始处理: {v0_steps} 步, 压缩倍数: {compression_factor}x")
    print(f"{'=' * 20}")

    # 使用 mmap_mode 高效读取大文件
    d0_mmap = np.load(V0_FILE, mmap_mode='r')
    d1_mmap = np.load(V1_FILE, mmap_mode='r')

    # --- 处理 v0.npy ---
    print(f"--- 正在处理 v0.npy ---")

    # 1. 截取数据
    v0_slice = d0_mmap[0, :, :v0_steps]

    # 2. 改变形状以进行窗口化压缩
    # 新的长度将是原始长度除以压缩倍数
    new_len_v0 = v0_steps // compression_factor
    # Reshape -> (20, new_len, compression_factor)
    reshaped_v0 = v0_slice.reshape(v0_slice.shape[0], new_len_v0, compression_factor)

    # 3. 压缩：如果窗口内有任何 30，则结果为 30，否则为 -70
    # np.any(reshaped_v0 == 30, axis=-1) 会返回一个布尔数组
    spikes_present = np.any(reshaped_v0 == 30, axis=-1)
    compressed_v0 = np.where(spikes_present, 30, -70).astype(np.int8)

    # 4. 恢复 (1, 20, new_len) 的形状
    compressed_v0 = compressed_v0[np.newaxis, :, :]

    # 5. 保存文件
    output_fn_v0 = os.path.join(base_path, f'v0_small_{compression_factor}x.npy')
    np.save(output_fn_v0, compressed_v0)
    print(f"已保存: {output_fn_v0}，形状: {compressed_v0.shape}")

    # --- 处理 v1.npy ---
    print(f"\n--- 正在处理 v1.npy ---")

    # 1. 计算 v1 的截取长度，确保压缩后比 v0 多一步
    v1_steps = (new_len_v0 + 1) * compression_factor
    v1_slice = d1_mmap[0, :, :v1_steps]

    # 2. 改变形状以进行窗口化压缩
    new_len_v1 = v1_steps // compression_factor
    # Reshape -> (6, new_len, compression_factor)
    reshaped_v1 = v1_slice.reshape(v1_slice.shape[0], new_len_v1, compression_factor)

    # 3. 压缩：计算窗口内的平均值
    compressed_v1 = reshaped_v1.mean(axis=-1).astype(np.float32)

    # 4. 恢复 (1, 6, new_len) 的形状
    compressed_v1 = compressed_v1[np.newaxis, :, :]

    # 5. 保存文件
    output_fn_v1 = os.path.join(base_path, f'v1_small_{compression_factor}x.npy')
    np.save(output_fn_v1, compressed_v1)
    print(f"已保存: {output_fn_v1}，形状: {compressed_v1.shape}")


if __name__ == '__main__':
    # 定义数据文件所在目录
    DATA_PATH = '/mnt/data/yukaihuang/neuron_data'

    # 定义处理配置: (v0截取步数, 压缩倍数)
    CONFIGS = [
        (300000, 3),  # 截取30万步，3倍压缩
        (3000000, 30)  # 截取300万步，30倍压缩
    ]

    # 检查路径是否存在
    if not os.path.isdir(DATA_PATH):
        print(f"错误: 目录不存在 - '{DATA_PATH}'")
        sys.exit(1)

    # 循环执行所有压缩配置
    for steps, factor in CONFIGS:
        process_and_save_data(DATA_PATH, steps, factor)

    # --- 处理不压缩的特殊情况 (1x) ---
    print(f"\n{'=' * 20}")
    print(f"开始处理: 不压缩，截取特定步数")
    print(f"{'=' * 20}")

    V0_FILE_SPECIAL = os.path.join(DATA_PATH, 'v0.npy')
    V1_FILE_SPECIAL = os.path.join(DATA_PATH, 'v1.npy')

    # 确保文件存在
    if all(os.path.exists(f) for f in [V0_FILE_SPECIAL, V1_FILE_SPECIAL]):
        d0_mmap_special = np.load(V0_FILE_SPECIAL, mmap_mode='r')
        d1_mmap_special = np.load(V1_FILE_SPECIAL, mmap_mode='r')

        # --- 处理 v0.npy (截取 100,000 步) ---
        print(f"--- 正在处理 v0.npy (不压缩) ---")
        v0_steps_special = 100000
        # **修正**: 先正确截取2D数据，再恢复3D形状
        v0_special_slice = d0_mmap_special[0, :, :v0_steps_special]
        v0_special_slice = v0_special_slice[np.newaxis, :, :]

        output_fn_v0_special = os.path.join(DATA_PATH, 'v0_small.npy')
        np.save(output_fn_v0_special, v0_special_slice.astype(np.int8))
        print(f"已保存: {output_fn_v0_special}，形状: {v0_special_slice.shape}")

        # --- 处理 v1.npy (截取 100,001 步) ---
        print(f"\n--- 正在处理 v1.npy (不压缩) ---")
        v1_steps_special = 100001
        # **修正**: 先正确截取2D数据，再恢复3D形状
        v1_special_slice = d1_mmap_special[0, :, :v1_steps_special]
        v1_special_slice = v1_special_slice[np.newaxis, :, :]

        output_fn_v1_special = os.path.join(DATA_PATH, 'v1_small.npy')
        np.save(output_fn_v1_special, v1_special_slice.astype(np.float32))
        print(f"已保存: {output_fn_v1_special}，形状: {v1_special_slice.shape}")
    else:
        print(f"错误：在处理不压缩情况时，无法在 '{DATA_PATH}' 中找到 v0.npy 或 v1.npy。")

    print(f"\n{'=' * 20}")
    print("所有任务已完成！")
    print(f"{'=' * 20}")