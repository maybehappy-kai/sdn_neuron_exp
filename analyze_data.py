import numpy as np
import os
import argparse
import glob


def analyze_v0_data(file_path: str):
    """
    分析 v0 (输入刺激) 数据文件。

    打印文件的形状、数据稀疏度、验证其是否为二进制 (0/1)，并展示数据样本。
    """
    try:
        data = np.load(file_path)

        # --- 验证 ---
        unique_vals = np.unique(data)
        is_binary = np.all(np.isin(unique_vals, [0, 1]))

        # --- 稀疏度计算 ---
        sparsity = np.mean(data) * 100

        # --- 打印报告 ---
        separator = '=' * 20
        print(f'\n{separator} {os.path.basename(file_path)} {separator}')
        print(f'Shape: {data.shape}')
        print(f'Stimulus Sparsity: {sparsity:.4f}%')
        print(f'Verification: Contains only 0s and 1s? -> {is_binary}')

        # --- 打印样本 ---
        mid = data.shape[2] // 2
        np.set_printoptions(precision=1, suppress=True)
        print('\n--- Data Sample ---')
        print('First 5 timesteps:\n', data[..., :5])
        print('\nMiddle 5 timesteps:\n', data[..., mid - 2:mid + 3])
        print('\nLast 5 timesteps:\n', data[..., -5:])

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def analyze_v1_data(file_path: str):
    """
    分析 v1 (神经元响应) 数据文件。

    打印文件的形状、各通道的电压范围、脉冲通道的二进制验证、脉冲稀疏度以及数据样本。
    """
    try:
        data = np.load(file_path)

        # --- 稀疏度计算 ---
        spike_channel = data[:, -1, :]
        sparsity = np.mean(spike_channel) * 100

        # --- 打印报告 ---
        separator = '=' * 20
        print(f'\n{separator} {os.path.basename(file_path)} {separator}')
        print(f'Shape: {data.shape}')
        print(f'Spike Sparsity: {sparsity:.4f}%')

        # --- 验证 ---
        print('\n--- Verifications ---')
        for i in range(data.shape[1] - 1):  # 遍历所有电压通道
            min_val = np.min(data[0, i, :])
            max_val = np.max(data[0, i, :])
            print(f'Ch {i} (Voltage):  Range [{min_val:.4f}, {max_val:.4f}]')

        last_ch_unique = np.unique(data[0, -1, :])
        is_last_ch_binary = np.all(np.isin(last_ch_unique, [0, 1]))
        print(f'Ch {data.shape[1] - 1} (Spikes):   Contains only 0s and 1s? -> {is_last_ch_binary}')

        # --- 打印样本 ---
        mid = data.shape[2] // 2
        np.set_printoptions(precision=3, suppress=True)
        print('\n--- Data Sample (all channels) ---')
        print('First 5 timesteps:\n', data[..., :5])
        print('\nMiddle 5 timesteps:\n', data[..., mid - 2:mid + 3])
        print('\nLast 5 timesteps:\n', data[..., -5:])

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main(data_dir: str):
    """
    主函数，查找并分析指定目录下的所有 'small' 或 'lif' 或 'simple' 数据文件。
    """
    print(f"Analyzing data in: {data_dir}")

    # 使用 glob 查找所有需要分析的文件
    patterns = ['*small*.npy', '*lif*.npy', '*simple*.npy']
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(os.path.join(data_dir, pattern)))

    # 去重并排序
    file_list = sorted(list(set(file_list)))

    if not file_list:
        print("No matching .npy files found in the specified directory.")
        return

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        if 'v0' in file_name:
            analyze_v0_data(file_path)
        elif 'v1' in file_name:
            analyze_v1_data(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze neuron simulation data files.")
    parser.add_argument('--data_dir', type=str, default='~/Data/neuron_data',
                        help='Directory containing the .npy data files.')
    args = parser.parse_args()

    expanded_data_dir = os.path.expanduser(args.data_dir)
    main(expanded_data_dir)