import numpy as np
import os
from scipy.signal import convolve

# =============================================================================
# --- 全局参数定义 ---
# =============================================================================
DATA_PATH = '~/Data/neuron_data'
TIMESTEPS = 100000
INPUT_CHANNELS = 20
OUTPUT_CHANNELS = 7
SPIKE_PROBABILITY = 0.01


# =============================================================================
# --- 数据集 1: 平滑的线性映射 ---
# =============================================================================
def create_simple_dataset(base_path):
    """
    创建一个平滑的、通过线性变换生成的玩具数据集。
    """
    print("--- 正在创建数据集 1: 简单平滑映射 ---")

    # *** 更正部分 1: 直接生成只包含 0 和 1 的 v0 数据 ***
    v0_data = np.zeros((1, INPUT_CHANNELS, TIMESTEPS), dtype=np.int8)
    np.random.seed(42)
    spike_mask = np.random.rand(1, INPUT_CHANNELS, TIMESTEPS) < SPIKE_PROBABILITY
    v0_data[spike_mask] = 1

    # 定义转换矩阵
    transform_matrix = np.random.randn(INPUT_CHANNELS, OUTPUT_CHANNELS) * 0.5
    # 直接使用 v0_data[0] 进行计算
    v1_transformed = np.dot(v0_data[0].T.astype(np.float32), transform_matrix).T

    # 对变换后的信号进行平滑处理
    smoothing_window = np.ones(100) / 100.0
    v1_smoothed = np.zeros_like(v1_transformed)
    for i in range(OUTPUT_CHANNELS):
        v1_smoothed[i, :] = convolve(v1_transformed[i, :], smoothing_window, mode='same')

    # 基于平滑后的信号生成v1
    v1_main = np.zeros((OUTPUT_CHANNELS, TIMESTEPS), dtype=np.float32)
    v1_main[:6, :] = 1 / (1 + np.exp(-v1_smoothed[:6, :]))
    v1_main[6, :] = (v1_smoothed[6, :] > 0.025).astype(np.float32)

    # 组合并保存
    initial_state = np.zeros((OUTPUT_CHANNELS, 1), dtype=np.float32)
    v1_data = np.concatenate([initial_state, v1_main], axis=1)[np.newaxis, :, :]

    v0_save_path = os.path.join(base_path, 'v0_simple.npy')
    v1_save_path = os.path.join(base_path, 'v1_simple.npy')
    np.save(v0_save_path, v0_data)
    np.save(v1_save_path, v1_data)

    print(f"已保存 '{v0_save_path}', 形状: {v0_data.shape}")
    print(f"已保存 '{v1_save_path}', 形状: {v1_data.shape}")
    print(f"简单数据集创建完毕！脉冲数: {np.sum(v1_main[6, :]):.0f}\n")


# =============================================================================
# --- 数据集 2: 平滑且稀疏的LIF神经元 ---
# =============================================================================
def create_lif_dataset(base_path):
    """
    创建一个膜电位平滑、脉冲发放稀疏的LIF数据集。
    """
    print("--- 正在创建数据集 2: 平滑稀疏LIF神经元 ---")

    # *** 更正部分 2: 直接生成只包含 0 和 1 的 v0 数据 ***
    v0_data = np.zeros((1, INPUT_CHANNELS, TIMESTEPS), dtype=np.int8)
    np.random.seed(1337)
    spike_mask = np.random.rand(1, INPUT_CHANNELS, TIMESTEPS) < SPIKE_PROBABILITY
    v0_data[spike_mask] = 1

    # 使用固定的、较小的电流缩放系数，并增大时间常数
    # 直接使用 v0_data[0] 进行计算
    input_current = np.sum(v0_data[0], axis=0) * 0.13
    tau_m = 25.0
    v_thresh, v_reset, dt = 1.0, 0.0, 1.0
    alpha = np.exp(-dt / tau_m)

    # 模拟神经元动态
    v_mem = np.zeros(TIMESTEPS, dtype=np.float32)
    spikes = np.zeros(TIMESTEPS, dtype=np.float32)
    v = 0.0
    for t in range(TIMESTEPS):
        v = v * alpha + input_current[t]
        if v >= v_thresh:
            spikes[t] = 1.0
            v = v_reset
        v_mem[t] = v

    # 创建 v1
    v1_main = np.zeros((OUTPUT_CHANNELS, TIMESTEPS), dtype=np.float32)
    normalized_v_mem = (v_mem - v_mem.min()) / (v_mem.max() - v_mem.min() + 1e-8)
    for i in range(6):
        v1_main[i, :] = normalized_v_mem
    v1_main[6, :] = spikes

    # 组合并保存
    initial_state = np.zeros((OUTPUT_CHANNELS, 1), dtype=np.float32)
    v1_data = np.concatenate([initial_state, v1_main], axis=1)[np.newaxis, :, :]

    v0_save_path = os.path.join(base_path, 'v0_lif.npy')
    v1_save_path = os.path.join(base_path, 'v1_lif.npy')
    np.save(v0_save_path, v0_data)
    np.save(v1_save_path, v1_data)

    print(f"已保存 '{v0_save_path}', 形状: {v0_data.shape}")
    print(f"已保存 '{v1_save_path}', 形状: {v1_data.shape}")
    print(f"LIF数据集创建完毕！脉冲数: {np.sum(spikes):.0f}\n")


if __name__ == '__main__':
    expanded_path = os.path.expanduser(DATA_PATH)
    os.makedirs(expanded_path, exist_ok=True)
    print(f"数据集将被保存在: {expanded_path}")

    create_simple_dataset(expanded_path)
    create_lif_dataset(expanded_path)

    print("所有玩具数据集已成功生成。")