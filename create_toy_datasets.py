import numpy as np
import os

# =============================================================================
# --- 全局参数定义 ---
# =============================================================================
DATA_PATH = '~/Data/neuron_data'
TIMESTEPS = 100000
INPUT_CHANNELS = 20
OUTPUT_CHANNELS = 7
# 回归到 1% 的输入稀疏度
SPIKE_PROBABILITY = 0.01


# =============================================================================
# --- 数据集 1: 简单的线性映射 + Sigmoid (初始版本) ---
# =============================================================================
def create_simple_dataset(base_path):
    """
    创建一个简单的、通过线性变换和非线性激活函数生成的玩具数据集。
    """
    print("--- 正在创建数据集 1: 简单映射 (初始版本) ---")

    # 1. 创建输入数据 v0
    v0_data = np.full((1, INPUT_CHANNELS, TIMESTEPS), -70, dtype=np.int8)
    np.random.seed(42)
    spike_mask = np.random.rand(1, INPUT_CHANNELS, TIMESTEPS) < SPIKE_PROBABILITY
    v0_data[spike_mask] = 30
    v0_binary = (v0_data == 30).astype(np.float32)

    # 2. 定义转换矩阵
    transform_matrix = np.random.randn(INPUT_CHANNELS, OUTPUT_CHANNELS) * 0.5
    v1_transformed = np.dot(v0_binary[0].T, transform_matrix).T

    # 3. 创建 v1 主体 (无平滑)
    v1_main = np.zeros((OUTPUT_CHANNELS, TIMESTEPS), dtype=np.float32)
    v1_main[:6, :] = 1 / (1 + np.exp(-v1_transformed[:6, :]))
    v1_main[6, :] = (v1_transformed[6, :] > 0.74).astype(np.float32)

    # 4. 组合并保存
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
# --- 数据集 2: Leaky Integrate-and-Fire (LIF) 神经元 (初始版本) ---
# =============================================================================
def create_lif_dataset(base_path):
    """
    创建一个基于LIF神经元动力学的玩具数据集 (初始版本)。
    """
    print("--- 正在创建数据集 2: LIF 神经元 (初始版本) ---")

    # 1. 创建输入数据 v0
    v0_data = np.full((1, INPUT_CHANNELS, TIMESTEPS), -70, dtype=np.int8)
    np.random.seed(1337)
    spike_mask = np.random.rand(1, INPUT_CHANNELS, TIMESTEPS) < SPIKE_PROBABILITY
    v0_data[spike_mask] = 30
    v0_binary = (v0_data == 30).astype(np.float32)

    # 2. 计算输入电流，使用能产生较多脉冲的原始参数
    input_current = np.sum(v0_binary[0], axis=0) * 0.25

    # 3. 定义LIF参数 (回归到初始值)
    tau_m = 10.0  # 较小的时间常数，响应更快
    v_thresh = 1.0  # 阈值电位
    v_reset = 0.0  # 硬复位电位
    dt = 1.0  # 时间步长 (ms)
    alpha = np.exp(-dt / tau_m)

    # 4. 模拟神经元动态
    v_mem = np.zeros(TIMESTEPS, dtype=np.float32)
    spikes = np.zeros(TIMESTEPS, dtype=np.float32)
    v = 0.0
    for t in range(TIMESTEPS):
        v = v * alpha + input_current[t]
        if v >= v_thresh:
            spikes[t] = 1.0
            v = v_reset
        v_mem[t] = v

    # 5. 创建 v1
    v1_main = np.zeros((OUTPUT_CHANNELS, TIMESTEPS), dtype=np.float32)
    normalized_v_mem = (v_mem - v_mem.min()) / (v_mem.max() - v_mem.min() + 1e-8)
    for i in range(6):
        v1_main[i, :] = normalized_v_mem
    v1_main[6, :] = spikes

    # 6. 组合并保存
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