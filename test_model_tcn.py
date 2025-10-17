# ~/sdn_neuron_exp/test_model_tcn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
from models.tcn import TCNModel  # 从我们定义的模型文件中导入TCNModel

# --- 1. 配置参数 ---
# ##############################################################################
#  请根据您的实际情况修改这里的路径和文件名
# ##############################################################################
DATA_DIR = '~/Data/neuron_data'  # 指向您的npy文件所在的目录
V0_FILE = 'v0_small.npy'  # 用于测试的v0文件名
V1_FILE = 'v1_small.npy'  # 用于测试的v1文件名

# 数据集和模型参数
SEQ_LEN = 1024  # 每个训练片段的长度
TRAIN_RATIO = 0.7  # 训练集比例
VAL_RATIO = 0.15  # 验证集比例 (测试集比例将自动计算)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 2. 自定义数据集类 ---
class NeuronDataset(Dataset):
    """
    用于加载神经元刺激-响应数据的自定义PyTorch数据集。
    将长时序数据切片成 (上下文状态, 输入序列, 目标序列) 的样本。
    """

    def __init__(self, v0_path, v1_path, seq_len):
        print(f"Loading data from {v0_path} and {v1_path}...")
        self.v0 = np.load(v0_path)
        self.v1 = np.load(v1_path)
        self.seq_len = seq_len

        # 验证数据形状
        assert self.v0.shape[2] + 1 == self.v1.shape[2], "v1 should have one more timestep than v0"
        self.total_timesteps = self.v0.shape[2]

    def __len__(self):
        # 计算可以生成的总片段数
        return self.total_timesteps - self.seq_len + 1

    def __getitem__(self, idx):
        # 根据索引确定片段的起始点
        start_idx = idx
        end_idx = start_idx + self.seq_len

        # 提取v0输入序列
        # [0]是用来去掉数据中 (1, 20, 100000) 的第一个维度
        v0_chunk = torch.from_numpy(self.v0[0, :, start_idx:end_idx]).float()

        # 提取v1上下文状态 (片段开始前一时刻的状态)
        v1_context = torch.from_numpy(self.v1[0, :, start_idx]).float()

        # 提取v1目标序列 (v0[t] 对应 v1[t+1])
        v1_target = torch.from_numpy(self.v1[0, :, start_idx + 1:end_idx + 1]).float()

        return v0_chunk, v1_context, v1_target


# --- 3. 辅助函数：寻找最大batch_size ---
def find_max_batch_size(model, dataset, start_batch_size=8192):
    """
    通过尝试- excepto循环找到不会导致OOM的最大batch_size。
    """
    print("\n--- Finding optimal batch size ---")
    batch_size = start_batch_size
    while batch_size > 0:
        try:
            # 使用一个虚拟的DataLoader和数据进行测试
            loader = DataLoader(dataset, batch_size=batch_size)
            v0_chunk, v1_context, _ = next(iter(loader))

            # 将数据移动到设备并进行一次前向传播
            model.to(DEVICE)
            v0_chunk = v0_chunk.to(DEVICE)
            v1_context = v1_context.to(DEVICE)
            _ = model(v0_chunk, v1_context)

            # 如果成功，则找到了合适的batch_size
            print(f"Successfully ran with batch_size = {batch_size}. Recommending this value.")
            # 清理显存
            del v0_chunk, v1_context
            torch.cuda.empty_cache()
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch_size = {batch_size} is too large (CUDA out of memory). Trying smaller.")
                # 清理显存
                torch.cuda.empty_cache()
                batch_size //= 2
            else:
                raise e  # 如果是其他错误，则抛出

    print("Error: Could not find a suitable batch_size, even with size 1.")
    return 0


# --- 4. 主测试流程 ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 路径处理
    v0_full_path = os.path.expanduser(os.path.join(DATA_DIR, V0_FILE))
    v1_full_path = os.path.expanduser(os.path.join(DATA_DIR, V1_FILE))

    # --- 数据加载和划分 ---
    print("\n--- Data Loading & Splitting ---")
    full_dataset = NeuronDataset(v0_full_path, v1_full_path, SEQ_LEN)
    dataset_size = len(full_dataset)
    print(f"Dataset created with {dataset_size} samples.")

    # 按时间顺序划分索引
    indices = list(range(dataset_size))
    split_train = int(np.floor(TRAIN_RATIO * dataset_size))
    split_val = int(np.floor(VAL_RATIO * dataset_size))

    train_indices = indices[:split_train]
    val_indices = indices[split_train:split_train + split_val]
    test_indices = indices[split_train + split_val:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"Data split chronologically:")
    print(f"  - Training set size: {len(train_dataset)}")
    print(f"  - Validation set size: {len(val_dataset)}")
    print(f"  - Test set size: {len(test_dataset)}")

    # --- 模型实例化 ---
    print("\n--- Model Initialization ---")
    # 使用与模型定义文件中示例相似的参数
    model = TCNModel(
        input_channels=20, output_channels=7,
        num_hidden_channels=[48] * 4,  # 4层, 每层48通道
        input_kernel_size=15, tcn_kernel_size=5, dropout=0.25
    )
    print(f"Model instantiated with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")

    # --- 寻找Batch Size ---
    # 我们使用训练集来寻找batch size，因为它最大
    RECOMMENDED_BATCH_SIZE = find_max_batch_size(model, train_dataset)

    if RECOMMENDED_BATCH_SIZE == 0:
        exit()

    # --- 端到端流程验证 ---
    print("\n--- End-to-End Pipeline Verification ---")
    model.to(DEVICE)
    model.train()  # 设置为训练模式

    # 使用推荐的batch_size创建一个DataLoader
    train_loader = DataLoader(train_dataset, batch_size=RECOMMENDED_BATCH_SIZE, shuffle=True)

    # 定义损失函数和优化器
    # 计算BCE的pos_weight来处理脉冲稀疏性
    spikes = full_dataset.v1[0, 6, :]
    pos_weight_val = (len(spikes) - spikes.sum()) / spikes.sum()
    pos_weight_tensor = torch.tensor([pos_weight_val], device=DEVICE)

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("Fetching one batch of data...")
    v0_batch, v1_context_batch, v1_target_batch = next(iter(train_loader))

    print(f"Data batch shapes:")
    print(f"  - v0_batch: {v0_batch.shape}")
    print(f"  - v1_context_batch: {v1_context_batch.shape}")
    print(f"  - v1_target_batch: {v1_target_batch.shape}")

    # 将数据移动到设备
    v0_batch = v0_batch.to(DEVICE)
    v1_context_batch = v1_context_batch.to(DEVICE)
    v1_target_batch = v1_target_batch.to(DEVICE)
    print("Data moved to device.")

    # 1. 前向传播
    print("Performing forward pass...")
    predictions = model(v0_batch, v1_context_batch)
    print(f"  - Prediction output shape: {predictions.shape}")
    assert predictions.shape == v1_target_batch.shape, "Shape mismatch between prediction and target!"
    print("  - Prediction shape matches target shape. OK.")

    # 2. 计算损失
    print("Calculating loss...")
    pred_voltage = predictions[:, :6, :]
    pred_spike_logits = predictions[:, 6, :].unsqueeze(1)  # 保持通道维度

    target_voltage = v1_target_batch[:, :6, :]
    target_spike = v1_target_batch[:, 6, :].unsqueeze(1)

    loss_mse = criterion_mse(pred_voltage, target_voltage)
    loss_bce = criterion_bce(pred_spike_logits, target_spike)
    total_loss = loss_mse + loss_bce  # 权重暂时都设为1
    print(f"  - MSE Loss: {loss_mse.item():.6f}")
    print(f"  - BCE Loss: {loss_bce.item():.6f}")
    print(f"  - Total Loss: {total_loss.item():.6f}")

    # 3. 反向传播和优化
    print("Performing backward pass and optimizer step...")
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("Backward pass and optimizer step completed successfully.")

    print("\n[SUCCESS] The entire pipeline is verified and works correctly!")