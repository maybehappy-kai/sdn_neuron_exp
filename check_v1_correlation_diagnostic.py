import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


# --- Focal Loss 定义 (不变) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha;
        self.gamma = gamma;
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss);
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss


# --- 模型定义: 两个版本 ---
class SpikeDetectorCNN(nn.Module):
    """
    CNN模型的基类，包含共享的架构。
    """

    def __init__(self, window_size: int):
        super(SpikeDetectorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        final_seq_len = window_size // 4
        fc1_input_features = 16 * final_seq_len
        self.fc1 = nn.Linear(fc1_input_features, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def _shared_forward(self, x):
        x = F.relu(self.conv1(x));
        x = self.pool(x)
        x = F.relu(self.conv2(x));
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x);
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class GatedSpikeCNN(SpikeDetectorCNN):
    """带硬编码门控的版本"""

    def forward(self, x):
        # x 形状: (batch_size, 1, window_size)
        # 规定最后一个特征是当前时刻的电压
        current_voltage_channel = x[:, 0, -1].unsqueeze(1)

        preliminary_logit = self._shared_forward(x)
        mask = (current_voltage_channel < 1.0).float()
        final_logit = preliminary_logit - (mask * 1e9)
        return final_logit


class StandardSpikeCNN(SpikeDetectorCNN):
    """不带硬编码门控的标准版本 (对照组)"""

    def forward(self, x):
        return self._shared_forward(x)


def create_windowed_dataset(voltage, spikes, window_size):
    # (与之前相同)
    X, y = [], []
    for i in range(window_size - 1, len(voltage)):
        X.append(voltage[i - window_size + 1: i + 1])
        y.append(spikes[i])
    return np.array(X, dtype=np.float32)[:, np.newaxis, :], np.array(y, dtype=np.float32).reshape(-1, 1)


def analyze_voltage_spike_correlation_diagnostic(dataset_name: str, epochs: int, alpha: float, window_size: int,
                                                 use_gate: bool):
    # --- 数据加载与准备 (与之前相同) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_data_dir = os.path.expanduser('~/Data/neuron_data')
    v1_file_path = os.path.join(base_data_dir, f'v1_{dataset_name}.npy')
    # ... (省略大部分与之前版本相同的数据准备代码) ...
    print(f"===== 正在分析数据集: v1_{dataset_name}.npy (诊断模式) =====")
    v1_data = np.load(v1_file_path)
    voltage_raw = v1_data[0, 0, 1:];
    spikes_raw = v1_data[0, 1, 1:]
    if window_size % 4 != 0: window_size = ((window_size // 4) + 1) * 4; print(f"警告: 窗口大小已调整为 {window_size}")
    X, y = create_windowed_dataset(voltage_raw, spikes_raw, window_size)
    sample_size = min(len(y), 200000)
    X_sample, y_sample = X[:sample_size], y[:sample_size]
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42,
                                                        stratify=y_sample)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_numpy = y_test.flatten()

    # --- 模型选择与训练 ---
    if use_gate:
        model = GatedSpikeCNN(window_size=window_size).to(device)
        model_type = "带门控 (Gated)"
    else:
        model = StandardSpikeCNN(window_size=window_size).to(device)
        model_type = "标准 (Standard)"

    print(f"--- 开始训练CNN模型: {model_type} ---")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = FocalLoss(alpha=alpha, gamma=2.0).to(device)
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad();
            outputs = model(batch_X);
            loss = criterion(outputs, batch_y);
            loss.backward();
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} 完成。")
    print("模型训练完成。\n")

    # --- 全局评估 (与之前相同) ---
    print(f"--- 1. 在【全部】测试集上的全局性能评估 ---")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        y_pred_scores = torch.sigmoid(test_outputs).cpu().numpy().flatten()
    fpr, tpr, thresholds = roc_curve(y_test_numpy, y_pred_scores)
    roc_auc = auc(fpr, tpr);
    optimal_idx = np.argmax(tpr - fpr);
    optimal_threshold = thresholds[optimal_idx]
    y_pred_optimal = (y_pred_scores >= optimal_threshold).astype(int)
    print(f"全局 AUC: {roc_auc:.4f}, 最佳阈值: {optimal_threshold:.4f}")
    print(classification_report(y_test_numpy, y_pred_optimal, target_names=['No Spike', 'Spike'], zero_division=0))

    # --- 关键新增: 在电压=1.0的子集上进行评估 ---
    print(f"\n--- 2. 在【电压=1.0】的子集上的专项性能评估 ---")
    # 找到测试集中所有当前电压为1.0的样本
    # X_test 形状: (N, 1, window_size)
    subset_indices = np.where(X_test[:, 0, -1] == 1.0)[0]

    if len(subset_indices) > 0:
        # 提取子集
        y_test_subset = y_test_numpy[subset_indices]
        y_pred_subset = y_pred_optimal[subset_indices]

        # 计算子集上的准确率
        subset_accuracy = accuracy_score(y_test_subset, y_pred_subset)

        print(f"在 {len(subset_indices)} 个电压为1.0的样本中:")
        print(f"模型的分类准确率: {subset_accuracy:.4f}")

        print("\n子集上的混淆矩阵:")
        print(confusion_matrix(y_test_subset, y_pred_subset))

        print("\n子集上的详细报告:")
        print(classification_report(y_test_subset, y_pred_subset, target_names=['No Spike', 'Spike'], zero_division=0))
    else:
        print("测试集中没有找到电压为1.0的样本，无法进行子集评估。")
    print("=" * 40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnostic analysis of CNN spike detectors.")
    parser.add_argument('dataset', type=str, choices=['new', 'raw'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--window_size', type=int, default=20)
    # --- 新增: 控制是否使用门控的开关 ---
    parser.add_argument('--no_gate', action='store_true', help="使用标准的CNN模型，不带硬编码的启发式门控。")
    args = parser.parse_args()

    analyze_voltage_spike_correlation_diagnostic(args.dataset, args.epochs, args.alpha, args.window_size,
                                                 use_gate=not args.no_gate)