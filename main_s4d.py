import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time

from torch.utils.data import Dataset


# --- 数据集类 (与TCN版本完全一致) ---
class NeuronTimeSeriesDataset(Dataset):
    def __init__(self, v0_data, v1_data, sequence_length, indices):
        """
        修改后的数据集类，只处理指定索引范围内的样本，并生成脉冲目标。

        :param v0_data: 完整的v0数据
        :param v1_data: 完整的v1数据
        :param sequence_length: 序列窗口长度
        :param indices: 这个数据集实例应该使用的索引列表
        """
        super().__init__()
        self.v0_data = v0_data
        self.v1_data = v1_data
        self.sequence_length = sequence_length
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        v0_seq = self.v0_data[actual_idx: actual_idx + self.sequence_length]
        v1_initial = self.v1_data[actual_idx]
        v1_target = self.v1_data[actual_idx + 1: actual_idx + 1 + self.sequence_length]

        # 脉冲发放的阈值，高于此电压的时刻被认为是脉冲
        spike_threshold = 0
        # 我们只关心 soma 的脉冲，即 v1 的第6个通道 (索引为5)
        spike_target = (v1_target[:, 5] > spike_threshold).float()

        # 返回四个值
        return v0_seq, v1_initial, v1_target, spike_target


# --- S4D核心层 (保持不变) ---
class S4D(nn.Module):
    """一个简化的、自包含的S4D层实现"""

    def __init__(self, d_model, N=64):
        super().__init__()
        self.d_model = d_model
        self.N = N

        A_re = -0.5 * torch.ones(d_model, N)
        A_im = torch.arange(N, dtype=torch.float32).view(1, N).expand(d_model, -1)
        self.A = nn.Parameter(torch.complex(A_re, A_im))
        self.C = nn.Parameter(torch.randn(d_model, N, dtype=torch.cfloat))
        self.log_dt = nn.Parameter(torch.rand(d_model) * (np.log(0.1) - np.log(0.001)) + np.log(0.001))
        self.B = torch.ones(self.d_model, self.N, dtype=torch.cfloat)

    def forward(self, u):
        """ u: (batch, L, d_model) """
        L = u.size(1)
        B = self.B.to(u.device)
        dt = torch.exp(self.log_dt)
        delta = torch.einsum('d,dn->dn', dt, self.A)
        A_bar = torch.exp(delta)
        B_bar = torch.einsum('d,dn->dn', dt, B) * (torch.exp(delta) - 1 + 1e-8) / (delta + 1e-8)

        v = torch.arange(L, device=u.device)
        A_bar_v = A_bar.unsqueeze(-1) ** v
        K = torch.einsum('dn,dnl,dn->dl', self.C, A_bar_v, B_bar).real

        k_f = torch.fft.rfft(K, n=2 * L)
        u_f = torch.fft.rfft(u.transpose(1, 2), n=2 * L)
        y_f = k_f * u_f
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]

        return y.transpose(1, 2)


# =====================================================================================
#  将您训练脚本中的 SurrogateS4D 类替换为下面的版本
# =====================================================================================

class SurrogateS4D(nn.Module):
    def __init__(self, d_model, N, v0_features=20, v1_features=6):
        super().__init__()
        self.in_proj = nn.Linear(v0_features, d_model)
        self.h0_proj = nn.Linear(v1_features, d_model)

        self.s4d = S4D(d_model=d_model, N=N)
        self.activation = nn.GELU()

        self.voltage_out = nn.Linear(d_model, v1_features)
        self.spike_out = nn.Linear(d_model, 1)

        self._calculate_params()

    def forward(self, v0_seq, v1_initial):
        """
        修正后的前向传播。
        v0_seq: (Batch, Seq_Len, v0_features)
        v1_initial: (Batch, v1_features)
        """
        # 将输入的脉冲刺激 one-hot 编码 (0或1)
        v0_processed = torch.zeros_like(v0_seq)
        v0_processed[v0_seq == 30] = 1.0

        # --- 这是关键的修改部分 ---

        # 1. 先将整个输入序列 v0 投影到 S4D 的隐藏维度 d_model
        x = self.in_proj(v0_processed)

        # 2. 将初始电压 v1_initial 投影到 d_model，并【仅仅】加到序列的第一个时间步 (t=0) 上。
        #    这正确地将 v1_initial 作为了序列的初始条件。
        #    之前的错误做法是 .unsqueeze(1) 再相加，导致 v1_initial 被加到了每一个时间步上。
        x[:, 0, :] = x[:, 0, :] + self.h0_proj(v1_initial)

        # --- 修改结束 ---

        # 后续的 S4D 处理和输出头保持不变
        x = self.s4d(x)
        x = self.activation(x)

        voltage_prediction = self.voltage_out(x)
        spike_prediction = self.spike_out(x)

        # 为了与损失函数期望的 (B, C, L) 格式对齐，需要转置
        return voltage_prediction.transpose(1, 2), spike_prediction.transpose(1, 2)

    def _calculate_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SurrogateS4D 初始化完毕，总参数量: {total_params}")


# =====================================================================================
#  脚本中的其余部分 (Dataset, S4D, get_dataloaders, train, main等) 保持不变
# =====================================================================================

# --- 数据加载函数 (修正为块状切分, 避免数据泄露) ---
def get_dataloaders(args):
    """
    加载神经元数据集，并按【时间顺序】切分训练/验证/测试集，避免数据泄露。
    """
    v0_path = os.path.join(args.data_dir, 'v0_small.npy')
    v1_path = os.path.join(args.data_dir, 'v1_small.npy')

    if not (os.path.exists(v0_path) and os.path.exists(v1_path)):
        raise FileNotFoundError(f"错误：在 {args.data_dir} 中找不到 v0(_small).npy 或 v1(_small).npy。")

    print(f"从 {v0_path} 和 {v1_path} 加载数据...")
    v0_full = torch.from_numpy(np.load(v0_path)).float().squeeze(0).transpose(0, 1)
    v1_full = torch.from_numpy(np.load(v1_path)).float().squeeze(0).transpose(0, 1)

    num_possible_samples = v0_full.shape[0] - args.sequence_length
    all_indices = np.arange(num_possible_samples)

    # --- 关键修正：不再随机打乱时序索引 ---
    # np.random.shuffle(all_indices) # <- 已删除此行以修复数据泄露

    # 按 6:1:3 的比例，【按顺序】切分索引块
    train_end_idx = int(0.6 * num_possible_samples)
    val_end_idx = train_end_idx + int(0.1 * num_possible_samples)

    train_indices = all_indices[:train_end_idx]
    val_indices = all_indices[train_end_idx:val_end_idx]
    test_indices = all_indices[val_end_idx:]

    train_dataset = NeuronTimeSeriesDataset(v0_full, v1_full, args.sequence_length, train_indices)
    val_dataset = NeuronTimeSeriesDataset(v0_full, v1_full, args.sequence_length, val_indices)
    test_dataset = NeuronTimeSeriesDataset(v0_full, v1_full, args.sequence_length, test_indices)

    print(f"数据集划分 (块状切分): 训练样本数={len(train_dataset)}, 验证样本数={len(val_dataset)}, 测试样本数={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


# --- 训练函数 (与TCN版本完全一致) ---
def train(args, model, train_loader, val_loader):
    """训练模型，并保存最佳检查点"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    model.to(device)

    criterion_voltage = nn.MSELoss()
    criterion_spike = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    # 修改检查点文件名以区分模型
    checkpoint_path = os.path.join(args.model_dir, "best_s4d_model.pt")
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"开始训练 {args.epochs} 个 epochs...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for v0_seq, v1_initial, v1_target, spike_target in train_loader:
            # S4D模型期望输入是 (B, L, C)，所以不需要像TCN那样转置v0_seq
            v0_seq, v1_initial = v0_seq.to(device), v1_initial.to(device)
            # 目标需要转置以匹配模型输出 (B, L, C) -> (B, C, L)
            v1_target = v1_target.transpose(1, 2).to(device)
            spike_target = spike_target.unsqueeze(1).to(device)

            optimizer.zero_grad()

            voltage_pred, spike_pred_logits = model(v0_seq, v1_initial)

            voltage_clamp_threshold = -55.0
            v1_target_clamped = torch.clamp(v1_target, max=voltage_clamp_threshold)

            loss_voltage = criterion_voltage(voltage_pred, v1_target_clamped)
            loss_spike = criterion_spike(spike_pred_logits, spike_target)

            loss = 1.0 * loss_spike + 0.005 * loss_voltage

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        running_val_loss, running_val_loss_v, running_val_loss_s = 0.0, 0.0, 0.0
        with torch.no_grad():
            for v0_seq, v1_initial, v1_target, spike_target in val_loader:
                v0_seq, v1_initial = v0_seq.to(device), v1_initial.to(device)
                v1_target = v1_target.transpose(1, 2).to(device)
                spike_target = spike_target.unsqueeze(1).to(device)

                voltage_pred, spike_pred_logits = model(v0_seq, v1_initial)

                voltage_clamp_threshold = -55.0
                v1_target_clamped = torch.clamp(v1_target, max=voltage_clamp_threshold)

                loss_voltage = criterion_voltage(voltage_pred, v1_target_clamped)
                loss_spike = criterion_spike(spike_pred_logits, spike_target)

                loss = 1.0 * loss_spike + 0.005 * loss_voltage
                running_val_loss += loss.item()
                running_val_loss_v += loss_voltage.item()
                running_val_loss_s += loss_spike.item()

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_loss_v = running_val_loss_v / len(val_loader)
        avg_val_loss_s = running_val_loss_s / len(val_loader)

        print(
            f"Epoch [{epoch + 1:03d}/{args.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (v_loss: {avg_val_loss_v:.6f}, s_loss: {avg_val_loss_s:.6f})")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> 验证损失创新低，模型已保存至: {checkpoint_path}")

        scheduler.step(avg_val_loss)

# --- 测试与可视化函数 (与TCN版本风格完全一致) ---
def test_and_visualize(args, test_loader):
    """在测试集上评估并将可视化结果保存为图片"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(args.model_dir, "best_s4d_model.pt")
    vis_dir = os.path.join(args.model_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print("错误：找不到模型检查点。请先进行训练。")
        return

    # 实例化与训练时完全相同的S4D模型结构
    model = SurrogateS4D(d_model=args.s4d_d_model, N=args.s4d_N)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    criterion_voltage = nn.MSELoss()
    criterion_spike = nn.BCEWithLogitsLoss()
    total_v_loss, total_s_loss, total_loss = 0.0, 0.0, 0.0

    print("\n在测试集上评估最佳模型...")
    with torch.no_grad():
        for v0_seq, v1_initial, v1_target, spike_target in test_loader:
            v0_seq, v1_initial = v0_seq.to(device), v1_initial.to(device)
            v1_target = v1_target.transpose(1, 2).to(device)
            spike_target = spike_target.unsqueeze(1).to(device)

            voltage_pred, spike_pred_logits = model(v0_seq, v1_initial)

            voltage_clamp_threshold = -55.0
            v1_target_clamped = torch.clamp(v1_target, max=voltage_clamp_threshold)

            loss_v = criterion_voltage(voltage_pred, v1_target_clamped)
            loss_s = criterion_spike(spike_pred_logits, spike_target)

            total_v_loss += loss_v.item()
            total_s_loss += loss_s.item()
            total_loss += (1.0 * loss_s + 0.005 * loss_v).item()

    avg_test_loss = total_loss / len(test_loader)
    avg_test_v_loss = total_v_loss / len(test_loader)
    avg_test_s_loss = total_s_loss / len(test_loader)
    print(f"======================================================")
    print(f"S4D模型 - 最终测试集加权损失: {avg_test_loss:.6f}")
    print(f"        - 电压MSE损失 (钳制后): {avg_test_v_loss:.6f}")
    print(f"        - 脉冲BCE损失: {avg_test_s_loss:.6f}")
    print(f"========================================================")

    print(f"\n在测试集上进行可视化，图片将保存至: {vis_dir}")
    v0_seq_b, v1_initial_b, v1_target_b, spike_target_b = next(iter(test_loader))

    with torch.no_grad():
        voltage_pred_b, spike_pred_logits_b = model(v0_seq_b.to(device), v1_initial_b.to(device))

    v0_plot = (v0_seq_b.numpy() > 0).astype(float)
    # v1_target_b 是 (B, L, C)，用于绘图
    v1_target_plot = v1_target_b.numpy()
    # voltage_pred_b 是 (B, C, L)，需要转置为 (B, L, C)
    predictions_plot = voltage_pred_b.cpu().transpose(1, 2).numpy()
    # spike_pred_logits_b 是 (B, 1, L)，需要激活并移除通道维度
    spike_preds_plot = torch.sigmoid(spike_pred_logits_b).cpu().squeeze(1).numpy()

    for i in range(min(args.num_visuals, v0_plot.shape[0])):
        soma_channel_idx = 5

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})
        fig.suptitle(f'S4D Prediction vs. Ground Truth - Sample {i+1}', fontsize=16)

        ax1.set_title("Input Spike Stimuli (v0)")
        ax1.imshow(v0_plot[i].T, aspect='auto', interpolation='nearest', cmap='binary')
        ax1.set_ylabel("Synaptic Channels")
        ax1.set_yticks([])

        ax2.set_title(f"Somatic Potential (Channel {soma_channel_idx})")
        ax2.plot(v1_target_plot[i, :, soma_channel_idx], label='Ground Truth Voltage', color='blue', linewidth=2.5, alpha=0.8)
        ax2.plot(predictions_plot[i, :, soma_channel_idx], label='Predicted Voltage', color='red', linestyle='--', linewidth=1.5)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Membrane Potential (mV)")
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax2_spike = ax2.twinx()
        ax2_spike.plot(spike_preds_plot[i, :], label='Predicted Spike Probability', color='green', alpha=0.5, linewidth=1.5)
        ax2_spike.set_ylabel('Spike Probability', color='green')
        ax2_spike.tick_params(axis='y', labelcolor='green')
        ax2_spike.set_ylim(0, 1)

        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_spike.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = os.path.join(vis_dir, f"s4d_prediction_sample_{i + 1}.png")
        plt.savefig(save_path)
        plt.close(fig)

    print(f"{args.num_visuals} 张可视化图片已保存。")


def main():
    parser = argparse.ArgumentParser(description="大规模代理网络训练实验 (S4D版本)")
    parser.add_argument('--data_dir', type=str, default='/mnt/data/yukaihuang/neuron_data', help='数据集所在目录')
    parser.add_argument('--model_dir', type=str, default='./models_s4d', help='S4D模型检查点保存目录')
    parser.add_argument('--sequence_length', type=int, default=1024, help='用于训练的滑动窗口长度')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练周期数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--s4d_d_model', type=int, default=64, help='S4D 隐藏维度')
    parser.add_argument('--s4d_N', type=int, default=32, help='S4D 状态维度')
    parser.add_argument('--num_visuals', type=int, default=5, help='可视化样本数')
    args = parser.parse_args()

    print("实验参数配置:")
    for arg in vars(args):
        print(f"  - {arg}: {getattr(args, arg)}")

    train_loader, val_loader, test_loader = get_dataloaders(args)

    print(f"初始化S4D模型: d_model={args.s4d_d_model}, N={args.s4d_N}")
    model = SurrogateS4D(
        d_model=args.s4d_d_model,
        N=args.s4d_N,
        v0_features=20,
        v1_features=6
    )

    train(args, model, train_loader, val_loader)

    test_and_visualize(args, test_loader)


if __name__ == '__main__':
    main()