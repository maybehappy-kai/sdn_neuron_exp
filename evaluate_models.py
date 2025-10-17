import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm


# =====================================================================================
#  第1部分: 从训练脚本中复制必要的类定义 (模型, 数据集等)
# =====================================================================================

# --- 数据集类 (与训练时完全一致) ---
class NeuronTimeSeriesDataset(Dataset):
    def __init__(self, v0_data, v1_data, sequence_length, indices):
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
        spike_target = (v1_target[:, 5] > 0).float()  # 脉冲目标，这里计算VE用不到但保持结构一致
        return v0_seq, v1_initial, v1_target, spike_target


# --- TCN 模型定义 ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.gelu1, self.dropout1,
                                 self.conv2, self.chomp2, self.gelu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.gelu_out = nn.GELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.gelu_out(out + res)


class SurrogateNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, kernel_size=7, dropout=0.2):
        super(SurrogateNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        self.voltage_out = nn.Conv1d(num_channels[-1], num_outputs, 1)
        self.spike_out = nn.Conv1d(num_channels[-1], 1, 1)

    def forward(self, v0_seq, v1_initial):
        v0_processed = torch.zeros_like(v0_seq)
        v0_processed[v0_seq == 30] = 1.0
        seq_len = v0_seq.shape[2]
        v1_expanded = v1_initial.unsqueeze(2).expand(-1, -1, seq_len)
        combined_input = torch.cat([v0_processed, v1_expanded], dim=1)
        tcn_out = self.tcn(combined_input)
        voltage_prediction = self.voltage_out(tcn_out)
        spike_prediction = self.spike_out(tcn_out)
        return voltage_prediction, spike_prediction


# --- S4D 模型定义 ---
class S4D(nn.Module):
    def __init__(self, d_model, N=64):
        super().__init__()
        # =================== 已修正 ===================
        self.d_model = d_model
        self.N = N
        # ============================================
        A_re = -0.5 * torch.ones(d_model, N)
        A_im = torch.arange(N, dtype=torch.float32).view(1, N).expand(d_model, -1)
        self.A = nn.Parameter(torch.complex(A_re, A_im))
        self.C = nn.Parameter(torch.randn(d_model, N, dtype=torch.cfloat))
        self.log_dt = nn.Parameter(torch.rand(d_model) * (np.log(0.1) - np.log(0.001)) + np.log(0.001))
        self.B = torch.ones(self.d_model, self.N, dtype=torch.cfloat)

    def forward(self, u):
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


class SurrogateS4D(nn.Module):
    def __init__(self, d_model, N, v0_features=20, v1_features=6):
        super().__init__()
        self.in_proj = nn.Linear(v0_features, d_model)
        self.h0_proj = nn.Linear(v1_features, d_model)
        self.s4d = S4D(d_model=d_model, N=N)
        self.activation = nn.GELU()
        self.voltage_out = nn.Linear(d_model, v1_features)
        self.spike_out = nn.Linear(d_model, 1)

    def forward(self, v0_seq, v1_initial):
        v0_processed = torch.zeros_like(v0_seq)
        v0_processed[v0_seq == 30] = 1.0
        x = self.in_proj(v0_processed)
        x[:, 0, :] = x[:, 0, :] + self.h0_proj(v1_initial)
        x = self.s4d(x)
        x = self.activation(x)
        voltage_prediction = self.voltage_out(x)
        spike_prediction = self.spike_out(x)
        return voltage_prediction.transpose(1, 2), spike_prediction.transpose(1, 2)


# =====================================================================================
#  第2部分: 评估逻辑 (保持不变)
# =====================================================================================

def get_test_loader(data_dir, sequence_length, batch_size):
    """只加载并创建测试集数据加载器"""
    v0_path = os.path.join(data_dir, 'v0_small.npy')
    v1_path = os.path.join(data_dir, 'v1_small.npy')

    print(f"从 {v0_path} 和 {v1_path} 加载数据...")
    v0_full = torch.from_numpy(np.load(v0_path)).float().squeeze(0).transpose(0, 1)
    v1_full = torch.from_numpy(np.load(v1_path)).float().squeeze(0).transpose(0, 1)

    num_possible_samples = v0_full.shape[0] - sequence_length
    all_indices = np.arange(num_possible_samples)

    train_end_idx = int(0.6 * num_possible_samples)
    val_end_idx = train_end_idx + int(0.1 * num_possible_samples)
    test_indices = all_indices[val_end_idx:]

    test_dataset = NeuronTimeSeriesDataset(v0_full, v1_full, sequence_length, test_indices)
    print(f"测试集样本数: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def calculate_ve(model, test_loader, device, model_type):
    """计算并返回指定模型在测试集上的方差解释率 VE"""
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print(f"正在评估 {model_type} 模型...")
    with torch.no_grad():
        for v0_seq, v1_initial, v1_target, _ in test_loader:
            v1_initial = v1_initial.to(device)

            if model_type == 'TCN':
                v0_seq = v0_seq.transpose(1, 2).to(device)
            else:
                v0_seq = v0_seq.to(device)

            voltage_pred, _ = model(v0_seq, v1_initial)

            all_preds.append(voltage_pred.cpu().numpy())
            all_targets.append(v1_target.transpose(1, 2).numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    mse_overall = np.mean((y_true - y_pred) ** 2)
    var_overall = np.var(y_true)
    ve_overall = 1 - (mse_overall / var_overall)

    soma_channel_idx = 5
    y_true_soma = y_true[:, soma_channel_idx, :]
    y_pred_soma = y_pred[:, soma_channel_idx, :]

    mse_soma = np.mean((y_true_soma - y_pred_soma) ** 2)
    var_soma = np.var(y_true_soma)
    ve_soma = 1 - (mse_soma / var_soma)

    return ve_overall, ve_soma


def main():
    parser = argparse.ArgumentParser(description="评估TCN和S4D模型的VE")
    parser.add_argument('--data_dir', type=str, default='/mnt/data/yukaihuang/neuron_data', help='数据集目录')
    parser.add_argument('--cnn_model_path', type=str, default='./models_cnn/best_cnn_model.pt', help='TCN模型权重路径')
    parser.add_argument('--s4d_model_path', type=str, default='./models_s4d/best_s4d_model.pt', help='S4D模型权重路径')
    parser.add_argument('--sequence_length', type=int, default=1024, help='序列长度')
    parser.add_argument('--batch_size', type=int, default=64, help='评估时的批处理大小')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    test_loader = get_test_loader(args.data_dir, args.sequence_length, args.batch_size)

    if os.path.exists(args.cnn_model_path):
        cnn_model = SurrogateNet(
            num_inputs=26,
            num_channels=[24] * 3,
            num_outputs=6,
            kernel_size=7,
            dropout=0.2
        )
        cnn_model.load_state_dict(torch.load(args.cnn_model_path, map_location=device))
        ve_overall_cnn, ve_soma_cnn = calculate_ve(cnn_model, test_loader, device, 'TCN')
    else:
        print(f"警告：找不到TCN模型权重 {args.cnn_model_path}，跳过评估。")
        ve_overall_cnn, ve_soma_cnn = "N/A", "N/A"

    if os.path.exists(args.s4d_model_path):
        s4d_model = SurrogateS4D(
            d_model=64,
            N=32,
            v0_features=20,
            v1_features=6
        )
        s4d_model.load_state_dict(torch.load(args.s4d_model_path, map_location=device))
        ve_overall_s4d, ve_soma_s4d = calculate_ve(s4d_model, test_loader, device, 'S4D')
    else:
        print(f"警告：找不到S4D模型权重 {args.s4d_model_path}，跳过评估。")
        ve_overall_s4d, ve_soma_s4d = "N/A", "N/A"

    print("\n==================== 最终评估结果 ====================")
    print(f" TCN 模型 ({os.path.basename(args.cnn_model_path)}):")
    print(f"  - 总体方差解释率 (VE)      : {ve_overall_cnn:.4f}")
    print(f"  - Soma通道方差解释率 (VE)  : {ve_soma_cnn:.4f}")
    print("---------------------------------------------------------")
    print(f" S4D 模型 ({os.path.basename(args.s4d_model_path)}):")
    print(f"  - 总体方差解释率 (VE)      : {ve_overall_s4d:.4f}")
    print(f"  - Soma通道方差解释率 (VE)  : {ve_soma_s4d:.4f}")
    print("=========================================================\n")


if __name__ == '__main__':
    main()