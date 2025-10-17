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


# --- 数据集类 (修改后) ---
class NeuronTimeSeriesDataset(Dataset):
    def __init__(self, v0_data, v1_data, sequence_length, indices):
        """
        修改后的数据集类，只处理指定索引范围内的样本。

        :param v0_data: 完整的v0数据
        :param v1_data: 完整的v1数据
        :param sequence_length: 序列窗口长度
        :param indices: 这个数据集实例应该使用的索引列表
        """
        super().__init__()
        self.v0_data = v0_data
        self.v1_data = v1_data
        self.sequence_length = sequence_length
        self.indices = indices  # 只使用传入的索引

    def __len__(self):
        # 长度是分配给这个数据集的索引数量
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        v0_seq = self.v0_data[actual_idx: actual_idx + self.sequence_length]
        v1_initial = self.v1_data[actual_idx]
        v1_target = self.v1_data[actual_idx + 1: actual_idx + 1 + self.sequence_length]

        # --- 新增内容 ---
        # 脉冲发放的阈值，高于此电压的时刻被认为是脉冲
        spike_threshold = 0
        # 我们只关心 soma 的脉冲，即 v1 的第6个通道 (索引为5)
        spike_target = (v1_target[:, 5] > spike_threshold).float()

        # 返回四个值
        return v0_seq, v1_initial, v1_target, spike_target


# --- TCN核心残差模块 ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        一个TCN残差模块。

        :param n_inputs: 输入通道数
        :param n_outputs: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param dilation: 空洞系数
        :param padding: 填充大小
        :param dropout: Dropout比例
        """
        super(TemporalBlock, self).__init__()

        # 第一个卷积层 + WeightNorm + GELU + Dropout
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 这里的 padding 是为了实现因果卷积效果
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层 + WeightNorm + GELU + Dropout
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        # 将上面的层打包成一个序列
        self.net = nn.Sequential(self.conv1, self.chomp1, self.gelu1, self.dropout1,
                                 self.conv2, self.chomp2, self.gelu2, self.dropout2)

        # 如果输入和输出通道数不同，需要一个1x1卷积来匹配维度以便进行残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.gelu_out = nn.GELU()
        self.init_weights()

    def init_weights(self):
        # 初始化权重
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        前向传播。
        x 的形状: (Batch, Channels, Seq_Len)
        """
        # 通过两层卷积网络
        out = self.net(x)
        # 进行残差连接
        res = x if self.downsample is None else self.downsample(x)
        # 将网络输出和残差相加，并通过最后的GELU激活
        return self.gelu_out(out + res)


# --- 代理网络模型 (TCN版本) ---
class SurrogateNet(nn.Module):
    def __init__(self, num_inputs=26, num_channels=[32, 32, 32, 32], num_outputs=6, kernel_size=7, dropout=0.2):
        """
        完整的TCN模型，由多个TemporalBlock堆叠而成。

        :param num_inputs: 初始输入通道数 (v0+v1_initial)
        :param num_channels: 一个列表，定义了每个残差模块的输出通道数 (也即网络的深度)
        :param num_outputs: 最终输出通道数 (v1的维度)
        :param kernel_size: 卷积核大小
        :param dropout: Dropout比例
        """
        super(SurrogateNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 空洞系数按2的幂次增加
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        # self.output_conv = nn.Conv1d(num_channels[-1], num_outputs, 1)
        self.voltage_out = nn.Conv1d(num_channels[-1], num_outputs, 1)
        self.spike_out = nn.Conv1d(num_channels[-1], 1, 1)  # 只为soma预测一个脉冲概率通道

        self._calculate_params()

    def forward(self, v0_seq, v1_initial):
        """
        模型的前向传播。
        v0_seq: (Batch, Features_v0, Seq_Len) -> (B, 20, L)
        v1_initial: (Batch, Features_v1) -> (B, 6)
        """
        # --- 新增的预处理部分 ---
        # 说明：由于v0的输入只有-70和30两个值，直接输入模型可能会让模型误解其数值关系。
        # 因此，这里将其映射为0和1的二进制信号。
        # 创建一个和 v0_seq 形状相同、元素全为0的张量
        v0_processed = torch.zeros_like(v0_seq)
        # 将 v0_seq 中等于 30 的位置在 v0_processed 中置为 1
        v0_processed[v0_seq == 30] = 1.0
        # --- 预处理结束 ---

        seq_len = v0_seq.shape[2]
        v1_expanded = v1_initial.unsqueeze(2).expand(-1, -1, seq_len)

        # 将处理后的 v0_processed 与 v1_expanded 拼接
        combined_input = torch.cat([v0_processed, v1_expanded], dim=1)

        # 通过TCN网络
        tcn_out = self.tcn(combined_input)
        # 最后的1x1卷积层，用于输出最终结果
        # final_output = self.output_conv(tcn_out)

        # return final_output
        # TCN的输出分别进入两个输出头
        voltage_prediction = self.voltage_out(tcn_out)
        spike_prediction = self.spike_out(tcn_out)

        # 返回两个预测结果
        return voltage_prediction, spike_prediction

    def _calculate_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SurrogateNet (TCN) 初始化完毕，总参数量: {total_params}")


# --- 数据加载函数 (修正后) ---
def get_dataloaders(args):
    """
    加载神经元数据集，并按【时间顺序】切分训练/验证/测试集，避免数据泄露。
    """
    # ... (加载v0_full, v1_full的代码保持不变) ...
    v0_path = os.path.join(args.data_dir, 'v0_small.npy')
    v1_path = os.path.join(args.data_dir, 'v1_small.npy')

    if not (os.path.exists(v0_path) and os.path.exists(v1_path)):
        raise FileNotFoundError(f"错误：在 {args.data_dir} 中找不到 v0(_small).npy 或 v1(_small).npy。")

    print(f"从 {v0_path} 和 {v1_path} 加载数据...")
    v0_full = torch.from_numpy(np.load(v0_path)).float().squeeze(0).transpose(0, 1)
    v1_full = torch.from_numpy(np.load(v1_path)).float().squeeze(0).transpose(0, 1)

    # 计算所有可能的起始索引
    num_possible_samples = v0_full.shape[0] - args.sequence_length
    all_indices = np.arange(num_possible_samples)

    # --- 关键修正：不再随机打乱索引 ---
    # np.random.shuffle(all_indices) # <--- 删除或注释掉这一行

    # 按 6:1:3 的比例，【按顺序】切分索引块
    train_end_idx = int(0.6 * num_possible_samples)
    val_end_idx = train_end_idx + int(0.1 * num_possible_samples)

    # 直接使用连续的索引块
    train_indices = all_indices[:train_end_idx]
    val_indices = all_indices[train_end_idx:val_end_idx]
    test_indices = all_indices[val_end_idx:]

    # 创建Dataset实例，传入各自的索引子集
    train_dataset = NeuronTimeSeriesDataset(v0_full, v1_full, args.sequence_length, train_indices)
    val_dataset = NeuronTimeSeriesDataset(v0_full, v1_full, args.sequence_length, val_indices)
    test_dataset = NeuronTimeSeriesDataset(v0_full, v1_full, args.sequence_length, test_indices)

    print(f"数据集划分: 训练样本数={len(train_dataset)}, 验证样本数={len(val_dataset)}, 测试样本数={len(test_dataset)}")

    # 注意：在训练加载器中，shuffle=True 仍然是推荐的。
    # 这只是在每个epoch中打乱训练样本的顺序，而不会导致训练集和验证集之间的数据泄露。
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def train(args, model, train_loader, val_loader):
    """训练模型，并保存最佳检查点"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    model.to(device)

    # 在 train 函数中，替换 criterion = nn.MSELoss()
    # --- 修改内容 ---
    # 为两个任务分别定义损失函数
    criterion_voltage = nn.MSELoss()
    # BCEWithLogitsLoss 更稳定，因为它内置了sigmoid激活函数
    criterion_spike = nn.BCEWithLogitsLoss()
    # 在 train 函数中
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # --- 新增内容 ---
    # 定义一个学习率调度器，当验证损失在3个epoch内不下降时，学习率乘以0.5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    checkpoint_path = os.path.join(args.model_dir, "best_cnn_model.pt")  # 文件名区分
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"开始训练 {args.epochs} 个 epochs...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        # 1. 解包新的三元组数据
        # 在 train 函数中，替换 for v0_seq, v1_initial, v1_target in train_loader: 循环的内部
        # --- 修改内容 ---
        for v0_seq, v1_initial, v1_target, spike_target in train_loader:  # 1. 解包新的四元组
            v0_seq = v0_seq.transpose(1, 2).to(device)
            v1_initial = v1_initial.to(device)
            v1_target = v1_target.transpose(1, 2).to(device)
            # 将spike_target增加一个通道维度以匹配模型输出 (B, L) -> (B, 1, L)
            spike_target = spike_target.unsqueeze(1).to(device)

            optimizer.zero_grad()

            # 2. 获取模型的两个输出
            voltage_pred, spike_pred_logits = model(v0_seq, v1_initial)

            # 3. 实现目标钳制 (Target Clipping)
            voltage_clamp_threshold = -55.0
            # 将目标和预测电压中高于阈值的部分都设置为阈值
            v1_target_clamped = torch.clamp(v1_target, max=voltage_clamp_threshold)
            # voltage_pred_clamped = torch.clamp(voltage_pred, max=voltage_clamp_threshold)

            # 4. 分别计算两个任务的损失
            loss_voltage = criterion_voltage(voltage_pred, v1_target_clamped)
            loss_spike = criterion_spike(spike_pred_logits, spike_target)

            # 5. 加权求和得到最终损失 (脉冲的权重设为1.0，电压的权重设为0.05，可调)
            loss = 1.0 * loss_spike + 0.005 * loss_voltage

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # 在 train 函数中，替换 model.eval() 及其后的 with torch.no_grad(): 循环

        model.eval()
        running_val_loss = 0.0
        running_val_loss_v = 0.0  # 用于记录电压损失
        running_val_loss_s = 0.0  # 用于记录脉冲损失
        with torch.no_grad():
            # --- 新增内容 ---
            for v0_seq, v1_initial, v1_target, spike_target in val_loader:  # 1. 解包四元组
                v0_seq = v0_seq.transpose(1, 2).to(device)
                v1_initial = v1_initial.to(device)
                v1_target = v1_target.transpose(1, 2).to(device)
                spike_target = spike_target.unsqueeze(1).to(device)

                # 2. 获取模型的两个输出
                voltage_pred, spike_pred_logits = model(v0_seq, v1_initial)

                # 3. 同样进行目标钳制
                voltage_clamp_threshold = -55.0
                v1_target_clamped = torch.clamp(v1_target, max=voltage_clamp_threshold)
                # voltage_pred_clamped = torch.clamp(voltage_pred, max=voltage_clamp_threshold)

                # 4. 分别计算并累加两个任务的损失
                loss_voltage = criterion_voltage(voltage_pred, v1_target_clamped)
                loss_spike = criterion_spike(spike_pred_logits, spike_target)

                # 5. 计算加权总损失并累加
                loss = 1.0 * loss_spike + 0.005 * loss_voltage
                running_val_loss += loss.item()
                running_val_loss_v += loss_voltage.item()
                running_val_loss_s += loss_spike.item()

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_loss_v = running_val_loss_v / len(val_loader)
        avg_val_loss_s = running_val_loss_s / len(val_loader)

        # 打印更详细的损失信息
        print(
            f"Epoch [{epoch + 1:03d}/{args.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (v_loss: {avg_val_loss_v:.6f}, s_loss: {avg_val_loss_s:.6f})")

        # --- 修改结束 ---

        # 在 train 函数的 epoch 循环末尾

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> 验证损失创新低，模型已保存至: {checkpoint_path}")

        # --- 新增内容 ---
        # 在每个epoch结束后，根据验证损失更新学习率
        scheduler.step(avg_val_loss)


# --- 完整替换 test_and_visualize 函数 ---
def test_and_visualize(args, test_loader):
    """在测试集上评估并将可视化结果保存为图片"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(args.model_dir, "best_cnn_model.pt")
    vis_dir = os.path.join(args.model_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print("错误：找不到模型检查点。请先进行训练。")
        return

    # 定义与训练时完全相同的模型结构
    num_channels = [24] * 3
    kernel_size = 7
    dropout = 0.2
    model = SurrogateNet(
        num_inputs=26,
        num_channels=num_channels,
        num_outputs=6,
        kernel_size=kernel_size,
        dropout=dropout
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    criterion_voltage = nn.MSELoss()
    criterion_spike = nn.BCEWithLogitsLoss()
    total_v_loss, total_s_loss, total_loss = 0.0, 0.0, 0.0

    print("\n在测试集上评估最佳模型...")
    with torch.no_grad():
        for v0_seq, v1_initial, v1_target, spike_target in test_loader:
            v0_seq = v0_seq.transpose(1, 2).to(device)
            v1_initial = v1_initial.to(device)
            v1_target = v1_target.transpose(1, 2).to(device)
            spike_target = spike_target.unsqueeze(1).to(device)

            voltage_pred, spike_pred_logits = model(v0_seq, v1_initial)

            voltage_clamp_threshold = -55.0
            v1_target_clamped = torch.clamp(v1_target, max=voltage_clamp_threshold)
            # voltage_pred_clamped = torch.clamp(voltage_pred, max=voltage_clamp_threshold)

            loss_v = criterion_voltage(voltage_pred, v1_target_clamped)
            loss_s = criterion_spike(spike_pred_logits, spike_target)

            total_v_loss += loss_v.item()
            total_s_loss += loss_s.item()
            total_loss += (1.0 * loss_s + 0.005 * loss_v).item()

    avg_test_loss = total_loss / len(test_loader)
    avg_test_v_loss = total_v_loss / len(test_loader)
    avg_test_s_loss = total_s_loss / len(test_loader)
    print(f"======================================================")
    print(f"TCN模型 - 最终测试集加权损失: {avg_test_loss:.6f}")
    print(f"          - 电压MSE损失 (钳制后): {avg_test_v_loss:.6f}")
    print(f"          - 脉冲BCE损失: {avg_test_s_loss:.6f}")
    print(f"========================================================")

    # 可视化并保存
    print(f"\n在测试集上进行可视化，图片将保存至: {vis_dir}")
    v0_seq_b, v1_initial_b, v1_target_b, spike_target_b = next(iter(test_loader))

    with torch.no_grad():
        v0_model_in = v0_seq_b.to(device).transpose(1, 2)
        v1_initial_in = v1_initial_b.to(device)
        voltage_pred_b, spike_pred_logits_b = model(v0_model_in, v1_initial_in)

    # 将v0的-70/30也转换为0/1用于可视化
    v0_plot = (v0_seq_b.numpy() > 0).astype(float)
    v1_target_plot = v1_target_b.numpy()
    predictions_plot = voltage_pred_b.cpu().transpose(1, 2).numpy()

    # 将脉冲信息添加到可视化中
    spike_preds_plot = torch.sigmoid(spike_pred_logits_b).cpu().squeeze(1).numpy()


    for i in range(min(args.num_visuals, v0_plot.shape[0])):
        soma_channel_idx = 5

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})
        fig.suptitle(f'TCN Prediction vs. Ground Truth - Sample {i+1}', fontsize=16)

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

        # 创建第二个y轴来绘制脉冲概率
        ax2_spike = ax2.twinx()
        ax2_spike.plot(spike_preds_plot[i, :], label='Predicted Spike Probability', color='green', alpha=0.5, linewidth=1.5)
        ax2_spike.set_ylabel('Spike Probability', color='green')
        ax2_spike.tick_params(axis='y', labelcolor='green')
        ax2_spike.set_ylim(0, 1)

        # 合并图例
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_spike.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = os.path.join(vis_dir, f"tcn_prediction_sample_{i + 1}.png")
        plt.savefig(save_path)
        plt.close(fig)

    print(f"{args.num_visuals} 张可视化图片已保存。")


def main():
    parser = argparse.ArgumentParser(description="大规模代理网络训练实验")
    parser.add_argument('--data_dir', type=str, default='/mnt/data/yukaihuang/neuron_data', help='数据集所在目录')
    parser.add_argument('--model_dir', type=str, default='./models_cnn', help='CNN模型检查点保存目录')
    parser.add_argument('--sequence_length', type=int, default=1024, help='用于训练的滑动窗口长度')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练周期数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--num_visuals', type=int, default=5, help='可视化样本数')
    args = parser.parse_args()

    print("实验参数配置:")
    for arg in vars(args):
        print(f"  - {arg}: {getattr(args, arg)}")

    # 1. 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(args)

    # main 函数中, 替换 model = SurrogateNet()
    # 2. 初始化TCN模型并训练
    num_channels = [24] * 3
    kernel_size = 7
    dropout = 0.2

    print(
        f"初始化TCN模型: {len(num_channels)} 层, {num_channels[0]} 通道, kernel_size={kernel_size}, dropout={dropout}")

    model = SurrogateNet(
        num_inputs=26,  # v0 (20) + v1_initial (6)
        num_channels=num_channels,
        num_outputs=6,  # v1 的维度
        kernel_size=kernel_size,
        dropout=dropout
    )

    train(args, model, train_loader, val_loader)

    # 3. 在测试集上评估和可视化
    test_and_visualize(args, test_loader)


if __name__ == '__main__':
    main()