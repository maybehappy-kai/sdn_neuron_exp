# ~/sdn_neuron_exp/models/tcn.py

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """
    TCN的基本残差模块。
    包含两个因果、空洞卷积层，并应用了权重归一化、Dropout和残差连接。

    参数:
        n_inputs (int): 输入通道数。
        n_outputs (int): 输出通道数。
        kernel_size (int): 卷积核大小。
        stride (int): 卷积步长，通常为1。
        dilation (int): 空洞因子。
        padding (int): 左侧填充大小，用于维持因果性。
        dropout (float): Dropout比率。
    """

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int,
                 dropout: float = 0.2):
        super(TemporalBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 激活函数和Dropout
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将上述层按顺序组合
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)

        # 如果输入输出通道数不同，需要一个1x1卷积来匹配维度以便进行残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # 修复：直接将输出的序列长度裁剪为和输入x的序列长度一致
        out = out[:, :, :x.shape[2]]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    带状态调节功能的TCN模型。
    (已更新: 第一层使用完整的TemporalBlock以进行更强大的初始处理)
    """

    def __init__(self, input_channels: int, output_channels: int, num_hidden_channels: List[int],
                 input_kernel_size: int, tcn_kernel_size: int, dropout: float = 0.2, fusion_mode: str = 'add',
                 use_voltage_filter: bool = False):
        super(TCNModel, self).__init__()
        self.fusion_mode = fusion_mode
        self.use_voltage_filter = use_voltage_filter  # <-- 保存参数
        self.output_channels = output_channels  # <-- 保存通道数以供后续使用

        # --- ★★★ 关键修改 1: 用一个TemporalBlock替换原来的第一层 ★★★ ---
        # 我们现在使用一个完整的、带有残差连接的TemporalBlock作为“事件探测器”。
        # 它的空洞因子固定为1，因为它负责初始的、非空洞的特征提取。
        self.input_processor = TemporalBlock(
            n_inputs=input_channels,
            n_outputs=num_hidden_channels[0],
            kernel_size=input_kernel_size,
            stride=1,
            dilation=1,  # 第一层空洞因子为1
            padding=(input_kernel_size - 1) * 1,  # 因果填充
            dropout=dropout
        )

        # --- 状态调节模块 (不变) ---
        self.state_conditioner = nn.Linear(output_channels, num_hidden_channels[0])

        # --- TCN核心 (逻辑微调，现在从第二层开始) ---
        layers = []
        num_levels = len(num_hidden_channels)
        # 循环从i=1开始，因为i=0（第一层）已经被上面的input_processor处理了
        for i in range(1, num_levels):
            dilation_size = 2 ** (i)  # 空洞因子从2开始
            in_channels = num_hidden_channels[i - 1]
            out_channels = num_hidden_channels[i]

            layers.append(TemporalBlock(in_channels, out_channels, tcn_kernel_size, stride=1, dilation=dilation_size,
                                        padding=(tcn_kernel_size - 1) * dilation_size, dropout=dropout))

        self.tcn_core = nn.Sequential(*layers)

        self.output_layer = nn.Conv1d(num_hidden_channels[-1], output_channels, 1)

        if self.use_voltage_filter:
            self.voltage_to_spike_filter = nn.Sequential(
                # --- ↓↓↓ 将 in_channels 从 1 修改为 2 ↓↓↓ ---
                # <--- 修改: 使用 'causal'  padding 来防止数据泄漏 ---
                nn.Conv1d(in_channels=2, out_channels=8, kernel_size=256, padding='causal', bias=False),
                nn.ReLU(),
                nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1)
            )

    def forward(self, stimulus_seq: torch.Tensor, initial_state: torch.Tensor,
                target_v1: torch.Tensor = None) -> torch.Tensor:
        """
        模型的前向传播 (已更新)
        """
        # 1. 通过新的、更强大的输入处理器
        stim_features = self.input_processor(stimulus_seq)

        # 2. 处理并广播初始状态
        if self.fusion_mode == 'add':
            state_embedding = self.state_conditioner(initial_state)
            state_embedding_broadcasted = state_embedding.unsqueeze(2).expand_as(stim_features)
            fused_features = stim_features + state_embedding_broadcasted
        elif self.fusion_mode == 'ablate':
            fused_features = stim_features
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        # 3. 通过TCN核心
        tcn_output = self.tcn_core(fused_features)
        base_prediction = self.output_layer(tcn_output)

        num_voltage_channels = self.output_channels - 1
        spike_channel_idx = self.output_channels - 1

        # 1. 分离原始的电压和脉冲输出
        pred_voltage_part_raw = base_prediction[:, :num_voltage_channels, :]
        pred_spike_logits_raw = base_prediction[:, spike_channel_idx, :].unsqueeze(1)

        # 2. 只对电压部分应用1.1倍的Sigmoid激活
        # activated_voltage = torch.sigmoid(pred_voltage_part_raw) * 1.1
        activated_voltage = pred_voltage_part_raw

        if self.use_voltage_filter:
            # --- 新增的定义 ---
            soma_voltage_idx = num_voltage_channels - 1  # 假设soma电压是电压通道中的最后一个
            # -----------------

            # 注意：这里的逻辑使用未经激活的原始电压来影响脉冲
            pred_soma_voltage_raw = pred_voltage_part_raw[:, soma_voltage_idx, :]

            # 计算电压差分（近似导数）
            pred_derivative = torch.diff(pred_soma_voltage_raw, n=1, dim=-1)
            pred_derivative = F.pad(pred_derivative, (1, 0), "constant", 0)

            # 将电压和导数拼接成一个 (B, 2, L) 的张量
            input_voltage_for_filter = torch.stack([pred_soma_voltage_raw, pred_derivative], dim=1)

            # 如果是训练且提供了目标，则使用混合信号
            if self.training and target_v1 is not None:
                true_soma_voltage = target_v1[:, soma_voltage_idx, :]
                true_derivative = torch.diff(true_soma_voltage, n=1, dim=-1)
                true_derivative = F.pad(true_derivative, (1, 0), "constant", 0)

                mixed_soma_voltage = (pred_soma_voltage_raw + true_soma_voltage) / 2.0
                mixed_derivative = (pred_derivative + true_derivative) / 2.0
                input_voltage_for_filter = torch.stack([mixed_soma_voltage, mixed_derivative], dim=1)

            filter_effect = self.voltage_to_spike_filter(input_voltage_for_filter)

            final_spike_logits = pred_spike_logits_raw + filter_effect
            # 3. 组合已激活的电压和最终的脉冲logits
            return torch.cat([activated_voltage, final_spike_logits], dim=1)
        else:
            # 3. 组合已激活的电压和原始的脉冲logits
            return torch.cat([activated_voltage, pred_spike_logits_raw], dim=1)

# =============================================================================
#                               示例用法
# =============================================================================
if __name__ == '__main__':
    # --- 模型超参数定义 ---
    # 与数据相关的固定参数
    INPUT_CHANNELS = 20  # v0有20个通道
    OUTPUT_CHANNELS = 7  # v1有7个通道

    # 架构相关的参数 (这些可以在后续实验中调整)
    HIDDEN_CHANNELS = [48, 48, 48, 48]  # 定义一个4层的TCN，每层有48个隐藏通道
    INPUT_KERNEL_SIZE = 15  # 事件探测层的卷积核大小
    TCN_KERNEL_SIZE = 5  # TCN核心的卷积核大小
    DROPOUT = 0.25

    # --- 实例化模型 ---
    print("正在实例化TCN模型...")
    model = TCNModel(
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        num_hidden_channels=HIDDEN_CHANNELS,
        input_kernel_size=INPUT_KERNEL_SIZE,
        tcn_kernel_size=TCN_KERNEL_SIZE,
        dropout=DROPOUT
    )
    print("模型实例化成功！")
    print(model)

    # --- 创建模拟输入数据 ---
    BATCH_SIZE = 8
    SEQ_LENGTH = 1024  # 假设我们将长序列切成1024步长的片段

    # 模拟的刺激序列 (v0 chunk)
    dummy_stimulus = torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQ_LENGTH)
    # 模拟的初始状态 (v1 context)
    dummy_state = torch.randn(BATCH_SIZE, OUTPUT_CHANNELS)

    print(f"\n模拟输入数据形状:")
    print(f"  - 刺激序列 (stimulus_seq): {dummy_stimulus.shape}")
    print(f"  - 初始状态 (initial_state): {dummy_state.shape}")

    # --- 前向传播测试 ---
    try:
        print("\n正在进行前向传播测试...")
        output = model(dummy_stimulus, dummy_state)
        print("前向传播成功！")
        print(f"  - 输出序列形状: {output.shape}")

        # 验证输出形状是否正确
        assert output.shape == (BATCH_SIZE, OUTPUT_CHANNELS, SEQ_LENGTH)
        print("  - 输出形状验证通过！")

    except Exception as e:
        print(f"前向传播失败: {e}")

    # --- 打印模型参数量 ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params / 1e6:.2f} M")