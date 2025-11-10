# ~/sdn_neuron_exp/models/tcn.py

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List
import torch.nn.functional as F


class CustomVoltageActivation(nn.Module):
    """
    自定义激活函数 (已修正为三段式 C^1 连续函数):
    - f(x) = 2x                          (如果 x <= 0)
    - f(x) = 2(x - 0.5)^3 + 0.5x + 0.25    (如果 0 < x < 1)
    - f(x) = 2x - 1                      (如果 x >= 1)

    该函数在 x=0 和 x=1 处 C^1 连续 (值和导数都连续)。
    - 导数在 x=0.5 处为 0.5
    - 导数在 x=0 和 x=1 处为 2，并平滑过渡到两侧的线性部分。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 定义三个部分的计算

        # 中间部分 (0 < x < 1)
        poly_part = 2.0 * torch.pow(x - 0.5, 3) + 0.5 * x + 0.25

        # 左侧部分 (x <= 0)
        left_part = 2.0 * x

        # 右侧部分 (x >= 1)
        right_part = 2.0 * x - 1.0

        # 2. 创建条件 masks
        condition_mid = (x > 0) & (x < 1)
        condition_left = (x <= 0)

        # 3. 使用嵌套的 torch.where 高效地应用分段函数
        #    (这比使用多个掩码和乘法更推荐)

        # 3a. 首先, 构建 "非中间" (即 x <= 0 或 x >= 1) 的部分
        #     - 如果 x <= 0, 使用 left_part
        #     - 否则 (即 x >= 1), 使用 right_part
        others_part = torch.where(condition_left, left_part, right_part)

        # 3b. 然后, 构建最终结果
        #     - 如果 0 < x < 1, 使用 poly_part
        #     - 否则, 使用我们刚计算的 others_part
        return torch.where(condition_mid, poly_part, others_part)


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
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        # 将上述层按顺序组合
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)

        # 如果输入输出通道数不同，需要一个1x1卷积来匹配维度以便进行残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()

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
                 use_voltage_filter: bool = False,
                 voltage_activation: str = 'linear'):
        super(TCNModel, self).__init__()
        self.fusion_mode = fusion_mode
        self.use_voltage_filter = use_voltage_filter
        self.output_channels = output_channels

        # --- ★★★ 关键修改 1: 添加一个独立的1x1卷积用于升维 ★★★ ---
        # 这个层只负责将 20 通道 -> 48 通道，它没有残差连接
        self.input_projection = nn.Conv1d(input_channels, num_hidden_channels[0], 1)
        # --- 修改结束 ---

        # --- ★★★ 关键修改 2: 修改 input_processor ★★★ ---
        # 现在它的输入和输出通道数相同 (都等于 num_hidden_channels[0])
        self.input_processor = TemporalBlock(
            n_inputs=num_hidden_channels[0],  # <-- [修改] 原为 input_channels
            n_outputs=num_hidden_channels[0], # <-- [修改] 保持不变
            kernel_size=input_kernel_size,
            stride=1,
            dilation=1,
            padding=(input_kernel_size - 1) * 1,
            dropout=dropout
        )
        # 这样修改后, input_processor 内部的 n_inputs == n_outputs，
        # self.downsample 将为 None, 残差路径将是恒等映射。

        # --- 状态调节模块 (不变) ---
        self.state_conditioner = nn.Linear(output_channels, num_hidden_channels[0])

        # --- TCN核心 (不变) ---
        # (这部分已经是恒等映射了, 因为 num_hidden_channels 列表中的值都相同)
        layers = []
        num_levels = len(num_hidden_channels)
        for i in range(1, num_levels):
            dilation_size = 2 ** (i)
            in_channels = num_hidden_channels[i - 1]
            out_channels = num_hidden_channels[i]

            layers.append(TemporalBlock(in_channels, out_channels, tcn_kernel_size, stride=1, dilation=dilation_size,
                                        padding=(tcn_kernel_size - 1) * dilation_size, dropout=dropout))

        self.tcn_core = nn.Sequential(*layers)

        self.output_layer = nn.Conv1d(num_hidden_channels[-1], output_channels, 1)

        # --- ↓↓↓ 添加这个逻辑块 ↓↓↓ ---
        if voltage_activation == 'linear':
            self.voltage_activation = nn.Identity()
        elif voltage_activation == 'custom_x2':
            self.voltage_activation = CustomVoltageActivation()
        else:
            raise ValueError(f"未知的 voltage_activation: {voltage_activation}")
        # --- ↑↑↑ 添加结束 ↑↑↑ ---

        if self.use_voltage_filter:
            # 不要再使用 nn.Sequential
            self.voltage_filter_conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=256, padding=0,
                                                  bias=False)  # 改为 padding=0
            self.voltage_filter_relu = nn.GELU()  # 保持一致使用 GELU
            self.voltage_filter_conv2 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1)

    def forward(self, stimulus_seq: torch.Tensor, initial_state: torch.Tensor,
                target_v1: torch.Tensor = None) -> torch.Tensor:
        """
        模型的前向传播 (已更新)
        """
        # --- ★★★ 关键修改 3: 在 input_processor 之前应用升维 ★★★ ---
        # 1. 先用 1x1 卷积将 (B, 20, L) -> (B, 48, L)
        projected_stimulus = self.input_projection(stimulus_seq)

        # 2. 将 (B, 48, L) 送入 input_processor
        #    (input_processor 现在使用恒等映射残差连接)
        stim_features = self.input_processor(projected_stimulus)
        # --- 修改结束 (原为: stim_features = self.input_processor(stimulus_seq)) ---

        # 3. 处理并广播初始状态 (不变)
        if self.fusion_mode == 'add':
            state_embedding = self.state_conditioner(initial_state)
            state_embedding_broadcasted = state_embedding.unsqueeze(2).expand_as(stim_features)
            fused_features = stim_features + state_embedding_broadcasted
        elif self.fusion_mode == 'ablate':
            fused_features = stim_features
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        # 4. 通过TCN核心 (不变)
        tcn_output = self.tcn_core(fused_features)
        base_prediction = self.output_layer(tcn_output)

        num_voltage_channels = self.output_channels - 1
        spike_channel_idx = self.output_channels - 1

        # 1. 分离原始的电压和脉冲输出
        pred_voltage_part_raw = base_prediction[:, :num_voltage_channels, :]
        pred_spike_logits_raw = base_prediction[:, spike_channel_idx, :].unsqueeze(1)

        # 2. 只对电压部分应用1.1倍的Sigmoid激活
        # activated_voltage = torch.sigmoid(pred_voltage_part_raw) * 1.1
        activated_voltage = self.voltage_activation(pred_voltage_part_raw)

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
            # 这是默认的滤波器输入 (用于验证/测试)
            input_voltage_for_filter = torch.stack([pred_soma_voltage_raw, pred_derivative], dim=1)

            # 如果是训练且提供了目标，则使用混合信号
            if self.training and target_v1 is not None:
                true_soma_voltage = target_v1[:, soma_voltage_idx, :]
                true_derivative = torch.diff(true_soma_voltage, n=1, dim=-1)
                true_derivative = F.pad(true_derivative, (1, 0), "constant", 0)

                mixed_soma_voltage = (pred_soma_voltage_raw + true_soma_voltage) / 2.0
                mixed_derivative = (pred_derivative + true_derivative) / 2.0

                # --- 仅在训练时覆盖滤波器输入 ---
                input_voltage_for_filter = torch.stack([mixed_soma_voltage, mixed_derivative], dim=1)

            # --- 修正：将滤波器应用逻辑移到 'if' 块之外 ---
            # 无论是否在训练，都必须应用滤波器

            # 1. 手动实现因果填充
            # kernel_size = 256, 所以我们需要 255 的左侧填充
            padded_input = F.pad(input_voltage_for_filter, (255, 0), "constant", 0)

            # 2. 手动应用 filter 的各个层
            x_filter = self.voltage_filter_conv1(padded_input)
            x_filter = self.voltage_filter_relu(x_filter)
            filter_effect = self.voltage_filter_conv2(x_filter)

            # 3. 计算最终 logits
            final_spike_logits = pred_spike_logits_raw + filter_effect

            # 4. 组合已激活的电压和最终的脉冲logits
            return torch.cat([activated_voltage, final_spike_logits], dim=1)

        else:
            # (如果 use_voltage_filter 为 False, 则执行此操作)
            # 组合已激活的电压和原始的脉冲logits
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