# ~/sdn_neuron_exp/models/tcn.py

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List


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

    该模型接收刺激序列和初始状态作为输入，预测未来的电压和脉冲。

    参数:
        input_channels (int): 输入刺激的通道数 (v0部分, e.g., 20)。
        output_channels (int): 输出状态的通道数 (v1部分, e.g., 7)。
        num_hidden_channels (List[int]): 一个列表，定义了TCN每个残差块的隐藏通道数。列表的长度决定了TCN的层数。
        input_kernel_size (int): 第一个"事件探测"层的卷积核大小。
        tcn_kernel_size (int): TCN核心模块的卷积核大小。
        dropout (float): Dropout比率。
    """

    def __init__(self, input_channels: int, output_channels: int, num_hidden_channels: List[int],
                 input_kernel_size: int, tcn_kernel_size: int, dropout: float = 0.2):
        super(TCNModel, self).__init__()

        # --- 1. “事件探测”输入层 ---
        # 使用较大卷积核处理稀疏的原始刺激输入，并保持因果性
        self.event_detector = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden_channels[0], kernel_size=input_kernel_size,
                      padding=(input_kernel_size - 1)),
            nn.ReLU()
        )

        # --- 2. “状态调节”模块 ---
        # 使用一个线性层将初始状态向量投影到第一个TCN层的隐藏维度
        self.state_conditioner = nn.Linear(output_channels, num_hidden_channels[0])

        # --- 3. TCN核心 ---
        layers = []
        num_levels = len(num_hidden_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_hidden_channels[i - 1] if i > 0 else num_hidden_channels[0]
            out_channels = num_hidden_channels[i]

            layers.append(TemporalBlock(in_channels, out_channels, tcn_kernel_size, stride=1, dilation=dilation_size,
                                        padding=(tcn_kernel_size - 1) * dilation_size, dropout=dropout))

        self.tcn_core = nn.Sequential(*layers)

        # --- 4. 输出层 ---
        # 使用一个1x1卷积将最终的隐藏状态映射回7个输出通道
        self.output_layer = nn.Conv1d(num_hidden_channels[-1], output_channels, 1)

    def forward(self, stimulus_seq: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播。

        参数:
            stimulus_seq (Tensor): 刺激输入序列，形状为 [batch, input_channels, seq_len]。
            initial_state (Tensor): 片段的初始状态（上下文状态），形状为 [batch, output_channels]。

        返回:
            Tensor: 预测的输出序列，形状为 [batch, output_channels, seq_len]。
        """
        # 1. 处理刺激序列
        stim_features = self.event_detector(stimulus_seq)
        # 裁剪掉因果卷积产生的右侧padding
        stim_features = stim_features[:, :, :-self.event_detector[0].padding[0]]

        # 2. 处理并广播初始状态
        state_embedding = self.state_conditioner(initial_state)  # -> [batch, hidden_channels]
        # unsqueeze增加一个维度 -> [batch, hidden_channels, 1]
        # expand_as使其形状与stim_features匹配，以便广播相加
        state_embedding_broadcasted = state_embedding.unsqueeze(2).expand_as(stim_features)

        # 3. 融合状态与刺激特征
        fused_features = stim_features + state_embedding_broadcasted

        # 4. 通过TCN核心
        tcn_output = self.tcn_core(fused_features)

        # 5. 通过输出层得到最终预测
        prediction = self.output_layer(tcn_output)

        return prediction


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