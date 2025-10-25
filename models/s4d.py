# ~/sdn_neuron_exp/models/s4d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import torch.nn.functional as F



class S4D(nn.Module):
    """
    S4D-Lin层 (对角版本的S4)。
    从零开始实现，不依赖第三方库。
    实现了并行的卷积模式(forward)和串行的循环模式(step)。
    """

    def __init__(self, d_model, d_state=64, l_max=1024, bidirectional=False, **kwargs):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.l_max = l_max
        self.bidirectional = bidirectional

        # HiPPO-LegS 初始化 A 矩阵
        A = -0.5 * torch.ones(self.n) + 1j * torch.arange(self.n) * np.pi
        self.A = nn.Parameter(A)

        self.C = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat))
        self.D = nn.Parameter(torch.randn(self.h))
        self.log_step = nn.Parameter(torch.rand(self.h) * (np.log(0.1) - np.log(1e-4)) + np.log(1e-4))

        # 在__init__中不再预计算超大卷积核，改为在forward中按需计算
        # 这可以加快模型实例化速度

        # 预计算循环模式下的参数
        # (需要手动移到GPU)
        self.dA, self.dC = self.compute_recurrent_params()

    def compute_kernel(self, L: int) -> torch.Tensor:
        """ 计算S4D的卷积核 K (已修正) """
        step = torch.exp(self.log_step)  # Δ, shape (H,)
        C = self.C  # shape (H, N)
        A = self.A  # shape (N,)

        # Δ * A, 形状 (H, N)
        delta_A = step.unsqueeze(-1) * A.unsqueeze(0)

        # 离散化 C, 形状 (H, N)
        C_bar = C * (torch.exp(delta_A) - 1.0) / A.unsqueeze(0)

        # A_bar 的幂, 形状 (L, H, N)
        l_vals = torch.arange(L, device=A.device).view(L, 1, 1)
        powers_of_A_bar = torch.exp(l_vals * delta_A.unsqueeze(0))

        # 计算卷积核 K, 形状 (H, L)
        K = torch.einsum('hn,lhn->hl', C_bar, powers_of_A_bar)

        return K.cfloat()  # 保持复数类型

    def compute_recurrent_params(self):
        """ 预计算循环模式下的离散化参数 A_bar 和 C_bar """
        step = torch.exp(self.log_step) # Δ
        A = self.A
        C = self.C
        dA = torch.exp(step.unsqueeze(-1) * A)
        dC = C * (dA - 1) / A
        return dA.cfloat(), dC.cfloat()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """ 卷积模式 (用于训练) """
        L = u.size(-1)

        # 按需计算卷积核
        K = self.compute_kernel(L)

        # 使用FFT进行快速卷积
        k_f = torch.fft.rfft(K.real, n=2 * L) + 1j * torch.fft.rfft(K.imag, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y_f = k_f * u_f
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]

        # 双向处理
        if self.bidirectional:
            if L > self.l_max:
                K_rev = self.compute_kernel(L)
            else:
                K_rev = self.K_rev[:, :L]
            K_rev = K_rev.to(u.device)

            u_rev = torch.flip(u, dims=[-1])
            k_rev_f = torch.fft.rfft(K_rev, n=2 * L)
            u_rev_f = torch.fft.rfft(u_rev, n=2 * L)
            y_rev_f = k_rev_f * u_rev_f
            y_rev = torch.fft.irfft(y_rev_f, n=2 * L)[..., :L]
            y = y + torch.flip(y_rev, dims=[-1])

        # 加上跳跃连接 D*u
        y = y + u * self.D.unsqueeze(-1)
        return y

    def step(self, u_step, x_prev):
        """ 循环模式 (用于自回归推理) """
        if x_prev is None:
            x_prev = torch.zeros(u_step.size(0), self.h, self.n, device=u_step.device, dtype=torch.cfloat)

        if self.dA.device != u_step.device:
            self.dA = self.dA.to(u_step.device)
            self.dC = self.dC.to(u_step.device)

        x_new = self.dA.unsqueeze(0) * x_prev + u_step.unsqueeze(-1)
        y_step = torch.einsum('h n, b h n -> b h', self.dC, x_new).real.float()
        y_step = y_step + u_step * self.D
        return y_step, x_new


class S4Block(nn.Module):
    """ S4构建块, 类似于Transformer Encoder层 """

    def __init__(self, d_model, d_state=64, l_max=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.s4 = S4D(d_model, d_state, l_max)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, L, H)
        # Pre-Norm 结构
        x = x + self.dropout(self.s4(self.norm1(x).transpose(-1, -2)).transpose(-1, -2))
        x = x + self.ffn(self.norm2(x))
        return x

    def step(self, u_step, x_prev):
        # 循环模式下的单步前向传播
        y_s4, x_new = self.s4.step(self.norm1(u_step), x_prev)
        u_step = u_step + self.dropout(y_s4)
        u_step = u_step + self.ffn(self.norm2(u_step))
        return u_step, x_new


class S4DModel(nn.Module):
    """ 最终的S4D模型 """

    def __init__(self, input_channels, output_channels, d_model, n_layers, d_state=64, l_max=1024, dropout=0.1,
                 fusion_mode: str = 'add',
                 use_voltage_filter: bool = False):
        super(S4DModel, self).__init__()
        self.d_model = d_model
        self.fusion_mode = fusion_mode
        self.use_voltage_filter = use_voltage_filter  # <-- 保存参数
        self.output_channels = output_channels  # <-- 保存通道数

        self.input_proj = nn.Linear(input_channels, d_model)
        self.state_conditioner = nn.Linear(output_channels, d_model)

        self.s4_blocks = nn.ModuleList([
            S4Block(d_model, d_state, l_max, dropout) for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(d_model, output_channels)

        if self.use_voltage_filter:
            self.voltage_to_spike_filter = nn.Sequential(
                # --- ↓↓↓ 将 in_channels 从 1 修改为 2 ↓↓↓ ---
                # <--- 修改: 使用 'causal'  padding 来防止数据泄漏 ---
                nn.Conv1d(in_channels=2, out_channels=8, kernel_size=256, padding='causal', bias=False),
                nn.ReLU(),
                nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1)
            )

    def forward(self, stimulus_seq, initial_state, target_v1: torch.Tensor = None):
        """ 并行模式 (用于训练) """
        # B, C, L -> B, L, C
        stimulus_seq = stimulus_seq.transpose(-1, -2)

        # 1. 输入投影和状态调节
        stim_features = self.input_proj(stimulus_seq)
        if self.fusion_mode == 'add':
            state_embedding = self.state_conditioner(initial_state)
            fused_features = stim_features + state_embedding.unsqueeze(1)
        elif self.fusion_mode == 'ablate':
            fused_features = stim_features
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        # 2. 通过S4核心
        x = fused_features
        for block in self.s4_blocks:
            x = block(x)

        # 3. 输出投影
        prediction_transposed = self.output_proj(x)
        base_prediction = prediction_transposed.transpose(-1, -2)

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

    @torch.no_grad()
    def generate(self, stimulus_seq, initial_state):
        """ 自回归模式 (用于推理) """
        self.eval()
        B, C, L = stimulus_seq.shape

        # 准备输入和状态
        stimulus_seq = stimulus_seq.transpose(-1, -2)
        stim_features = self.input_proj(stimulus_seq)
        state_embedding = self.state_conditioner(initial_state)
        fused_features = stim_features + state_embedding.unsqueeze(1)

        # 初始化所有S4层的隐藏状态
        s4_states = [None] * len(self.s4_blocks)

        predictions = []
        for i in range(L):
            x = fused_features[:, i, :]  # 取当前时间步的输入

            # 逐层通过S4Block的step方法
            for j, block in enumerate(self.s4_blocks):
                x, s4_states[j] = block.step(x, s4_states[j])

            # 输出投影
            out_step = self.output_proj(x)
            predictions.append(out_step)

        predictions = torch.stack(predictions, dim=1)  # (B, L, C)
        raw_output = predictions.transpose(-1, -2)  # (B, C, L)

        # 1. 分离原始的电压和脉冲输出
        num_voltage_channels = self.output_channels - 1
        pred_voltage_part_raw = raw_output[:, :num_voltage_channels, :]
        pred_spike_logits_raw = raw_output[:, num_voltage_channels:, :]

        # 2. 只对电压部分应用1.1倍的Sigmoid激活
        activated_voltage = torch.sigmoid(pred_voltage_part_raw) * 1.1

        # 3. 组合已激活的电压和原始的脉冲logits
        return torch.cat([activated_voltage, pred_spike_logits_raw], dim=1)


# =============================================================================
#                               示例用法
# =============================================================================
if __name__ == '__main__':
    # --- 模型超参数定义 ---
    INPUT_CHANNELS = 20
    OUTPUT_CHANNELS = 7
    D_MODEL = 128
    N_LAYERS = 4
    D_STATE = 64
    SEQ_LENGTH = 1024  # S4D可以处理更长的序列，这里仅为示例

    # --- 实例化模型 ---
    print("正在实例化S4D模型...")
    model = S4DModel(
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        l_max=SEQ_LENGTH
    )
    print("模型实例化成功！")

    # --- 创建模拟输入数据 ---
    BATCH_SIZE = 8
    dummy_stimulus = torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQ_LENGTH)
    dummy_state = torch.randn(BATCH_SIZE, OUTPUT_CHANNELS)

    # --- 1. 训练模式测试 (并行卷积) ---
    print("\n--- 1. 训练模式 (并行) 测试 ---")
    output_train = model.forward(dummy_stimulus, dummy_state)
    print(f"  - 训练模式输出形状: {output_train.shape}")
    assert output_train.shape == (BATCH_SIZE, OUTPUT_CHANNELS, SEQ_LENGTH)
    print("  - 形状验证通过！")

    # --- 2. 推理模式测试 (自回归) ---
    print("\n--- 2. 推理模式 (自回归) 测试 ---")
    output_generate = model.generate(dummy_stimulus, dummy_state)
    print(f"  - 推理模式输出形状: {output_generate.shape}")
    assert output_generate.shape == (BATCH_SIZE, OUTPUT_CHANNELS, SEQ_LENGTH)
    print("  - 形状验证通过！")

    # 验证两种模式输出是否接近 (对于线性S4D，理论上应非常接近)
    # print(f"  - 训练与推理模式输出差异: {torch.mean(torch.abs(output_train - output_generate)):.6f}")

    # --- 打印模型参数量 ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params / 1e6:.2f} M")