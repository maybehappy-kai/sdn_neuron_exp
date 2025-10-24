# analyze_loss.py

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import os # <--- 1. 确保导入 os 模块

# --- 从 train.py 复制必要的类和函数 ---
from train import get_datasets, TCNModel, S4DModel


def analyze_loss_magnitudes(args):
    """
    分析并打印在一个 epoch 内电压和脉冲损失的平均量级。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    expanded_data_dir = os.path.expanduser(args.data_dir)

    # --- 修改 1: 捕获动态通道数 ---
    # 注意：现在 get_datasets 返回5个值
    train_dataset, _, _, input_channels, output_channels = get_datasets(
        data_dir=expanded_data_dir,
        dataset_name=args.dataset_name,
        seq_len=args.seq_len
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"检测到输入通道: {input_channels}, 输出通道: {output_channels}")

    # --- 修改 2: 使用动态通道数初始化模型 ---
    if args.model == 'tcn':
        model = TCNModel(
            input_channels=input_channels, output_channels=output_channels,
            num_hidden_channels=[args.hidden_channels] * args.num_levels,
            input_kernel_size=args.input_kernel_size,
            tcn_kernel_size=args.tcn_kernel_size,
            dropout=0.2
        ).to(device)
    else:  # S4D
        model = S4DModel(
            input_channels=input_channels, output_channels=output_channels,
            d_model=args.d_model, n_layers=args.n_layers,
            d_state=args.d_state, l_max=args.seq_len, dropout=0.2
        ).to(device)

    model.train()

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    mse_losses = []
    bce_losses = []

    print(f"Analyzing loss magnitudes for one epoch on '{args.dataset_name}' dataset...")

    # --- 修改 3: 动态计算损失 ---
    num_voltage_channels = output_channels - 1
    with torch.no_grad():  # 无需计算梯度
        for v0_chunk, v1_context, v1_target in train_loader:
            v0_chunk, v1_context, v1_target = v0_chunk.to(device), v1_context.to(device), v1_target.to(device)

            predictions = model(v0_chunk, v1_context)

            loss_mse = criterion_mse(predictions[:, :num_voltage_channels, :], v1_target[:, :num_voltage_channels, :])
            loss_bce = criterion_bce(predictions[:, num_voltage_channels, :].unsqueeze(1),
                                     v1_target[:, num_voltage_channels, :].unsqueeze(1))

            mse_losses.append(loss_mse.item())
            bce_losses.append(loss_bce.item())

    # ... (后续打印和计算推荐权重的代码无需修改) ...

    avg_mse = np.mean(mse_losses)
    avg_bce = np.mean(bce_losses)

    print("\n" + "="*50)
    print("Loss Magnitude Analysis Complete")
    print(f"  - Average MSE Loss (Voltage): {avg_mse:.6f}")
    print(f"  - Average BCE Loss (Spikes):  {avg_bce:.6f}")

    # 计算推荐权重
    if avg_mse > 0:
        recommended_voltage_weight = avg_bce / avg_mse
        print(f"\nTo balance these, the MSE loss should be up-weighted.")
        print(f"Recommended Ratio (spike_weight=1.0):")
        print(f"  --voltage_weight {recommended_voltage_weight:.4f}")
        print(f"  --spike_weight 1.0")
    else:
        print("\nMSE loss is zero, cannot recommend a weight.")
    print("="*50)


if __name__ == '__main__':
    # --- 新增: 动态扫描数据集 ---
    import glob

    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--data_dir', type=str, default='~/Data/neuron_data')
    temp_args, _ = temp_parser.parse_known_args()
    expanded_data_dir = os.path.expanduser(temp_args.data_dir)

    available_datasets = []
    if os.path.isdir(expanded_data_dir):
        v0_files = glob.glob(os.path.join(expanded_data_dir, 'v0_*.npy'))
        for f in v0_files:
            dataset_name = os.path.basename(f)[3:-4]
            available_datasets.append(dataset_name)

    if not available_datasets:
        print(f"警告: 在 '{expanded_data_dir}' 中未找到任何数据集，将不限制 --dataset_name 参数。")
        dataset_choices = None
    else:
        dataset_choices = sorted(available_datasets)
        print(f"自动发现可用数据集: {dataset_choices}")
    # --- 修改结束 ---

    parser = argparse.ArgumentParser(description="Analyze loss magnitudes.")
    parser.add_argument('--model', type=str, required=True, choices=['tcn', 's4d'])
    parser.add_argument('--dataset_name', type=str, required=True, choices=dataset_choices,
                        help=f'Dataset to use. Discovered options: {dataset_choices}')  # <--- 使用动态列表
    parser.add_argument('--data_dir', type=str, default='~/Data/neuron_data')
    # --- 新增: 添加 train.py 中存在的其他通用参数 ---
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--fusion_mode', type=str, default='add', choices=['add', 'ablate'],
                        help="How to fuse initial state. 'add': add to stimulus features, 'ablate': ignore initial state.")
    parser.add_argument('--spike_weight', type=float, default=1.0, help='Weight for spike BCE loss.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay.')
    # --- 修改结束 ---

    # ... (后续动态设置模型参数的代码不变) ...

    # 根据模型动态设置
    temp_args, _ = parser.parse_known_args()
    if temp_args.model == 'tcn':
        parser.add_argument('--batch_size', type=int, default=4096)
        parser.add_argument('--seq_len', type=int, default=1024)
        parser.add_argument('--hidden_channels', type=int, default=48)
        parser.add_argument('--num_levels', type=int, default=4)
        parser.add_argument('--input_kernel_size', type=int, default=15)
        parser.add_argument('--tcn_kernel_size', type=int, default=5)
    elif temp_args.model == 's4d':
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--seq_len', type=int, default=4096)
        parser.add_argument('--d_model', type=int, default=128)
        parser.add_argument('--n_layers', type=int, default=4)
        parser.add_argument('--d_state', type=int, default=64)

    args = parser.parse_args()
    analyze_loss_magnitudes(args)