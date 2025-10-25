# ~/sdn_neuron_exp/train.py

import os
import glob
import argparse
import logging
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from models.tcn import TCNModel
from models.s4d import S4DModel


# --- 1. 自定义数据集类 (接受Numpy数组作为输入) ---
class NeuronDataset(Dataset):
    def __init__(self, v0_data, v1_data, seq_len):
        self.v0 = torch.from_numpy(v0_data).float()
        self.v1 = torch.from_numpy(v1_data).float()
        self.seq_len = seq_len
        self.total_timesteps = self.v0.shape[1]

    def __len__(self):
        return self.total_timesteps - self.seq_len + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.seq_len
        v0_chunk = self.v0[:, start_idx:end_idx]
        v1_context = self.v1[:, start_idx]
        v1_target = self.v1[:, start_idx + 1:end_idx + 1]
        return v0_chunk, v1_context, v1_target


# --- 2. 日志设置 ---
def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 输出到文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 输出到终端
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


# --- 3. 数据集准备 ---
def get_datasets(data_dir, dataset_name, seq_len):
    logging.info(f"Loading data for dataset: {dataset_name}")
    v0_path = os.path.join(data_dir, f"v0_{dataset_name}.npy")
    v1_path = os.path.join(data_dir, f"v1_{dataset_name}.npy")

    v0_full_raw = np.load(v0_path)
    v1_full_raw = np.load(v1_path)

    # --- 新增: 格式校验与维度捕获 ---
    if v0_full_raw.ndim != 3 or v0_full_raw.shape[0] != 1:
        raise ValueError(f"错误: v0 文件 '{os.path.basename(v0_path)}' 的形状应为 (1, m, T)，实际为 {v0_full_raw.shape}")
    if v1_full_raw.ndim != 3 or v1_full_raw.shape[0] != 1:
        raise ValueError(
            f"错误: v1 文件 '{os.path.basename(v1_path)}' 的形状应为 (1, n, T+1)，实际为 {v1_full_raw.shape}")

    # 去掉批次维度以匹配后续代码
    v0_full = v0_full_raw[0]
    v1_full = v1_full_raw[0]

    input_channels = v0_full.shape[0]
    output_channels = v1_full.shape[0]
    logging.info(f"自动检测到 -> 输入通道数: {input_channels}, 输出通道数: {output_channels}")
    # --- 修改结束 ---

    total_timesteps = v0_full.shape[1]

    # --- 新增: 创建一个包含完整数据的Dataset对象 ---
    full_dataset = NeuronDataset(v0_full, v1_full, seq_len)

    # 按 6:1:3 的比例在时间轴上分割
    train_end_idx = int(total_timesteps * 0.6)
    val_end_idx = int(total_timesteps * 0.7)

    # 分割Numpy数组，确保无重叠
    v0_train, v1_train = v0_full[:, :train_end_idx], v1_full[:, :train_end_idx + 1]
    v0_val, v1_val = v0_full[:, train_end_idx:val_end_idx], v1_full[:, train_end_idx:val_end_idx + 1]
    v0_test, v1_test = v0_full[:, val_end_idx:], v1_full[:, val_end_idx:]

    train_dataset = NeuronDataset(v0_train, v1_train, seq_len)
    val_dataset = NeuronDataset(v0_val, v1_val, seq_len)
    test_dataset = NeuronDataset(v0_test, v1_test, seq_len)

    logging.info(
        f"Datasets created. Train: {len(train_dataset.v0[0])}, Val: {len(val_dataset.v0[0])}, Test: {len(test_dataset.v0[0])} timesteps.")
    # --- 修改: 额外返回 full_dataset ---
    return train_dataset, val_dataset, test_dataset, full_dataset, input_channels, output_channels


# --- 4. 性能指标计算 ---
def calculate_metrics(predictions, targets, output_channels, optimal_threshold=None):
    metrics = {}

    # 分离电压和脉冲 (动态地)
    num_voltage_channels = output_channels - 1
    pred_voltage = predictions[:, :num_voltage_channels, :]
    pred_spike_logits = predictions[:, num_voltage_channels, :]
    target_voltage = targets[:, :num_voltage_channels, :]
    target_spike = targets[:, num_voltage_channels, :]

    # 方差解释率 (VE)
    ss_res = torch.sum((target_voltage - pred_voltage) ** 2)
    ss_tot = torch.sum((target_voltage - torch.mean(target_voltage)) ** 2)
    metrics['voltage_ve'] = (1 - ss_res / ss_tot).item()

    # --- ↓↓↓ 最佳阈值计算 (已修正) ↓↓↓ ---

    # 1. 获取概率
    spike_probs = torch.sigmoid(pred_spike_logits)

    # 2. 准备ROC曲线所需的数据
    y_true_np = target_spike.cpu().numpy().ravel()

    # 3. 仅在未提供阈值时才计算 (例如在验证集上)
    if optimal_threshold is None:
        y_scores_np = spike_probs.cpu().numpy().ravel()

        # 确保至少有一个正样本和一个负样本，否则roc_curve会报错
        if len(np.unique(y_true_np)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true_np, y_scores_np)

            # 找到最佳阈值 (使用Youden's J statistic: tpr - fpr)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            # 处理可能的nan值（虽然罕见）
            if np.isnan(optimal_threshold):
                optimal_threshold = 0.5
        else:
            # 如果所有样本都是同一个类别，则无法计算ROC，回退到0.5
            optimal_threshold = 0.5

    # --- ↑↑↑ 修正结束 (已删除重复代码) ↑↑↑ ---

    metrics['spike_threshold'] = float(optimal_threshold)

    # 精确率, 召回率, F1 (使用最佳阈值)
    preds_binary = (spike_probs > optimal_threshold).float()

    tp = torch.sum(preds_binary * target_spike)
    fp = torch.sum(preds_binary * (1 - target_spike))
    fn = torch.sum((1 - preds_binary) * target_spike)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    metrics['spike_precision'] = precision.item()
    metrics['spike_recall'] = recall.item()
    metrics['spike_f1'] = f1.item()

    return metrics, float(optimal_threshold)


# --- 5. 评估函数 (自回归生成) ---
# [最终修正版 v3 - 简洁正确]
@torch.no_grad()
def evaluate(model, dataset, device, seq_len, overlap_size, output_channels, return_predictions=False, optimal_threshold=None):
    model.eval()

    num_voltage_channels = output_channels - 1  # <--- 在函数开头定义

    if overlap_size >= seq_len:
        raise ValueError("overlap_size must be smaller than seq_len.")

    stride = seq_len - overlap_size

    full_v0 = dataset.v0.unsqueeze(0).to(device)
    full_v1_target = dataset.v1.unsqueeze(0).to(device)
    total_timesteps = dataset.total_timesteps

    current_state = full_v1_target[:, :, 0]
    all_predictions = []

    # 第一次预测
    v0_chunk_first = full_v0[:, :, :seq_len]
    predicted_chunk_first = model(v0_chunk_first, current_state)
    # ★ 在这里加入钳制 ★
    predicted_chunk_first[:, :num_voltage_channels, :] = torch.clamp(predicted_chunk_first[:, :num_voltage_channels, :], 0.0, 1.0)
    all_predictions.append(predicted_chunk_first)

    start_pos = stride
    # ★ 状态更新应基于钳制后的值 ★
    current_state = predicted_chunk_first[:, :, stride - 1]

    # 循环处理
    while start_pos < total_timesteps - seq_len:
        v0_chunk = full_v0[:, :, start_pos: start_pos + seq_len]
        predicted_chunk = model(v0_chunk, current_state)
        # ★ 在这里加入钳制 ★
        predicted_chunk[:, :num_voltage_channels, :] = torch.clamp(predicted_chunk[:, :num_voltage_channels, :], 0.0, 1.0)

        new_prediction_part = predicted_chunk[:, :, -stride:]
        all_predictions.append(new_prediction_part)

        # ★ 状态更新应基于钳制后的值 ★
        current_state = new_prediction_part[:, :, -1]
        start_pos += stride

    # 处理末尾
    if start_pos < total_timesteps:
        last_chunk_len = total_timesteps - start_pos
        v0_chunk_last = full_v0[:, :, start_pos:]
        if v0_chunk_last.shape[2] < seq_len:
            padding = torch.zeros((1, v0_chunk_last.shape[1], seq_len - v0_chunk_last.shape[2]), device=device)
            v0_chunk_last = torch.cat([v0_chunk_last, padding], dim=2)

        predicted_chunk_last = model(v0_chunk_last, current_state)
        # ★ 在这里加入钳制 ★
        predicted_chunk_last[:, :num_voltage_channels, :] = torch.clamp(predicted_chunk_last[:, :num_voltage_channels, :], 0.0, 1.0)

        # [修正后]
        all_predictions.append(predicted_chunk_last[:, :, overlap_size:last_chunk_len])

    # 最终拼接
    final_prediction = torch.cat(all_predictions, dim=2)
    final_target = full_v1_target[:, :, 1:final_prediction.shape[2] + 1]

    # <--- 修改: 传递阈值参数 ---
    metrics, optimal_threshold = calculate_metrics(final_prediction, final_target, output_channels,
                                                   optimal_threshold=optimal_threshold)

    if return_predictions:
        # <--- 修改: 额外返回阈值 ---
        return metrics, final_prediction, final_target, optimal_threshold
    else:
        # <--- 修改: 额外返回阈值 ---
        return metrics, optimal_threshold


def plot_test_results(predictions, targets, output_path, output_channels, optimal_threshold):
    """
    在测试集上生成并保存可视化对比图。

    Args:
        predictions (torch.Tensor): 模型的完整预测输出 (B, C, L)。
        targets (torch.Tensor): 完整的真实目标 (B, C, L)。
        output_path (str): 图片保存的完整路径 (包含文件名)。
    """
    # 将Tensor移动到CPU并转换为Numpy数组
    preds_np = predictions.squeeze(0).cpu().numpy()
    targets_np = targets.squeeze(0).cpu().numpy()

    num_voltage_channels = output_channels - 1
    soma_channel_index = num_voltage_channels - 1

    # --- 1. 提取最后一个电压通道 (通常是胞体电位) ---
    soma_pred = preds_np[num_voltage_channels - 1, :]
    soma_true = targets_np[num_voltage_channels - 1, :]
    timesteps = np.arange(len(soma_true))

    # --- 2. 提取脉冲数据 (最后一个通道) ---
    spike_logits_pred = preds_np[num_voltage_channels, :]
    spike_prob_pred = 1 / (1 + np.exp(-spike_logits_pred))  # Sigmoid激活
    spike_binary_pred = (spike_prob_pred > optimal_threshold).astype(int)
    spike_true = targets_np[num_voltage_channels, :].astype(int)

    # --- 3. 计算 TP, FP, FN ---
    tp = np.where((spike_binary_pred == 1) & (spike_true == 1))[0]
    fp = np.where((spike_binary_pred == 1) & (spike_true == 0))[0]
    fn = np.where((spike_binary_pred == 0) & (spike_true == 1))[0]

    # --- 4. 绘图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Final Test Set Evaluation', fontsize=16)

    # 子图1: 胞体电位对比
    ax1.plot(timesteps, soma_true, label='Ground Truth Voltage', color='royalblue', linewidth=2)
    ax1.plot(timesteps, soma_pred, label='Predicted Voltage', color='darkorange', linestyle='--', linewidth=1.5)
    ax1.set_title(f'Soma Membrane Potential (Channel {soma_channel_index})')  # <--- 修改这里
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 子图2: 脉冲事件分类
    ax2.set_title('Spike Event Analysis (TP, FP, FN)')
    ax2.vlines(tp, ymin=0, ymax=1, color='green', alpha=0.7, label=f'TP ({len(tp)})')
    ax2.vlines(fp, ymin=0, ymax=1, color='red', alpha=0.7, label=f'FP ({len(fp)})')
    ax2.vlines(fn, ymin=0, ymax=1, color='orange', alpha=0.7, label=f'FN ({len(fn)})')
    ax2.set_xlabel('Time Step')
    ax2.set_yticks([])  # 隐藏y轴刻度
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图像
    plt.savefig(output_path)
    plt.close(fig)
    logging.info(f"Test result visualization saved to {output_path}")


# --- 6. 训练函数 ---
def train_one_epoch(model, loader, optimizer, criterion_mse, criterion_bce, device, output_channels):
    model.train()
    total_loss = 0
    num_voltage_channels = output_channels - 1 # <--- 在循环外定义一次即可
    for v0_chunk, v1_context, v1_target in loader:
        v0_chunk, v1_context, v1_target = v0_chunk.to(device), v1_context.to(device), v1_target.to(device)

        optimizer.zero_grad()
        predictions = model(v0_chunk, v1_context, v1_target)

        loss_mse = criterion_mse(predictions[:, :num_voltage_channels, :], v1_target[:, :num_voltage_channels, :])
        loss_bce = criterion_bce(predictions[:, num_voltage_channels, :].unsqueeze(1), v1_target[:, num_voltage_channels, :].unsqueeze(1))
        loss = args.voltage_weight * loss_mse + args.spike_weight * loss_bce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# --- 7. 主函数 ---
def main(args):
    # 创建输出目录和日志
    # 创建一个唯一的、信息丰富的运行ID
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_id = f"{args.model}_{args.dataset_name}_{timestamp}"

    # 根据模型类型，添加关键的超参数到ID中
    if args.model == 'tcn':
        run_id += f"_lr{args.lr}_h{args.hidden_channels}_l{args.num_levels}_k{args.input_kernel_size}"
    elif args.model == 's4d':
        run_id += f"_lr{args.lr}_d{args.d_model}_l{args.n_layers}"

    run_id += f"_fusion-{args.fusion_mode}"

    run_id += f"_sw{args.spike_weight}"

    # --- ↓↓↓ 在此处添加新代码 ↓↓↓ ---
    # 将是否使用电压滤波器（voltage filter）的信息加入命名
    if args.use_voltage_filter:
        run_id += "_vf-on"
    else:
        run_id += "_vf-off"
    # --- ↑↑↑ 修改结束 ↑↑↑ ---

    run_id += f"_v-act-{args.voltage_activation}"

    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(os.path.join(run_dir, 'run.log'))

    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Arguments: {vars(args)}")

    # 设置设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    # --- 修改: 捕获 full_dataset ---
    train_dataset, val_dataset, test_dataset, full_dataset, input_channels, output_channels = get_datasets(
        os.path.expanduser(args.data_dir), args.dataset_name, args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 实例化模型
    if args.model == 'tcn':
        model = TCNModel(
            input_channels=input_channels, output_channels=output_channels,  # <--- 使用动态通道数
            num_hidden_channels=[args.hidden_channels] * args.num_levels,
            input_kernel_size=args.input_kernel_size,
            tcn_kernel_size=args.tcn_kernel_size,
            dropout=args.dropout,
            fusion_mode=args.fusion_mode,
            use_voltage_filter=args.use_voltage_filter,
            voltage_activation=args.voltage_activation
        ).to(device)
    elif args.model == 's4d':
        model = S4DModel(
            input_channels=input_channels, output_channels=output_channels,  # <--- 使用动态通道数
            d_model=args.d_model, n_layers=args.n_layers,
            d_state=args.d_state, l_max=args.seq_len, dropout=args.dropout, fusion_mode=args.fusion_mode, use_voltage_filter=args.use_voltage_filter,
            voltage_activation=args.voltage_activation
        ).to(device)

    logging.info(f"Model: {args.model}")
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 损失函数和优化器
    # 损失函数和优化器
    spike_channel_index = output_channels - 1
    spikes = train_dataset.v1[spike_channel_index, :]
    pos_weight_val = (len(spikes) - spikes.sum()) / spikes.sum() if spikes.sum() > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练循环
    # 1. 初始化两个最佳指标，而不是一个
    best_val_ve = -float('inf')
    best_val_f1 = -float('inf')
    patience_counter = 0
    best_val_threshold = 0.5  # 初始化一个默认值

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_mse, criterion_bce, device,
                                     output_channels)  # <--- 传递
        val_metrics, val_threshold = evaluate(model, val_dataset, device, args.seq_len, args.overlap_size, output_channels)  # <--- 传递

        scheduler.step()

        logging.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.6f} | "
                     f"Val VE: {val_metrics['voltage_ve']:.4f} | Val F1: {val_metrics['spike_f1']:.4f} | "
                     f"Val Threshold: {val_threshold:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 2. 检查模型是否满足新的保存条件：一方提升，另一方下降不超过其最佳值的10%
        current_ve = val_metrics['voltage_ve']
        current_f1 = val_metrics['spike_f1']

        # 定义可接受的退化阈值（10% of the absolute best value）
        # 即使best_val_ve为负，此计算也有效
        ve_degradation_threshold = best_val_ve - abs(best_val_ve) * 0.1
        f1_degradation_threshold = best_val_f1 - abs(best_val_f1) * 0.1  # F1>=0, abs是安全的

        # 检查两个主要条件
        ve_improves_while_f1_is_ok = (current_ve > best_val_ve) and (current_f1 >= f1_degradation_threshold)
        f1_improves_while_ve_is_ok = (current_f1 > best_val_f1) and (current_ve >= ve_degradation_threshold)

        if ve_improves_while_f1_is_ok or f1_improves_while_ve_is_ok:
            improvement_message = []
            # 分别更新各自的最佳值
            if current_ve > best_val_ve:
                best_val_ve = current_ve
                improvement_message.append(f"VE to {best_val_ve:.4f}")
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                improvement_message.append(f"F1 to {best_val_f1:.4f}")

            # 重置耐心计数器并保存模型
            patience_counter = 0
            best_val_threshold = val_threshold  # <--- 捕获与最佳模型对应的阈值
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
            logging.info(f"New best model found ({', '.join(improvement_message)}). Saving model.")
        else:
            patience_counter += 1
            logging.info(f"No improvement based on criteria. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logging.info("Early stopping triggered.")
            break

    # 最终测试
    logging.info("Loading best model for final testing...")

    # --- ★★★ 关键修正: 在加载前重新实例化模型 ★★★
    # 这确保了模型的架构与保存的权重文件完全匹配，无论之前的循环中发生了什么。
    if args.model == 'tcn':
        model = TCNModel(
            input_channels=input_channels, output_channels=output_channels, # <--- 修改这里
            num_hidden_channels=[args.hidden_channels] * args.num_levels,
            input_kernel_size=args.input_kernel_size,
            tcn_kernel_size=args.tcn_kernel_size,
            dropout=args.dropout,
            fusion_mode=args.fusion_mode,
            use_voltage_filter = args.use_voltage_filter,
            voltage_activation=args.voltage_activation
        ).to(device)
    elif args.model == 's4d':
        model = S4DModel(
            input_channels=input_channels, output_channels=output_channels, # <--- 修改这里
            d_model=args.d_model, n_layers=args.n_layers,
            d_state=args.d_state, l_max=args.seq_len, dropout=args.dropout, fusion_mode=args.fusion_mode, use_voltage_filter=args.use_voltage_filter,
            voltage_activation=args.voltage_activation
        ).to(device)

    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))

    # +++ 修改部分 +++
    # <--- 修改: 捕获测试集指标和最佳阈值 ---
    # 调用evaluate时，请求返回预测结果
    test_metrics, test_preds, test_targets, test_threshold = evaluate(
        model, test_dataset, device, args.seq_len, args.overlap_size, output_channels, return_predictions=True,
        optimal_threshold=best_val_threshold  # <--- 修改: 传入在验证集上找到的最佳阈值
    )
    # <--- 修改: 打印测试集阈值 ---
    logging.info(f"Final Test Metrics: {test_metrics}")
    logging.info(f"Final Test Optimal Threshold: {test_threshold:.4f}")

    # 调用新的可视化函数
    # <--- 修改: 传入最佳阈值用于绘图 ---
    plot_test_results(
        predictions=test_preds,
        targets=test_targets,
        output_path=os.path.join(run_dir, 'final_test_visualization.png'),
        output_channels=output_channels,
        optimal_threshold=test_threshold
    )
    # +++ 修改结束 +++

    # --- 新增: 在完整数据集上进行评估和可视化 ---
    logging.info("Evaluating on the FULL dataset for visualization...")
    # 注意: 完整数据集可能非常大，这步可能很慢并消耗大量VRAM
    try:
        # <--- 修改: 传入 best_val_threshold 并在 full_threshold 中捕获它 ---
        _, full_preds, full_targets, full_threshold = evaluate(
            model, full_dataset, device, args.seq_len, args.overlap_size, output_channels, return_predictions=True,
            optimal_threshold=best_val_threshold
        )

        # <--- 修改: 传入 full_threshold 用于绘图 ---
        plot_test_results(
            predictions=full_preds,
            targets=full_targets,
            output_path=os.path.join(run_dir, 'full_dataset_visualization.png'),
            output_channels=output_channels,
            optimal_threshold=full_threshold
        )
        logging.info(f"Full dataset visualization saved to {os.path.join(run_dir, 'full_dataset_visualization.png')}")

    except RuntimeError as e:
        # 捕获可能的 OOM (Out of Memory) 错误
        logging.error(f"Failed to evaluate or plot on FULL dataset (possibly OOM): {e}")
    # --- 新增结束 ---

    # 保存结果
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        # <--- 修改: 将 test_threshold 添加到结果文件中 ---
        results_data = {
            'args': vars(args),
            'test_metrics': test_metrics,
            'test_optimal_threshold': test_threshold
        }
        json.dump(results_data, f, indent=4)

    logging.info("Training finished.")


if __name__ == '__main__':
    # --- 新增: 动态扫描数据集 ---

    # 临时的 argparse 用来获取 data_dir，以便我们知道去哪里扫描
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--data_dir', type=str, default='~/Data/neuron_data')
    temp_args, _ = temp_parser.parse_known_args()
    expanded_data_dir = os.path.expanduser(temp_args.data_dir)

    # 扫描 v0_{name}.npy 文件并提取 name
    available_datasets = []
    if os.path.isdir(expanded_data_dir):
        v0_files = glob.glob(os.path.join(expanded_data_dir, 'v0_*.npy'))
        for f in v0_files:
            basename = os.path.basename(f)
            # 从 "v0_" 和 ".npy" 之间提取数据集名称
            dataset_name = basename[3:-4]
            available_datasets.append(dataset_name)

    if not available_datasets:
        # 如果找不到任何数据集，就允许任意字符串，后续让文件加载逻辑去报错
        print(f"警告: 在 '{expanded_data_dir}' 中未找到任何数据集 (v0_*.npy)，将不限制 --dataset_name 参数。")
        dataset_choices = None
    else:
        dataset_choices = sorted(available_datasets)
        print(f"自动发现可用数据集: {dataset_choices}")
    # --- 修改结束 ---
    parser = argparse.ArgumentParser(description="Train TCN or S4D models on neuron data.")
    # 通用参数
    parser.add_argument('--model', type=str, required=True, choices=['tcn', 's4d'], help='Model to train.')
    parser.add_argument('--dataset_name', type=str, required=True, choices=dataset_choices,
                        help=f'Dataset to use. Discovered options: {dataset_choices}')  # <--- 使用动态列表
    parser.add_argument('--data_dir', type=str, default='~/Data/neuron_data', help='Directory containing neuron data.')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save results.')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay.')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--overlap_size', type=int, default=256,
                        help='Overlap size for autoregressive generation window.')
    parser.add_argument('--fusion_mode', type=str, default='add', choices=['add', 'ablate'],
                        help="How to fuse initial state. 'add': add to stimulus features, 'ablate': ignore initial state.")
    parser.add_argument('--voltage_activation', type=str, default='linear', choices=['linear', 'custom_x2'],
                        help="Activation function for the voltage output head.")
    parser.add_argument('--use_voltage_filter', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Enable the biexponential filter from voltage to spike logits (e.g., --use_voltage_filter True).")

    # 根据模型动态设置batch size和seq_len
    temp_args, _ = parser.parse_known_args()
    if temp_args.model == 'tcn':
        parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for TCN.')
        parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length for TCN.')
        # TCN专属参数
        parser.add_argument('--hidden_channels', type=int, default=48)
        parser.add_argument('--num_levels', type=int, default=4)
        parser.add_argument('--input_kernel_size', type=int, default=15)
        parser.add_argument('--tcn_kernel_size', type=int, default=5)
        parser.add_argument('--voltage_weight', type=float, default=1.0,
                            help='Weight for voltage MSE loss (default for TCN).')
        parser.add_argument('--spike_weight', type=float, default=1.0, help='Weight for spike BCE loss.')
    elif temp_args.model == 's4d':
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size for S4D.')
        parser.add_argument('--seq_len', type=int, default=4096, help='Sequence length for S4D.')
        # S4D专属参数
        parser.add_argument('--d_model', type=int, default=128)
        parser.add_argument('--n_layers', type=int, default=4)
        parser.add_argument('--d_state', type=int, default=64)
        parser.add_argument('--voltage_weight', type=float, default=1.0,
                            help='Weight for voltage MSE loss (default for S4D).')
        parser.add_argument('--spike_weight', type=float, default=1.0, help='Weight for spike BCE loss.')

    args = parser.parse_args()
    main(args)