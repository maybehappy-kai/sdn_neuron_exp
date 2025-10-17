# ~/sdn_neuron_exp/train.py

import os
import argparse
import logging
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

    v0_full = np.load(v0_path)[0]  # 去掉批次维度
    v1_full = np.load(v1_path)[0]

    total_timesteps = v0_full.shape[1]

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
    return train_dataset, val_dataset, test_dataset


# --- 4. 性能指标计算 ---
def calculate_metrics(predictions, targets):
    metrics = {}

    # 分离电压和脉冲
    pred_voltage = predictions[:, :6, :]
    pred_spike_logits = predictions[:, 6, :]
    target_voltage = targets[:, :6, :]
    target_spike = targets[:, 6, :]

    # 方差解释率 (VE)
    ss_res = torch.sum((target_voltage - pred_voltage) ** 2)
    ss_tot = torch.sum((target_voltage - torch.mean(target_voltage)) ** 2)
    metrics['voltage_ve'] = (1 - ss_res / ss_tot).item()

    # 精确率, 召回率, F1
    preds_binary = (torch.sigmoid(pred_spike_logits) > 0.5).float()
    tp = torch.sum(preds_binary * target_spike)
    fp = torch.sum(preds_binary * (1 - target_spike))
    fn = torch.sum((1 - preds_binary) * target_spike)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    metrics['spike_precision'] = precision.item()
    metrics['spike_recall'] = recall.item()
    metrics['spike_f1'] = f1.item()

    return metrics


# --- 5. 评估函数 (自回归生成) ---
# [最终修正版 v3 - 简洁正确]
@torch.no_grad()
def evaluate(model, dataset, device, seq_len, overlap_size):
    model.eval()

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
    predicted_chunk_first[:, :6, :] = torch.clamp(predicted_chunk_first[:, :6, :], 0.0, 1.0)
    all_predictions.append(predicted_chunk_first)

    start_pos = stride
    # ★ 状态更新应基于钳制后的值 ★
    current_state = predicted_chunk_first[:, :, stride - 1]

    # 循环处理
    while start_pos < total_timesteps - seq_len:
        v0_chunk = full_v0[:, :, start_pos: start_pos + seq_len]
        predicted_chunk = model(v0_chunk, current_state)
        # ★ 在这里加入钳制 ★
        predicted_chunk[:, :6, :] = torch.clamp(predicted_chunk[:, :6, :], 0.0, 1.0)

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
        predicted_chunk_last[:, :6, :] = torch.clamp(predicted_chunk_last[:, :6, :], 0.0, 1.0)

        # [修正后]
        all_predictions.append(predicted_chunk_last[:, :, overlap_size:last_chunk_len])

    # 最终拼接
    final_prediction = torch.cat(all_predictions, dim=2)
    final_target = full_v1_target[:, :, 1:final_prediction.shape[2] + 1]

    return calculate_metrics(final_prediction, final_target)


# --- 6. 训练函数 ---
def train_one_epoch(model, loader, optimizer, criterion_mse, criterion_bce, device):
    model.train()
    total_loss = 0
    for v0_chunk, v1_context, v1_target in loader:
        v0_chunk, v1_context, v1_target = v0_chunk.to(device), v1_context.to(device), v1_target.to(device)

        optimizer.zero_grad()
        predictions = model(v0_chunk, v1_context)

        loss_mse = criterion_mse(predictions[:, :6, :], v1_target[:, :6, :])
        loss_bce = criterion_bce(predictions[:, 6, :].unsqueeze(1), v1_target[:, 6, :].unsqueeze(1))
        loss = loss_mse + loss_bce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# --- 7. 主函数 ---
def main(args):
    # 创建输出目录和日志
    run_dir = os.path.join(args.output_dir, f"{args.model}_{args.dataset_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(os.path.join(run_dir, 'run.log'))

    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Arguments: {vars(args)}")

    # 设置设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    train_dataset, val_dataset, test_dataset = get_datasets(os.path.expanduser(args.data_dir), args.dataset_name,
                                                            args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 实例化模型
    if args.model == 'tcn':
        model = TCNModel(
            input_channels=20, output_channels=7,
            num_hidden_channels=[args.hidden_channels] * args.num_levels,
            input_kernel_size=args.input_kernel_size,
            tcn_kernel_size=args.tcn_kernel_size,
            dropout=args.dropout
        ).to(device)
    elif args.model == 's4d':
        model = S4DModel(
            input_channels=20, output_channels=7,
            d_model=args.d_model, n_layers=args.n_layers,
            d_state=args.d_state, l_max=args.seq_len, dropout=args.dropout
        ).to(device)

    logging.info(f"Model: {args.model}")
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 损失函数和优化器
    spikes = train_dataset.v1[6, :]
    pos_weight_val = (len(spikes) - spikes.sum()) / spikes.sum() if spikes.sum() > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练循环
    best_val_metric = -float('inf')  # 使用F1作为主要指标
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_mse, criterion_bce, device)
        val_metrics = evaluate(model, val_dataset, device, args.seq_len, args.overlap_size)

        scheduler.step()

        logging.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.6f} | "
                     f"Val VE: {val_metrics['voltage_ve']:.4f} | Val F1: {val_metrics['spike_f1']:.4f} | "
                     f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_metrics['spike_f1'] > best_val_metric:
            best_val_metric = val_metrics['spike_f1']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
            logging.info("Validation metric improved, saving best model.")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logging.info("Early stopping triggered.")
            break

    # 最终测试
    logging.info("Loading best model for final testing...")
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))
    test_metrics = evaluate(model, test_dataset, device, args.seq_len, args.overlap_size)
    logging.info(f"Final Test Metrics: {test_metrics}")

    # 保存结果
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump({'args': vars(args), 'test_metrics': test_metrics}, f, indent=4)

    logging.info("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train TCN or S4D models on neuron data.")
    # 通用参数
    parser.add_argument('--model', type=str, required=True, choices=['tcn', 's4d'], help='Model to train.')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['small', 'small_3x', 'small_30x'],
                        help='Dataset to use.')
    parser.add_argument('--data_dir', type=str, default='~/Data/neuron_data', help='Directory containing neuron data.')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save results.')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--overlap_size', type=int, default=256,
                        help='Overlap size for autoregressive generation window.')

    # 根据模型动态设置batch size和seq_len
    temp_args, _ = parser.parse_known_args()
    if temp_args.model == 'tcn':
        parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for TCN.')
        parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length for TCN.')
        # TCN专属参数
        parser.add_argument('--hidden_channels', type=int, default=48)
        parser.add_argument('--num_levels', type=int, default=4)
        parser.add_argument('--input_kernel_size', type=int, default=15)
        parser.add_argument('--tcn_kernel_size', type=int, default=5)
    elif temp_args.model == 's4d':
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size for S4D.')
        parser.add_argument('--seq_len', type=int, default=4096, help='Sequence length for S4D.')
        # S4D专属参数
        parser.add_argument('--d_model', type=int, default=128)
        parser.add_argument('--n_layers', type=int, default=4)
        parser.add_argument('--d_state', type=int, default=64)

    args = parser.parse_args()
    main(args)