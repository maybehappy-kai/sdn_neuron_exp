# ~/sdn_neuron_exp/test_model_s4d.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
from models.s4d import S4DModel  # <-- 1. 导入S4DModel

# --- 1. 配置参数 ---
DATA_DIR = '~/Data/neuron_data'
V0_FILE = 'v0_small.npy'
V1_FILE = 'v1_small.npy'

# --- 2. 关键调整：使用更长的序列长度来发挥S4D的优势 ---
SEQ_LEN = 4096  # <-- 从1024增加到4096

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 2. 自定义数据集类 (与TCN版本完全相同，无需修改) ---
class NeuronDataset(Dataset):
    def __init__(self, v0_path, v1_path, seq_len):
        print(f"Loading data from {v0_path} and {v1_path}...")
        self.v0 = np.load(v0_path)
        self.v1 = np.load(v1_path)
        self.seq_len = seq_len
        assert self.v0.shape[2] + 1 == self.v1.shape[2], "v1 should have one more timestep than v0"
        self.total_timesteps = self.v0.shape[2]

    def __len__(self):
        return self.total_timesteps - self.seq_len + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.seq_len
        v0_chunk = torch.from_numpy(self.v0[0, :, start_idx:end_idx]).float()
        v1_context = torch.from_numpy(self.v1[0, :, start_idx]).float()
        v1_target = torch.from_numpy(self.v1[0, :, start_idx + 1:end_idx + 1]).float()
        return v0_chunk, v1_context, v1_target


# --- 3. 辅助函数：寻找最大batch_size (与TCN版本完全相同) ---
def find_max_batch_size(model, dataset, start_batch_size=256):  # 起始点可以设小一些，因为序列更长
    print("\n--- Finding optimal batch size for S4D ---")
    batch_size = start_batch_size
    while batch_size > 0:
        try:
            loader = DataLoader(dataset, batch_size=batch_size)
            v0_chunk, v1_context, _ = next(iter(loader))
            model.to(DEVICE)
            v0_chunk = v0_chunk.to(DEVICE)
            v1_context = v1_context.to(DEVICE)
            _ = model(v0_chunk, v1_context)
            print(f"Successfully ran with batch_size = {batch_size}. Recommending this value.")
            del v0_chunk, v1_context
            torch.cuda.empty_cache()
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch_size = {batch_size} is too large (CUDA out of memory). Trying smaller.")
                torch.cuda.empty_cache()
                batch_size //= 2
            else:
                raise e
    print("Error: Could not find a suitable batch_size, even with size 1.")
    return 0


# --- 4. 主测试流程 ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    v0_full_path = os.path.expanduser(os.path.join(DATA_DIR, V0_FILE))
    v1_full_path = os.path.expanduser(os.path.join(DATA_DIR, V1_FILE))

    print("\n--- Data Loading & Splitting ---")
    full_dataset = NeuronDataset(v0_full_path, v1_full_path, SEQ_LEN)
    dataset_size = len(full_dataset)
    print(f"Dataset created with {dataset_size} samples for seq_len={SEQ_LEN}.")

    indices = list(range(dataset_size))
    split_train = int(np.floor(TRAIN_RATIO * dataset_size))
    split_val = int(np.floor(VAL_RATIO * dataset_size))
    train_indices, val_indices, test_indices = indices[:split_train], indices[
        split_train:split_train + split_val], indices[split_train + split_val:]

    train_dataset = Subset(full_dataset, train_indices)
    print(f"Data split: Train={len(train_dataset)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # --- 模型实例化 (使用S4DModel) ---
    print("\n--- S4D Model Initialization ---")
    model = S4DModel(
        input_channels=20, output_channels=7,
        d_model=128, n_layers=4, d_state=64, l_max=SEQ_LEN, dropout=0.1
    )
    print(f"S4D Model instantiated with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")

    RECOMMENDED_BATCH_SIZE = find_max_batch_size(model, train_dataset)
    if RECOMMENDED_BATCH_SIZE == 0:
        exit()

    print("\n--- End-to-End Pipeline Verification for S4D ---")
    model.to(DEVICE)
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=RECOMMENDED_BATCH_SIZE, shuffle=True)

    spikes = full_dataset.v1[0, 6, :]
    pos_weight_val = (len(spikes) - spikes.sum()) / spikes.sum()
    pos_weight_tensor = torch.tensor([pos_weight_val], device=DEVICE)
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("Fetching one batch of data...")
    v0_batch, v1_context_batch, v1_target_batch = next(iter(train_loader))

    v0_batch = v0_batch.to(DEVICE)
    v1_context_batch = v1_context_batch.to(DEVICE)
    v1_target_batch = v1_target_batch.to(DEVICE)
    print("Data moved to device.")

    # --- 验证训练模式 (forward) ---
    print("\n1. Verifying Training Mode (model.forward)...")
    predictions = model(v0_batch, v1_context_batch)
    print(f"  - Prediction output shape: {predictions.shape}")
    assert predictions.shape == v1_target_batch.shape, "Shape mismatch!"
    print("  - Prediction shape matches target shape. OK.")

    loss_mse = criterion_mse(predictions[:, :6, :], v1_target_batch[:, :6, :])
    loss_bce = criterion_bce(predictions[:, 6, :].unsqueeze(1), v1_target_batch[:, 6, :].unsqueeze(1))
    total_loss = loss_mse + loss_bce
    print(f"  - Calculated Total Loss: {total_loss.item():.6f}")

    print("  - Performing backward pass...")
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("  - Backward pass and optimizer step completed successfully.")

    # --- 3. 额外验证：推理模式 (generate) ---
    print("\n2. Verifying Inference Mode (model.generate)...")
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        generated_output = model.generate(v0_batch, v1_context_batch)
    print(f"  - Generated output shape: {generated_output.shape}")
    assert generated_output.shape == v1_target_batch.shape, "Shape mismatch in generate method!"
    print("  - Generated output shape is correct. OK.")

    print("\n[SUCCESS] The S4D pipeline is verified and works correctly in both modes!")