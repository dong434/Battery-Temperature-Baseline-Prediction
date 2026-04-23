import os
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from model import BatteryTemperatureLSTM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
TRAIN_VAL_PLOTS_DIR = os.path.join(RESULTS_DIR, 'train_val_plots')
BEIJING_TZ = ZoneInfo('Asia/Shanghai')


def _set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(config.deterministic)
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)


def _ensure_dirs():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(TRAIN_VAL_PLOTS_DIR, exist_ok=True)


def _find_latest_checkpoint_for_seed(seed):
    if not os.path.exists(CHECKPOINTS_DIR):
        return None
    prefix = f'last_checkpoint_seed{seed}_'
    candidates = [
        f for f in os.listdir(CHECKPOINTS_DIR)
        if f.startswith(prefix) and f.endswith('.pth')
    ]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(CHECKPOINTS_DIR, candidates[-1])


def train_model(
    seed=config.seed,
    file_dir=config.data_path,
    batch_size=config.batch_size,
    lr=config.lr,
    epochs=config.epochs,
    weight_decay=config.weight_decay,
    scheduler_factor=config.scheduler_factor,
    scheduler_patience=config.scheduler_patience,
    scheduler_min_lr=config.scheduler_min_lr,
    scheduler_threshold=config.scheduler_threshold,
    scheduler_cooldown=config.scheduler_cooldown,
    resume_training=False,
):
    _set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ensure_dirs()

    run_id = datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')
    best_model_path = os.path.join(CHECKPOINTS_DIR, f'best_model_seed{seed}_{run_id}.pth')
    last_checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'last_checkpoint_seed{seed}_{run_id}.pth')

    print(f'开始加载 (seed={seed})')
    x_train_path = os.path.join(file_dir, 'train_x.pt')
    y_train_path = os.path.join(file_dir, 'train_y.pt')
    x_val_path = os.path.join(file_dir, 'evaluate_x.pt')
    y_val_path = os.path.join(file_dir, 'evaluate_y.pt')
    scaler_path = os.path.join(file_dir, 'train_scaler.gz')

    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError(f'训练张量数据未找到: {x_train_path}, {y_train_path}')
    if not os.path.exists(x_val_path) or not os.path.exists(y_val_path):
        raise FileNotFoundError(f'验证张量数据未找到: {x_val_path}, {y_val_path}')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'训练集 scaler 未找到: {scaler_path}')

    x_train = torch.load(x_train_path, weights_only=True)
    y_train = torch.load(y_train_path, weights_only=True)
    x_val = torch.load(x_val_path, weights_only=True)
    y_val = torch.load(y_val_path, weights_only=True)
    y_scaler = joblib.load(scaler_path)['y_scaler']

    train_dataset = TensorDataset(x_train, y_train)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator)
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = x_train.shape[-1]
    model = BatteryTemperatureLSTM(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        cooldown=scheduler_cooldown,
        min_lr=scheduler_min_lr,
    )

    start_epoch = 0
    best_val_mae_c = float('inf')

    if resume_training:
        latest_ckpt = _find_latest_checkpoint_for_seed(seed)
        if latest_ckpt is not None:
            checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = int(checkpoint.get('epoch', 0))
            best_val_mae_c = float(checkpoint.get('best_val_mae_c', best_val_mae_c))
            print(f'已恢复训练: checkpoint={os.path.basename(latest_ckpt)}, start_epoch={start_epoch}')

    train_losses_scaled = []
    val_losses_scaled = []
    val_mae_c_list = []

    for local_epoch in range(1, epochs + 1):
        global_epoch = start_epoch + local_epoch
        train_running_loss = 0.0
        val_running_loss = 0.0
        val_preds_scaled = []
        val_targets_scaled = []

        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).view(-1, 1)

            output = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_max_norm)
            optimizer.step()

            train_running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for v_batch_x, v_batch_y in val_loader:
                v_batch_x = v_batch_x.to(device)
                v_batch_y = v_batch_y.to(device).view(-1, 1)

                v_output = model(v_batch_x)
                v_loss = criterion(v_output, v_batch_y)
                val_running_loss += v_loss.item()

                val_preds_scaled.append(v_output.cpu().numpy())
                val_targets_scaled.append(v_batch_y.cpu().numpy())

        train_avg_loss_scaled = train_running_loss / len(train_loader)
        val_avg_loss_scaled = val_running_loss / len(val_loader)

        val_preds_scaled = np.concatenate(val_preds_scaled, axis=0)
        val_targets_scaled = np.concatenate(val_targets_scaled, axis=0)
        val_preds_c = y_scaler.inverse_transform(val_preds_scaled)
        val_targets_c = y_scaler.inverse_transform(val_targets_scaled)
        val_mae_c = float(np.mean(np.abs(val_preds_c - val_targets_c)))

        train_losses_scaled.append(train_avg_loss_scaled)
        val_losses_scaled.append(val_avg_loss_scaled)
        val_mae_c_list.append(val_mae_c)

        if val_mae_c < best_val_mae_c:
            best_val_mae_c = val_mae_c
            torch.save(model.state_dict(), best_model_path)
            print(f'本次最优更新(seed={seed}): epoch={global_epoch}, best_val_mae_c={best_val_mae_c:.6f}')

        torch.save(
            {
                'epoch': global_epoch,
                'seed': seed,
                'run_id': run_id,
                'best_val_mae_c': best_val_mae_c,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            last_checkpoint_path,
        )

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_mae_c)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            print(f'学习率衰减触发(seed={seed}): {prev_lr:.6f} -> {current_lr:.6f}')

        if global_epoch % 5 == 0:
            print(
                f'Epoch:{global_epoch} '
                f'train_loss_scaled:{train_avg_loss_scaled:.6f} '
                f'val_loss_scaled:{val_avg_loss_scaled:.6f} '
                f'val_mae_c:{val_mae_c:.4f} '
                f'best_val_mae_c:{best_val_mae_c:.4f} '
                f'lr:{current_lr:.6f}'
            )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses_scaled, 'r-', label='train_loss_scaled')
    plt.plot(val_losses_scaled, 'b--', label='val_loss_scaled')
    plt.title(f'Scaled Loss Curve (seed={seed})', fontsize=12)
    plt.xlabel('local epochs', fontsize=11)
    plt.ylabel('loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_mae_c_list, 'g-', label='val_mae_c')
    plt.title(f'Unscaled MAE Curve (seed={seed})', fontsize=12)
    plt.xlabel('local epochs', fontsize=11)
    plt.ylabel('mae (Celsius)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    curve_name = f'train_val_seed{seed}_{run_id}.png'
    curve_path = os.path.join(TRAIN_VAL_PLOTS_DIR, curve_name)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'训练完成(seed={seed})')
    print(f'本次最佳模型: {best_model_path}')
    print(f'本次最后断点: {last_checkpoint_path}')
    print(f'train/val 曲线: {curve_path}')


if __name__ == '__main__':
    train_model(seed=config.seed, resume_training=False)
