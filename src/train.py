import os
from datetime import datetime

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, 'best_battery_model.pth')
LAST_CHECKPOINT_PATH = os.path.join(RESULTS_DIR, 'last_checkpoint.pth')
BEST_METRIC_PATH = os.path.join(RESULTS_DIR, 'best_val_mse_c.txt')


def _load_best_metric_from_file(default=float('inf')):
    if not os.path.exists(BEST_METRIC_PATH):
        return default
    try:
        with open(BEST_METRIC_PATH, 'r', encoding='utf-8') as f:
            return float(f.read().strip())
    except Exception:
        return default


def _save_best_metric_to_file(best_val_mse_c):
    with open(BEST_METRIC_PATH, 'w', encoding='utf-8') as f:
        f.write(f'{best_val_mse_c:.12f}')


def train_model(file_dir=config.data_path,
                batch_size=config.batch_size,
                lr=config.lr,
                epochs=config.epochs,
                weight_decay=config.weight_decay,
                scheduler_factor=config.scheduler_factor,
                scheduler_patience=config.scheduler_patience,
                scheduler_min_lr=config.scheduler_min_lr,
                scheduler_threshold=config.scheduler_threshold,
                scheduler_cooldown=config.scheduler_cooldown,
                resume_training=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('开始加载')
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        min_lr=scheduler_min_lr
    )

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    start_epoch = 0
    best_val_mse_c = _load_best_metric_from_file(default=float('inf'))

    if resume_training and os.path.exists(LAST_CHECKPOINT_PATH):
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint.get('epoch', 0))
        best_val_mse_c = float(checkpoint.get('best_val_mse_c', best_val_mse_c))
        print(f'已恢复上次训练: start_epoch={start_epoch}, best_val_mse_c={best_val_mse_c:.6f}')
    else:
        print(f'从头开始训练: best_val_mse_c={best_val_mse_c:.6f}')

    train_losses_scaled = []
    val_losses_scaled = []
    val_mse_c_list = []

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
        val_mse_c = float(np.mean((val_preds_c - val_targets_c) ** 2))

        train_losses_scaled.append(train_avg_loss_scaled)
        val_losses_scaled.append(val_avg_loss_scaled)
        val_mse_c_list.append(val_mse_c)

        if val_mse_c < best_val_mse_c:
            best_val_mse_c = val_mse_c
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            _save_best_metric_to_file(best_val_mse_c)
            print(f'全局最优更新: epoch={global_epoch}, best_val_mse_c={best_val_mse_c:.6f}')

        # Save resumable training state every epoch.
        torch.save(
            {
                'epoch': global_epoch,
                'best_val_mse_c': best_val_mse_c,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            LAST_CHECKPOINT_PATH,
        )

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_mse_c)
        current_lr = optimizer.param_groups[0]['lr']
        lr_changed = current_lr < prev_lr

        if global_epoch % 5 == 0:
            print(
                f'Epoch:{global_epoch} '
                f'train_loss_scaled:{train_avg_loss_scaled:.6f} '
                f'val_loss_scaled:{val_avg_loss_scaled:.6f} '
                f'val_mse_c:{val_mse_c:.4f} '
                f'best_val_mse_c:{best_val_mse_c:.4f} '
                f'lr:{current_lr:.6f}'
            )
        if lr_changed:
            print(f'学习率衰减触发: {prev_lr:.6f} -> {current_lr:.6f}')

    print('训练完成')

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses_scaled, 'r-', label='train_loss_scaled')
    plt.plot(val_losses_scaled, 'b--', label='val_loss_scaled')
    plt.title('Scaled Loss Curve', fontsize=12)
    plt.xlabel('local epochs', fontsize=11)
    plt.ylabel('loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_mse_c_list, 'g-', label='val_mse_c')
    plt.title('Unscaled Loss Curve (Celsius MSE)', fontsize=12)
    plt.xlabel('local epochs', fontsize=11)
    plt.ylabel('mse (Celsius^2)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    curve_name = f'train_loss_curve_{run_id}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, curve_name), dpi=300, bbox_inches='tight')
    print(f'loss 曲线已保存为 {curve_name}')
    print(f'当前全局最优验证MSE(℃^2): {best_val_mse_c:.6f}')


if __name__ == '__main__':
    train_model(resume_training=False)
